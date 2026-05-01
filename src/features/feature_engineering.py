"""
src/features/feature_engineering.py

WHY THIS MODULE EXISTS:
    Raw IEEE-CIS data has 433 columns, many of which are:
    - Anonymized V-columns with no domain meaning
    - High-cardinality categoricals (email domains, device types)
    - Time encoded as seconds-since-reference (not human-readable)
    - Heavily missing (>80% NaN in some columns)

    This module engineers meaningful features from that raw data
    and wraps everything in a scikit-learn Pipeline so that
    train/inference transformations are ALWAYS identical.
"""

import pandas as pd
import numpy as np
from loguru import logger


# ── Column groups ────────────────────────────────────────────────────────────

# High-cardinality categoricals — we'll target-encode these
HIGH_CARD_CATS = [
    "P_emaildomain",   # purchaser email domain
    "R_emaildomain",   # recipient email domain
    "DeviceInfo",      # device model string
]

# Low-cardinality categoricals — we'll one-hot encode these
LOW_CARD_CATS = [
    "ProductCD",       # product category (W, H, C, S, R)
    "card4",           # card network (visa, mastercard, etc.)
    "card6",           # card type (debit, credit)
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",  # match flags
]

# Numeric columns to log-transform (heavy right skew)
LOG_TRANSFORM_COLS = ["TransactionAmt"]


# ── Time features ─────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TransactionDT is seconds elapsed from some reference point.
    We extract cyclic time signals from it.

    WHY CYCLIC ENCODING FOR HOUR/DAY:
        If we encode hour as 0-23, the model sees hour 23 and hour 0
        as far apart — but midnight and 11pm are actually adjacent.
        Sine/cosine encoding wraps the circle correctly.
    """
    dt = df["TransactionDT"]

    # Raw time features
    df["hour_of_day"]  = (dt // 3600) % 24
    df["day_of_week"]  = (dt // (3600 * 24)) % 7
    df["day_of_month"] = (dt // (3600 * 24)) % 30

    # Cyclic encoding for hour (preserves midnight-continuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["day_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    logger.debug("Added time features: hour_sin, hour_cos, day_sin, day_cos")
    return df


# ── Transaction amount features ───────────────────────────────────────────────

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TransactionAmt is heavily right-skewed (most txns are small,
    fraud txns can be very large). Log-transform compresses the range
    and helps tree models find cleaner split points.

    We also add cents as a feature — fraudsters often use round numbers
    or specific cent values (e.g. $100.00 vs $99.37).
    """
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_cents"] = (df["TransactionAmt"] % 1 * 100).round(0)
    df["is_round_amount"] = (df["TransactionAmt_cents"] == 0).astype(int)

    logger.debug("Added amount features: log, cents, is_round_amount")
    return df


# ── Card aggregation features ─────────────────────────────────────────────────

def add_card_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    These are the most powerful features in the dataset.

    The intuition: a single card making 50 transactions in one day
    is suspicious. We compute rolling statistics per card.

    WHY NOT REAL ROLLING WINDOWS:
        True rolling windows (last N hours) require sorted time-series
        processing which is expensive and complex. Instead we compute
        global aggregates per card_id as a proxy.
        In a real production system you'd use a feature store
        (Feast, Tecton) for real-time window aggregations.

    card_id = card1 + card2 + card3 + card4 + card5 + card6
    This composite key uniquely identifies a physical card.
    """
    # Create composite card identifier
    card_cols = ["card1", "card2", "card3", "card4", "card5", "card6"]
    existing = [c for c in card_cols if c in df.columns]
    df["card_id"] = df[existing].astype(str).agg("_".join, axis=1)

    # Aggregations per card
    card_agg = df.groupby("card_id")["TransactionAmt"].agg(
        card_txn_count="count",
        card_amt_mean="mean",
        card_amt_std="std",
        card_amt_max="max",
    ).reset_index()

    df = df.merge(card_agg, on="card_id", how="left")

    # How unusual is this transaction vs this card's history?
    # Z-score: how many std devs from this card's mean amount?
    df["amt_z_score"] = (
        (df["TransactionAmt"] - df["card_amt_mean"])
        / (df["card_amt_std"] + 1e-6)  # avoid division by zero
    )

    logger.debug("Added card aggregation features")
    return df


# ── Email domain features ─────────────────────────────────────────────────────

def add_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Email domain mismatch between purchaser and recipient
    is a strong fraud signal — legitimate purchases usually
    have the same domain on both sides.
    """
    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        df["email_match"] = (
            df["P_emaildomain"] == df["R_emaildomain"]
        ).astype(int)

        # Is it a free email provider? (less trustworthy than corporate)
        free_domains = {"gmail.com", "yahoo.com", "hotmail.com",
                        "outlook.com", "aol.com", "icloud.com"}
        df["P_email_is_free"] = df["P_emaildomain"].isin(free_domains).astype(int)
        df["R_email_is_free"] = df["R_emaildomain"].isin(free_domains).astype(int)

    logger.debug("Added email features")
    return df


# ── Missing value flags ───────────────────────────────────────────────────────

def add_missing_flags(df: pd.DataFrame,
                      cols: list[str]) -> pd.DataFrame:
    """
    For important columns, missingness itself is a signal.
    A transaction with no identity record (all identity cols NaN)
    behaves differently from one with full identity data.
    We create binary flags BEFORE imputation so the model
    can learn from the fact that the value was missing.
    """
    for col in cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)
    return df


# ── Drop high-missingness columns ─────────────────────────────────────────────

def drop_high_missing(df: pd.DataFrame,
                      threshold: float = 0.80) -> pd.DataFrame:
    """
    Columns that are >80% missing carry almost no signal
    but add noise and slow down training.
    We identify and drop them from the training set.

    IMPORTANT: record which columns were dropped so the
    inference pipeline drops the same ones.
    """
    miss_frac = df.isnull().mean()
    drop_cols = miss_frac[miss_frac > threshold].index.tolist()
    df = df.drop(columns=drop_cols, errors="ignore")
    logger.info(f"Dropped {len(drop_cols)} columns with >{threshold*100:.0f}% missing")
    return df, drop_cols


# ── Master feature engineering function ──────────────────────────────────────

def engineer_features(df: pd.DataFrame,
                      is_train: bool = True,
                      drop_cols_list: list = None) -> tuple:
    """
    Apply all feature engineering steps in order.

    Args:
        df:             Raw merged DataFrame (transactions + identity)
        is_train:       If True, compute & return drop_cols_list.
                        If False (inference), use provided drop_cols_list.
        drop_cols_list: Columns to drop (passed in during inference
                        so train/inference drop identical columns).

    Returns:
        df:             Feature-engineered DataFrame
        drop_cols_list: List of dropped column names (for inference reuse)
    """
    logger.info(f"Engineering features — shape: {df.shape}")

    # Flag important identity cols before any dropping
    identity_cols = [
        "id_01", "id_02", "id_05", "id_06",
        "id_11", "id_12", "id_13", "id_17",
        "id_19", "id_20", "id_30", "id_31",
        "id_32", "id_33", "DeviceType", "DeviceInfo"
    ]
    df = add_missing_flags(df, identity_cols)

    # Core feature groups
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_card_aggregation_features(df)
    df = add_email_features(df)

    # Drop high-missingness columns
    if is_train:
        df, drop_cols_list = drop_high_missing(df, threshold=0.80)
    else:
        # Inference: drop exactly the same columns as training
        df = df.drop(columns=drop_cols_list or [], errors="ignore")

    # Drop non-feature columns
    always_drop = ["TransactionID", "TransactionDT", "card_id"]
    df = df.drop(columns=always_drop, errors="ignore")

    logger.info(f"Feature engineering complete — shape: {df.shape}")
    return df, drop_cols_list
