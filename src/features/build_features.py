"""
src/features/build_features.py

This is the script you RUN to process raw data into model-ready features.

Usage (from project root):
    python -m src.features.build_features

What it does:
    1. Load raw train data
    2. Engineer features
    3. Target-encode high-cardinality categoricals
    4. Apply sklearn ColumnTransformer (impute + scale)
    5. Save processed X_train, X_val, y_train, y_val as parquet
    6. Save all preprocessing artefacts for inference reuse
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.utils.config import cfg
from src.utils.data_loader import load_raw, save_processed
from src.features.feature_engineering import engineer_features, HIGH_CARD_CATS
from src.features.preprocessing import (
    TargetEncoder,
    get_column_types,
    build_preprocessor,
    save_preprocessor,
)


def main():
    logger.info("=" * 60)
    logger.info("Building features")
    logger.info("=" * 60)

    # ── Step 1: Load raw data ─────────────────────────────────
    df = load_raw("train")
    logger.info(f"Raw data shape: {df.shape}")

    # ── Step 2: Separate target ───────────────────────────────
    y = df[cfg.target_col].copy()
    logger.info(f"Fraud rate: {y.mean()*100:.2f}%")

    # ── Step 3: Engineer features ─────────────────────────────
    df, drop_cols = engineer_features(df, is_train=True)
    # Remove target from feature df (it may still be there)
    df = df.drop(columns=[cfg.target_col], errors="ignore")

    # ── Step 4: Train / validation split ─────────────────────
    # WHY SPLIT BEFORE ENCODING:
    #   Target encoding must be fit ONLY on training data.
    #   If we fit on the full dataset, validation labels leak
    #   into the encoding — a subtle but serious form of leakage.
    X_train, X_val, y_train, y_val = train_test_split(
        df, y,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=y,        # preserve fraud rate in both splits
    )
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
    logger.info(f"Train fraud rate: {y_train.mean()*100:.2f}%")
    logger.info(f"Val   fraud rate: {y_val.mean()*100:.2f}%")

    # ── Step 5: Target encode high-cardinality categoricals ───
    high_card_cols = [c for c in HIGH_CARD_CATS if c in X_train.columns]
    te = TargetEncoder(cols=high_card_cols, smoothing=10)
    X_train = te.fit_transform(X_train, y_train)   # fit on train only
    X_val   = te.transform(X_val)                  # apply to val

    # ── Step 6: Build & fit column transformer ────────────────
    col_types = get_column_types(
        pd.concat([X_train, pd.Series(y_train, name=cfg.target_col)], axis=1)
    )
    preprocessor = build_preprocessor(col_types)

    X_train_arr = preprocessor.fit_transform(X_train)  # fit on train only
    X_val_arr   = preprocessor.transform(X_val)        # apply to val

    # Get feature names from ColumnTransformer
    feature_names = (
        col_types["numeric"] + col_types["low_card_cat"]
    )

    X_train_df = pd.DataFrame(X_train_arr, columns=feature_names)
    X_val_df   = pd.DataFrame(X_val_arr,   columns=feature_names)

    logger.info(f"Final train features shape: {X_train_df.shape}")
    logger.info(f"Final val   features shape: {X_val_df.shape}")

    # ── Step 7: Save processed data ───────────────────────────
    save_processed(X_train_df, "X_train")
    save_processed(X_val_df,   "X_val")
    save_processed(
        pd.DataFrame({"isFraud": y_train.values}), "y_train"
    )
    save_processed(
        pd.DataFrame({"isFraud": y_val.values}), "y_val"
    )

    # ── Step 8: Save preprocessing artefacts ─────────────────
    save_preprocessor(preprocessor, te, drop_cols, col_types)


if __name__ == "__main__":
    main()
