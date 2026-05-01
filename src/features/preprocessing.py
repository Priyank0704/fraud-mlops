"""
src/features/preprocessing.py

WHY A SKLEARN PIPELINE:
    A Pipeline chains transformers so that fit() on training data
    automatically applies the same learned parameters at predict time.

    Without Pipeline:
        # Training
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        # Inference — easy to forget this step or use wrong scaler
        X_new_scaled = scaler.transform(X_new)

    With Pipeline:
        pipeline.fit(X_train, y_train)
        pipeline.predict(X_new)  # scaling happens automatically inside

    This eliminates an entire class of training-serving skew bugs.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer
from loguru import logger

from src.utils.config import cfg


# ── Target Encoder (manual — avoids leakage) ─────────────────────────────────

class TargetEncoder:
    """
    Encode high-cardinality categoricals with their target mean.

    WHY NOT SKLEARN'S BUILT-IN TargetEncoder:
        sklearn's version is fine, but building it yourself shows
        interviewers you understand the leakage problem:
        - WRONG: compute mean on full training set, then cross-validate
        - RIGHT: compute mean only on training fold, apply to val fold

    WHY ADD SMOOTHING:
        Rare categories (e.g. an email domain seen once) have unreliable
        means. Smoothing pulls rare-category estimates toward the global
        mean, reducing variance.

        smoothed_mean = (n * cat_mean + k * global_mean) / (n + k)
        where k = smoothing strength (default 10), n = category count
    """

    def __init__(self, cols: list[str], smoothing: int = 10):
        self.cols = cols
        self.smoothing = smoothing
        self.encoding_map_ = {}
        self.global_mean_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_mean_ = y.mean()
        for col in self.cols:
            if col not in X.columns:
                continue
            stats = (
                pd.DataFrame({"col": X[col], "target": y})
                .groupby("col")["target"]
                .agg(["mean", "count"])
            )
            smooth = (
                (stats["count"] * stats["mean"]
                 + self.smoothing * self.global_mean_)
                / (stats["count"] + self.smoothing)
            )
            self.encoding_map_[col] = smooth.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            if col not in X.columns:
                continue
            X[col] = (
                X[col]
                .map(self.encoding_map_.get(col, {}))
                .fillna(self.global_mean_)
            )
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# ── Build the column transformer ─────────────────────────────────────────────

def get_column_types(df: pd.DataFrame,
                     target_col: str = "isFraud") -> dict:
    """
    Automatically detect numeric vs categorical columns.
    Returns dict with keys: numeric, low_card_cat, high_card_cat
    """
    from src.features.feature_engineering import HIGH_CARD_CATS, LOW_CARD_CATS

    feature_cols = [c for c in df.columns if c != target_col]

    high_card = [c for c in HIGH_CARD_CATS if c in feature_cols]
    low_card  = [c for c in LOW_CARD_CATS  if c in feature_cols]
    numeric   = [
        c for c in feature_cols
        if c not in high_card + low_card
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    logger.info(
        f"Column types — numeric: {len(numeric)}, "
        f"low_card_cat: {len(low_card)}, "
        f"high_card_cat: {len(high_card)}"
    )
    return {"numeric": numeric, "low_card_cat": low_card, "high_card_cat": high_card}


def build_preprocessor(col_types: dict) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies:
    - Numeric cols:      median impute → standard scale
    - Low-card cats:     most-frequent impute → ordinal encode
    - High-card cats:    handled separately by TargetEncoder before pipeline

    WHY MEDIAN IMPUTE (not mean):
        Transaction amounts and other financial features are skewed.
        Median is more robust to outliers than mean.

    WHY ORDINAL ENCODE (not one-hot) for categoricals:
        XGBoost and LightGBM handle ordinal integers natively and
        efficiently. One-hot encoding explodes dimensionality
        with no benefit for tree-based models.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,  col_types["numeric"]),
            ("cat", categorical_transformer, col_types["low_card_cat"]),
        ],
        remainder="drop",   # drop anything not explicitly listed
        verbose_feature_names_out=False,
    )
    return preprocessor


# ── Save / load helpers ───────────────────────────────────────────────────────

def save_preprocessor(preprocessor, target_encoder,
                      drop_cols: list, col_types: dict):
    """
    Save all preprocessing artefacts together so inference
    can reconstruct the exact same transformation pipeline.
    """
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    artefacts = {
        "preprocessor": preprocessor,
        "target_encoder": target_encoder,
        "drop_cols": drop_cols,
        "col_types": col_types,
    }
    path = cfg.models_dir / "preprocessor.pkl"
    joblib.dump(artefacts, path)
    logger.info(f"Saved preprocessor artefacts → {path}")
    return path


def load_preprocessor() -> dict:
    path = cfg.models_dir / "preprocessor.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {path}")
    return joblib.load(path)
