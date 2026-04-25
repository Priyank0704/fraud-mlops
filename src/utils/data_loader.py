"""
src/utils/data_loader.py

Handles loading and merging the IEEE-CIS fraud dataset.

WHY A SEPARATE MODULE:
  Training scripts, the API, and tests all need data.
  One source of truth prevents subtle bugs from different
  merge strategies in different places.
"""

import pandas as pd
from pathlib import Path
from loguru import logger


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_raw(split: str = "train") -> pd.DataFrame:
    """
    Load and merge transaction + identity tables.

    Args:
        split: "train" or "test"

    Returns:
        Merged DataFrame. Shape: ~590k rows × 434 cols (train).

    WHY LEFT JOIN:
        Not every transaction has an identity record (~40% missing).
        Left join keeps all transactions; identity features will be
        NaN where absent — we handle this in the preprocessing pipeline.
    """
    logger.info(f"Loading {split} data from {RAW_DIR}")

    txn_path = RAW_DIR / f"{split}_transaction.csv"
    idn_path = RAW_DIR / f"{split}_identity.csv"

    if not txn_path.exists():
        raise FileNotFoundError(
            f"Transaction file not found: {txn_path}\n"
            "Run: kaggle competitions download -c ieee-fraud-detection -p data/raw/"
        )

    txn = pd.read_csv(txn_path)
    logger.info(f"  Transactions: {txn.shape}")

    if idn_path.exists():
        idn = pd.read_csv(idn_path)
        logger.info(f"  Identity:     {idn.shape}")
        df = txn.merge(idn, on="TransactionID", how="left")
    else:
        logger.warning(f"Identity file not found at {idn_path}, skipping merge")
        df = txn

    logger.info(f"  Merged shape: {df.shape}")
    return df


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """
    Save a processed DataFrame as parquet.

    WHY PARQUET OVER CSV:
        - 5–10× smaller file size (columnar compression)
        - Preserves dtypes (no re-casting on reload)
        - Faster read/write for large DataFrames
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Saved {name}.parquet — {df.shape}")
    return out


def load_processed(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_parquet(path)
