"""
src/utils/config.py

WHY A CENTRAL CONFIG:
  Magic numbers scattered across scripts are a maintenance nightmare.
  One config file means one place to change the test split ratio,
  the random seed, or the fraud threshold — everywhere updates.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────
    raw_dir: Path       = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path    = Path("models")
    reports_dir: Path   = Path("reports")

    # ── Data ─────────────────────────────────────────────────
    target_col: str     = "isFraud"
    id_col: str         = "TransactionID"
    # Drop columns with more than this fraction missing
    max_missing_frac: float = 0.80
    # Columns to always drop (non-feature)
    drop_cols: list = field(default_factory=lambda: ["TransactionID"])

    # ── Training ─────────────────────────────────────────────
    random_seed: int    = 42
    test_size: float    = 0.20
    cv_folds: int       = 5

    # ── Class imbalance ──────────────────────────────────────
    # Strategy: use scale_pos_weight in XGBoost (neg/pos ratio)
    # SMOTE is an alternative but scale_pos_weight is faster & cleaner
    use_smote: bool     = False

    # ── Inference threshold ──────────────────────────────────
    # Default 0.5 is wrong for imbalanced data.
    # We'll tune this on the validation set to maximise F1.
    fraud_threshold: float = 0.5   # updated after threshold tuning

    # ── MLflow ───────────────────────────────────────────────
    mlflow_experiment: str = "fraud-detection"
    mlflow_tracking_uri: str = "mlruns"   # local; swap for remote URI in prod

    # ── API ──────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_path: Path = Path("models/best_model.pkl")


# Singleton — import this everywhere
cfg = Config()
