"""
src/monitoring/monitor.py

PURPOSE:
    Simulate production drift and generate Evidently monitoring reports.

HOW WE SIMULATE DRIFT:
    We don't have months of real production data, so we simulate it:
    - Reference data  = first 70% of validation set (training-time distribution)
    - Production data = last 30% of validation set + artificial drift injection

    This is a realistic and accepted approach for portfolio projects.
    In a real system, reference = last month's data, current = this month's.

Usage (from project root):
    python -m src.monitoring.monitor

Outputs:
    reports/data_drift_report.html        <- open in browser
    reports/model_performance_report.html <- open in browser
    reports/drift_summary.json            <- machine-readable summary
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    ClassificationPreset,
)
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
)

from src.utils.config import cfg
from src.utils.data_loader import load_processed


# ── Configuration ─────────────────────────────────────────────────────────────

# Features to monitor closely (most important SHAP features)
KEY_FEATURES = [
    "TransactionAmt_log",
    "amt_z_score",
    "hour_sin",
    "hour_cos",
    "card_txn_count",
    "card_amt_mean",
    "TransactionAmt_cents",
    "is_round_amount",
    "email_match",
    "P_email_is_free",
]

DRIFT_THRESHOLD = 0.15   # alert if >15% of features drift


# ── Data preparation ──────────────────────────────────────────────────────────

def load_monitoring_data() -> tuple:
    """
    Load validation data and split into reference vs production windows.

    WHY TEMPORAL SPLIT (not random):
        Random split would give both windows the same distribution.
        Temporal split mimics real-world scenario where production
        data arrives after training reference data.
    """
    logger.info("Loading validation data for monitoring...")
    X_val = load_processed("X_val")
    y_val = load_processed("y_val")["isFraud"]

    n = len(X_val)
    split = int(n * 0.70)

    X_ref  = X_val.iloc[:split].copy()
    X_prod = X_val.iloc[split:].copy()
    y_ref  = y_val.iloc[:split].copy()
    y_prod = y_val.iloc[split:].copy()

    logger.info(f"Reference window:   {X_ref.shape[0]:,} rows")
    logger.info(f"Production window:  {X_prod.shape[0]:,} rows")
    return X_ref, X_prod, y_ref, y_prod


def inject_drift(X_prod: pd.DataFrame,
                 drift_strength: float = 0.3) -> pd.DataFrame:
    """
    Inject artificial drift into production data to simulate
    a real-world distribution shift.

    WHY THIS IS REALISTIC:
        Fraudsters adapt. After a model is deployed, they change:
        - Transaction amounts (avoid round numbers the model learned)
        - Timing patterns (shift from 2am to 6am)
        - Device patterns

    We simulate this by shifting key feature distributions.

    drift_strength: 0 = no drift, 1 = extreme drift
    """
    X_drifted = X_prod.copy()

    # Shift transaction amount distribution upward
    # (fraudsters start using higher amounts to avoid low-amount filters)
    if "TransactionAmt_log" in X_drifted.columns:
        X_drifted["TransactionAmt_log"] += drift_strength * 0.8

    # Shift timing — more fraud during business hours (adaptation)
    if "hour_sin" in X_drifted.columns:
        X_drifted["hour_sin"] = X_drifted["hour_sin"] * (1 - drift_strength * 0.4)
        X_drifted["hour_cos"] = X_drifted["hour_cos"] * (1 - drift_strength * 0.3)

    # Increase z-score anomaly (larger amounts relative to card history)
    if "amt_z_score" in X_drifted.columns:
        X_drifted["amt_z_score"] += drift_strength * 1.5

    # More transactions per card (card-testing behaviour)
    if "card_txn_count" in X_drifted.columns:
        X_drifted["card_txn_count"] *= (1 + drift_strength * 0.5)

    logger.info(f"Drift injected — strength: {drift_strength}")
    return X_drifted


def add_predictions(X: pd.DataFrame,
                    y: pd.Series,
                    model) -> pd.DataFrame:
    """
    Add model predictions to a DataFrame for classification reporting.
    Evidently's ClassificationPreset needs columns:
        target     = actual label
        prediction = predicted probability
    """
    df = X.copy()
    df["target"]     = y.values
    df["prediction"] = model.predict_proba(X)[:, 1]
    return df


# ── Report generation ─────────────────────────────────────────────────────────

def generate_data_drift_report(X_ref: pd.DataFrame,
                                X_prod: pd.DataFrame) -> dict:
    """
    Generate Evidently DataDrift report.

    WHY DATA DRIFT MATTERS:
        If the distribution of input features changes, the model's
        predictions become unreliable — even if code hasn't changed.
        Example: card_txn_count suddenly increases because of
        a new marketing campaign. The model was never trained on
        this pattern and will make more errors.

    Evidently uses statistical tests (KS test for continuous,
    chi-squared for categorical) to detect distribution shifts.
    """
    logger.info("Generating data drift report...")

    # Use only key features for cleaner report
    # (377 features would produce an overwhelming report)
    available_keys = [f for f in KEY_FEATURES if f in X_ref.columns]
    X_ref_sub  = X_ref[available_keys]
    X_prod_sub = X_prod[available_keys]

    report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        *[ColumnDriftMetric(column_name=f) for f in available_keys],
    ])

    report.run(
        reference_data=X_ref_sub,
        current_data=X_prod_sub,
    )

    # Save HTML report
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = cfg.reports_dir / "data_drift_report.html"
    report.save_html(str(html_path))
    logger.info(f"Data drift report saved → {html_path}")

    # Extract summary metrics
    result = report.as_dict()
    drift_metric = next(
        (m for m in result["metrics"]
         if m["metric"] == "DatasetDriftMetric"),
        None
    )

    summary = {}
    if drift_metric:
        share = drift_metric["result"].get("share_of_drifted_columns", 0)
        n_drifted = drift_metric["result"].get("number_of_drifted_columns", 0)
        summary = {
            "share_of_drifted_columns": share,
            "number_of_drifted_columns": n_drifted,
            "drift_detected": share > DRIFT_THRESHOLD,
            "alert": share > DRIFT_THRESHOLD,
        }
        logger.info(f"Drifted features: {n_drifted}/{len(available_keys)} "
                    f"({share*100:.1f}%)")
        if summary["alert"]:
            logger.warning(
                f"DRIFT ALERT: {share*100:.1f}% of features drifted "
                f"(threshold: {DRIFT_THRESHOLD*100:.0f}%). "
                f"Consider retraining the model."
            )

    return summary


def generate_model_performance_report(ref_with_preds: pd.DataFrame,
                                       prod_with_preds: pd.DataFrame):
    """
    Generate Evidently ClassificationPreset report.

    Shows how model performance degrades on production data vs reference:
    - Precision, Recall, F1 comparison
    - Confusion matrix for both windows
    - Prediction distribution shift
    """
    logger.info("Generating model performance report...")

    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=ref_with_preds,
        current_data=prod_with_preds,
    )

    html_path = cfg.reports_dir / "model_performance_report.html"
    report.save_html(str(html_path))
    logger.info(f"Model performance report saved → {html_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────
    X_ref, X_prod, y_ref, y_prod = load_monitoring_data()

    # ── Inject drift into production window ───────────────────
    X_prod_drifted = inject_drift(X_prod, drift_strength=0.3)

    # ── Load model ────────────────────────────────────────────
    model_path = cfg.models_dir / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run: python -m src.train.train_model"
        )
    model = joblib.load(model_path)
    logger.info("Model loaded for monitoring")

    # ── Data drift report ─────────────────────────────────────
    drift_summary = generate_data_drift_report(X_ref, X_prod_drifted)

    # ── Model performance report ──────────────────────────────
    ref_with_preds  = add_predictions(X_ref,          y_ref,  model)
    prod_with_preds = add_predictions(X_prod_drifted, y_prod, model)
    generate_model_performance_report(ref_with_preds, prod_with_preds)

    # ── Save drift summary JSON ───────────────────────────────
    summary = {
        "drift_summary":      drift_summary,
        "drift_threshold":    DRIFT_THRESHOLD,
        "reference_rows":     len(X_ref),
        "production_rows":    len(X_prod_drifted),
        "monitored_features": KEY_FEATURES,
        "recommendation": (
            "RETRAIN MODEL — significant drift detected"
            if drift_summary.get("alert")
            else "Model stable — no retraining needed"
        )
    }
    summary_path = cfg.reports_dir / "drift_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Drift summary saved → {summary_path}")

    # ── Final summary ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MONITORING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Drifted features : "
                f"{drift_summary.get('number_of_drifted_columns', 'N/A')}"
                f"/{len(KEY_FEATURES)}")
    logger.info(f"Drift share      : "
                f"{drift_summary.get('share_of_drifted_columns', 0)*100:.1f}%")
    logger.info(f"Recommendation   : {summary['recommendation']}")
    logger.info("=" * 60)
    logger.info("Reports saved to reports/ folder:")
    logger.info("  reports/data_drift_report.html")
    logger.info("  reports/model_performance_report.html")
    logger.info("  reports/drift_summary.json")
    logger.info("Open the HTML files in your browser to view dashboards.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
