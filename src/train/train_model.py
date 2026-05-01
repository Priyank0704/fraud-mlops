"""
src/train/train_model.py

Usage (from project root):
    python -m src.train.train_model

What it does:
    1. Load processed features from data/processed/
    2. Compute scale_pos_weight for class imbalance
    3. Run Optuna hyperparameter search (N trials)
    4. Log every trial to MLflow
    5. Train final model on best hyperparameters
    6. Evaluate with AUC-PR, AUC-ROC, F1, threshold tuning
    7. Generate SHAP feature importance plot
    8. Register best model in MLflow Model Registry
    9. Save model artifact to models/best_model.pkl
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no GUI popups)
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.sklearn
import optuna
from loguru import logger
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,   # = AUC-PR
    f1_score,
    precision_recall_curve,
    classification_report,
)
from xgboost import XGBClassifier

from src.utils.config import cfg
from src.utils.data_loader import load_processed

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data():
    """Load train/val splits from parquet files."""
    logger.info("Loading processed features...")
    X_train = load_processed("X_train")
    X_val   = load_processed("X_val")
    y_train = load_processed("y_train")["isFraud"]
    y_val   = load_processed("y_val")["isFraud"]
    logger.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    XGBoost's built-in way to handle class imbalance.

    scale_pos_weight = count(negative) / count(positive)
    
    WHY THIS INSTEAD OF SMOTE:
        SMOTE creates synthetic samples which adds complexity and
        can introduce noise. scale_pos_weight simply tells XGBoost
        to penalise missing a fraud case more heavily than missing
        a legitimate transaction. Cleaner and faster.
    """
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos
    logger.info(f"scale_pos_weight: {spw:.2f} ({neg} neg / {pos} pos)")
    return spw


def tune_threshold(model, X_val: pd.DataFrame,
                   y_val: pd.Series) -> float:
    """
    Find the probability threshold that maximises F1 on validation set.

    WHY NOT USE 0.5:
        Default threshold of 0.5 is calibrated for balanced classes.
        With 3.5% fraud, the model is more conservative — the optimal
        threshold is usually around 0.3-0.4. We find it empirically.
    """
    probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)

    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = (
        2 * precision * recall
        / (precision + recall + 1e-8)
    )
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    logger.info(f"Best threshold: {best_threshold:.3f} → F1: {best_f1:.4f}")
    return best_threshold


def evaluate(model, X_val: pd.DataFrame,
             y_val: pd.Series, threshold: float) -> dict:
    """Compute all evaluation metrics."""
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_val, probs),
        "auc_pr":  average_precision_score(y_val, probs),
        "f1":      f1_score(y_val, preds),
        "threshold": threshold,
    }
    return metrics, probs


def plot_shap(model, X_val: pd.DataFrame,
              save_path: Path, n_samples: int = 2000):
    """
    Generate SHAP summary plot — the key explainability deliverable.

    WHY SHAP OVER FEATURE IMPORTANCE:
        Built-in XGBoost feature importance (gain/weight) is biased
        toward high-cardinality features. SHAP values are theoretically
        grounded (Shapley values from game theory) and show the actual
        magnitude AND direction of each feature's contribution.

    We sample 2000 rows because SHAP on 118k rows takes ~20 minutes.
    2000 rows gives a statistically representative picture.
    """
    logger.info("Computing SHAP values (this takes ~2 minutes)...")
    sample = X_val.sample(n=min(n_samples, len(X_val)),
                          random_state=cfg.random_seed)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Summary plot — beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, sample,
        max_display=20,
        show=False
    )
    plt.title("SHAP Feature Importance — Top 20 Features", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP plot saved → {save_path}")
    return save_path


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(X_train, X_val, y_train, y_val, scale_pos_weight):
    """
    Optuna minimises/maximises an objective function.
    Each call = one trial with a different set of hyperparameters.

    WHY OPTUNA OVER GRIDSEARCH:
        GridSearch tries every combination (exponential growth).
        Optuna uses Tree-structured Parzen Estimators (TPE) —
        a Bayesian approach that learns which regions of the
        hyperparameter space are promising and samples there more.
        Result: better hyperparameters in fewer trials.
    """
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 3),
            # Fixed params
            "scale_pos_weight":  scale_pos_weight,
            "eval_metric":       "aucpr",
            "use_label_encoder": False,
            "random_state":      cfg.random_seed,
            "n_jobs":            -1,
            "tree_method":       "hist",  # fast histogram method
        }

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Optimise AUC-PR (right metric for imbalanced classification)
        auc_pr = average_precision_score(
            y_val, model.predict_proba(X_val)[:, 1]
        )

        # Log this trial to MLflow as a nested run
        with mlflow.start_run(nested=True,
                              run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metric("auc_pr", auc_pr)

        return auc_pr

    return objective


# ── Main training pipeline ────────────────────────────────────────────────────

def main(n_trials: int = 20):
    """
    n_trials: number of Optuna hyperparameter search trials.
    20 trials ~15-20 min on a laptop. Increase to 50 for better results.
    """
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────
    X_train, X_val, y_train, y_val = load_data()
    scale_pos_weight = compute_scale_pos_weight(y_train)

    # ── MLflow setup ──────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    # ── Parent MLflow run (wraps all Optuna trials) ───────────
    with mlflow.start_run(run_name="fraud_xgboost_optuna") as parent_run:
        logger.info(f"MLflow run ID: {parent_run.info.run_id}")
        logger.info(f"Starting Optuna search — {n_trials} trials...")

        # ── Optuna hyperparameter search ──────────────────────
        study = optuna.create_study(
            direction="maximize",
            study_name="fraud_xgboost",
            sampler=optuna.samplers.TPESampler(seed=cfg.random_seed),
        )
        study.optimize(
            make_objective(X_train, X_val, y_train, y_val, scale_pos_weight),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_params["scale_pos_weight"] = scale_pos_weight
        best_params["eval_metric"]       = "aucpr"
        best_params["random_state"]      = cfg.random_seed
        best_params["n_jobs"]            = -1
        best_params["tree_method"]       = "hist"

        logger.info(f"Best AUC-PR from search: {study.best_value:.4f}")
        logger.info(f"Best params: {best_params}")

        # ── Train final model on best params ──────────────────
        logger.info("Training final model on best hyperparameters...")
        final_model = XGBClassifier(**best_params)
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        # ── Threshold tuning ──────────────────────────────────
        best_threshold = tune_threshold(final_model, X_val, y_val)

        # ── Evaluate ──────────────────────────────────────────
        metrics, probs = evaluate(
            final_model, X_val, y_val, best_threshold
        )
        logger.info(f"Final metrics: {metrics}")

        # Print classification report
        preds = (probs >= best_threshold).astype(int)
        print("\n" + classification_report(y_val, preds,
              target_names=["Legitimate", "Fraud"]))

        # ── Log to MLflow ─────────────────────────────────────
        mlflow.log_params(best_params)
        mlflow.log_param("best_threshold", best_threshold)
        mlflow.log_metrics(metrics)

        # ── SHAP plot ─────────────────────────────────────────
        shap_path = cfg.reports_dir / "shap_summary.png"
        plot_shap(final_model, X_val, shap_path)
        mlflow.log_artifact(str(shap_path))

        # ── Save model locally ────────────────────────────────
        model_path = cfg.models_dir / "best_model.pkl"
        joblib.dump(final_model, model_path)
        logger.info(f"Model saved → {model_path}")

        # Save threshold + feature names for inference
        meta = {
            "threshold":     best_threshold,
            "feature_names": list(X_train.columns),
            "metrics":       metrics,
            "run_id":        parent_run.info.run_id,
        }
        meta_path = cfg.models_dir / "model_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Model metadata saved → {meta_path}")

        # ── Log model to MLflow registry ──────────────────────
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="fraud_model",
            registered_model_name="fraud-detector",
        )
        logger.info("Model registered in MLflow Model Registry")

        logger.info("=" * 60)
        logger.info(f"AUC-ROC : {metrics['auc_roc']:.4f}")
        logger.info(f"AUC-PR  : {metrics['auc_pr']:.4f}  ← primary metric")
        logger.info(f"F1      : {metrics['f1']:.4f}")
        logger.info(f"Threshold: {best_threshold:.3f}")
        logger.info("=" * 60)

if __name__ == "__main__":
    main(n_trials=20)
