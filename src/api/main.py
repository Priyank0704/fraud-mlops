"""
src/api/main.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchTransactionInput,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.features.preprocessing import load_preprocessor
from src.utils.config import cfg

# ── Global artefacts ──────────────────────────────────────────────────────────

MODEL = None
META  = None
PREPROCESSOR_ARTEFACTS = None
EXPLAINER = None


def load_artefacts():
    global MODEL, META, PREPROCESSOR_ARTEFACTS, EXPLAINER

    model_path = cfg.models_dir / "best_model.pkl"
    meta_path  = cfg.models_dir / "model_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run: python -m src.train.train_model"
        )

    logger.info("Loading model artefacts...")
    MODEL = joblib.load(model_path)
    PREPROCESSOR_ARTEFACTS = load_preprocessor()

    with open(meta_path) as f:
        META = json.load(f)

    EXPLAINER = shap.TreeExplainer(MODEL)
    logger.info(f"Model loaded — threshold: {META['threshold']:.3f}")
    logger.info(f"Features expected: {len(META['feature_names'])}")


# ── Lifespan (MUST be defined BEFORE app) ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artefacts()
    yield


# ── App (lifespan passed in here) ─────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using XGBoost trained on IEEE-CIS data.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess_input(transaction: dict) -> pd.DataFrame:
    from src.features.feature_engineering import engineer_features

    df = pd.DataFrame([transaction])
    artefacts = PREPROCESSOR_ARTEFACTS

    df, _ = engineer_features(
        df,
        is_train=False,
        drop_cols_list=artefacts["drop_cols"],
    )
    df = df.drop(columns=[cfg.target_col], errors="ignore")
    df = artefacts["target_encoder"].transform(df)

    col_types    = artefacts["col_types"]
    preprocessor = artefacts["preprocessor"]

    # ── Add ALL missing columns as NaN BEFORE preprocessor ───
    # The preprocessor's SimpleImputer will fill them with
    # the median learned during training — correct behavior.
    all_expected = col_types["numeric"] + col_types["low_card_cat"]
    for col in all_expected:
        if col not in df.columns:
            df[col] = np.nan

    arr           = preprocessor.transform(df[all_expected])
    feature_names = all_expected
    X             = pd.DataFrame(arr, columns=feature_names)

    # Align to exact training feature order
    for col in META["feature_names"]:
        if col not in X.columns:
            X[col] = 0.0
    X = X[META["feature_names"]]
    return X


def get_risk_tier(probability: float) -> str:
    if probability >= 0.80:
        return "CRITICAL"
    elif probability >= META["threshold"]:
        return "HIGH"
    elif probability >= 0.30:
        return "MEDIUM"
    else:
        return "LOW"


def get_shap_explanation(X: pd.DataFrame, top_n: int = 5) -> list:
    shap_values   = EXPLAINER.shap_values(X)
    feature_names = META["feature_names"]
    abs_shap      = np.abs(shap_values[0])
    top_idx       = np.argsort(abs_shap)[::-1][:top_n]

    return [{
        "feature": feature_names[idx],
        "value":   float(X.iloc[0, idx]),
        "shap":    float(shap_values[0][idx]),
        "impact":  "increases_fraud_risk" if shap_values[0][idx] > 0 else "decreases_fraud_risk",
    } for idx in top_idx]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="healthy", model_loaded=True, threshold=META["threshold"])


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    if META is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_type="XGBoost",
        feature_count=len(META["feature_names"]),
        threshold=META["threshold"],
        metrics=META["metrics"],
        mlflow_run_id=META["run_id"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(transaction: TransactionInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        X           = preprocess_input(transaction.model_dump())
        prob        = float(MODEL.predict_proba(X)[0][1])
        is_fraud    = prob >= META["threshold"]
        explanation = get_shap_explanation(X, top_n=5)
        logger.info(f"Prediction: prob={prob:.3f}, fraud={is_fraud}")
        return PredictionResponse(
            fraud_probability=round(prob, 4),
            is_fraud=is_fraud,
            risk_tier=get_risk_tier(prob),
            explanation=explanation,
            threshold_used=META["threshold"],
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchTransactionInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(batch.transactions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 transactions per batch")

    predictions = []
    for txn in batch.transactions:
        try:
            X    = preprocess_input(txn.model_dump())
            prob = float(MODEL.predict_proba(X)[0][1])
            predictions.append({
                "fraud_probability": round(prob, 4),
                "is_fraud":          prob >= META["threshold"],
                "risk_tier":         get_risk_tier(prob),
            })
        except Exception as e:
            predictions.append({"error": str(e)})

    fraud_count = sum(1 for p in predictions if p.get("is_fraud"))
    return BatchPredictionResponse(total=len(predictions), fraud_count=fraud_count, predictions=predictions)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=cfg.api_host, port=cfg.api_port, reload=True)
