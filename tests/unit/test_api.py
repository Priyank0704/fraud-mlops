"""
tests/unit/test_api.py
Run with: pytest tests/ -v
"""

import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# ── Load real metadata ────────────────────────────────────────────────────────

META_PATH = Path("models/model_meta.json")
if META_PATH.exists():
    with open(META_PATH) as f:
        REAL_META = json.load(f)
else:
    REAL_META = {
        "threshold": 0.728,
        "feature_names": [f"feature_{i}" for i in range(377)],
        "metrics": {"auc_roc": 0.9676, "auc_pr": 0.8208, "f1": 0.7710},
        "run_id": "abc123",
    }

FEATURE_NAMES = REAL_META["feature_names"]
N_FEATURES    = len(FEATURE_NAMES)
THRESHOLD     = REAL_META["threshold"]

# ── Sample transaction ────────────────────────────────────────────────────────

SAMPLE_TRANSACTION = {
    "TransactionDT":  86400.0,
    "TransactionAmt": 49.99,
    "ProductCD":      "W",
    "card1":          9500.0,
    "card4":          "visa",
    "card6":          "debit",
    "P_emaildomain":  "gmail.com",
    "R_emaildomain":  "gmail.com",
    "M1": "T", "M2": "F", "M3": "T",
}

# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.13, 0.87]])

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.random.uniform(-1, 1, (1, N_FEATURES))

    mock_preprocessor = {
        "preprocessor":   MagicMock(),
        "target_encoder": MagicMock(),
        "drop_cols":      [],
        "col_types": {
            "numeric":      FEATURE_NAMES,
            "low_card_cat": [],
        },
    }
    mock_preprocessor["preprocessor"].transform.return_value = np.zeros((1, N_FEATURES))
    mock_preprocessor["target_encoder"].transform.side_effect = lambda x: x

    mock_X = pd.DataFrame(np.zeros((1, N_FEATURES)), columns=FEATURE_NAMES)

    with patch("src.api.main.MODEL",                  mock_model),        \
         patch("src.api.main.META",                   REAL_META),         \
         patch("src.api.main.PREPROCESSOR_ARTEFACTS", mock_preprocessor), \
         patch("src.api.main.EXPLAINER",              mock_explainer),     \
         patch("src.api.main.preprocess_input",       return_value=mock_X):
        from src.api.main import app
        with TestClient(app) as c:
            yield c

# ── Health tests ──────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_response_structure(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "threshold" in data

    def test_health_threshold_value(self, client):
        data = client.get("/health").json()
        assert round(data["threshold"], 3) == round(THRESHOLD, 3)

# ── Model info tests ──────────────────────────────────────────────────────────

class TestModelInfo:
    def test_model_info_returns_200(self, client):
        assert client.get("/model-info").status_code == 200

    def test_model_info_has_metrics(self, client):
        data = client.get("/model-info").json()
        assert "metrics" in data
        assert "auc_pr"  in data["metrics"]
        assert "auc_roc" in data["metrics"]

    def test_model_info_values(self, client):
        data = client.get("/model-info").json()
        assert data["model_type"]    == "XGBoost"
        assert data["feature_count"] == N_FEATURES
        assert data["metrics"]["auc_pr"] == REAL_META["metrics"]["auc_pr"]

# ── Predict tests ─────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_200(self, client):
        r = client.post("/predict", json=SAMPLE_TRANSACTION)
        assert r.status_code == 200, r.json()

    def test_predict_response_structure(self, client):
        data = client.post("/predict", json=SAMPLE_TRANSACTION).json()
        assert "fraud_probability" in data
        assert "is_fraud"          in data
        assert "risk_tier"         in data
        assert "explanation"       in data
        assert "threshold_used"    in data

    def test_predict_probability_range(self, client):
        data = client.post("/predict", json=SAMPLE_TRANSACTION).json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_predict_risk_tier_valid(self, client):
        data = client.post("/predict", json=SAMPLE_TRANSACTION).json()
        assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_predict_high_fraud_probability(self, client):
        data = client.post("/predict", json=SAMPLE_TRANSACTION).json()
        prob     = data["fraud_probability"]
        is_fraud = data["is_fraud"]
        assert 0.0 <= prob <= 1.0
        if prob >= THRESHOLD:
            assert is_fraud is True
        else:
            assert is_fraud is False
        

    def test_predict_explanation_has_5_features(self, client):
        data = client.post("/predict", json=SAMPLE_TRANSACTION).json()
        assert len(data["explanation"]) == 5

    def test_predict_explanation_structure(self, client):
        first = client.post("/predict", json=SAMPLE_TRANSACTION).json()["explanation"][0]
        assert "feature" in first
        assert "value"   in first
        assert "shap"    in first
        assert "impact"  in first

    def test_predict_missing_optional_fields(self, client):
        r = client.post("/predict", json={"TransactionDT": 86400.0, "TransactionAmt": 100.0})
        assert r.status_code == 200

    def test_predict_missing_required_field(self, client):
        r = client.post("/predict", json={"TransactionDT": 86400.0})
        assert r.status_code == 422

# ── Batch tests ───────────────────────────────────────────────────────────────

class TestBatchPredict:
    def test_batch_returns_200(self, client):
        r = client.post("/predict/batch", json={"transactions": [SAMPLE_TRANSACTION, SAMPLE_TRANSACTION]})
        assert r.status_code == 200

    def test_batch_response_structure(self, client):
        data = client.post("/predict/batch", json={"transactions": [SAMPLE_TRANSACTION]}).json()
        assert "total"       in data
        assert "fraud_count" in data
        assert "predictions" in data

    def test_batch_total_count(self, client):
        data = client.post("/predict/batch", json={"transactions": [SAMPLE_TRANSACTION, SAMPLE_TRANSACTION]}).json()
        assert data["total"] == 2

    def test_batch_exceeds_limit(self, client):
        r = client.post("/predict/batch", json={"transactions": [SAMPLE_TRANSACTION] * 101})
        assert r.status_code == 400
