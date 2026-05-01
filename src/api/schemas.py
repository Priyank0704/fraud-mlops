"""
src/api/schemas.py

WHY PYDANTIC SCHEMAS:
    Pydantic validates every incoming request automatically.
    If a required field is missing or the wrong type, FastAPI
    returns a clean 422 error BEFORE your code runs.
    No manual if/else validation needed.

    It also auto-generates the Swagger UI documentation at /docs
    so anyone can test your API without writing any code.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Input schema ──────────────────────────────────────────────────────────────

class TransactionInput(BaseModel):
    """
    A single transaction. All fields mirror the IEEE-CIS dataset columns.
    Optional fields default to None (missing is handled by preprocessor).
    """
    # Core transaction fields
    TransactionDT:  float = Field(..., description="Seconds since reference time", example=86400.0)
    TransactionAmt: float = Field(..., description="Transaction amount in USD", example=49.99)
    ProductCD:      Optional[str]   = Field(None, description="Product code", example="W")

    # Card details
    card1:  Optional[float] = Field(None, example=9500.0)
    card2:  Optional[float] = Field(None, example=325.0)
    card3:  Optional[float] = Field(None, example=150.0)
    card4:  Optional[str]   = Field(None, description="Card network", example="visa")
    card5:  Optional[float] = Field(None, example=226.0)
    card6:  Optional[str]   = Field(None, description="Card type", example="debit")

    # Address
    addr1:  Optional[float] = Field(None, example=315.0)
    addr2:  Optional[float] = Field(None, example=87.0)

    # Distance
    dist1:  Optional[float] = Field(None, example=19.0)
    dist2:  Optional[float] = Field(None, example=None)

    # Email domains
    P_emaildomain: Optional[str] = Field(None, example="gmail.com")
    R_emaildomain: Optional[str] = Field(None, example="gmail.com")

    # Match flags (T/F/M strings in raw data)
    M1: Optional[str] = Field(None, example="T")
    M2: Optional[str] = Field(None, example="F")
    M3: Optional[str] = Field(None, example="T")
    M4: Optional[str] = Field(None, example="M2")
    M5: Optional[str] = Field(None, example="F")
    M6: Optional[str] = Field(None, example="T")
    M7: Optional[str] = Field(None, example=None)
    M8: Optional[str] = Field(None, example=None)
    M9: Optional[str] = Field(None, example=None)

    class Config:
        # Allow extra fields to be passed without error
        # (the preprocessor will drop unknown columns)
        extra = "allow"
        json_schema_extra = {
            "example": {
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
        }


class BatchTransactionInput(BaseModel):
    transactions: list[TransactionInput] = Field(
        ..., description="List of transactions (max 100)"
    )


# ── Output schemas ────────────────────────────────────────────────────────────

class SHAPFeature(BaseModel):
    feature: str   = Field(..., description="Feature name")
    value:   float = Field(..., description="Feature value for this transaction")
    shap:    float = Field(..., description="SHAP value (magnitude = importance)")
    impact:  str   = Field(..., description="increases_fraud_risk or decreases_fraud_risk")


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="Model fraud score (0–1)", example=0.8734)
    is_fraud:          bool  = Field(..., description="Binary decision at tuned threshold")
    risk_tier:         str   = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    explanation:       list[SHAPFeature] = Field(..., description="Top 5 SHAP features")
    threshold_used:    float = Field(..., description="Decision threshold applied")

    class Config:
        json_schema_extra = {
            "example": {
                "fraud_probability": 0.8734,
                "is_fraud": True,
                "risk_tier": "CRITICAL",
                "threshold_used": 0.728,
                "explanation": [
                    {
                        "feature": "amt_z_score",
                        "value": 42.3,
                        "shap": 0.94,
                        "impact": "increases_fraud_risk"
                    },
                    {
                        "feature": "email_match",
                        "value": 0.0,
                        "shap": 0.61,
                        "impact": "increases_fraud_risk"
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    total:       int  = Field(..., description="Total transactions processed")
    fraud_count: int  = Field(..., description="Number flagged as fraud")
    predictions: list = Field(..., description="Per-transaction results")


class HealthResponse(BaseModel):
    status:       str   = Field(..., example="healthy")
    model_loaded: bool  = Field(..., example=True)
    threshold:    float = Field(..., example=0.728)


class ModelInfoResponse(BaseModel):
    model_type:    str   = Field(..., example="XGBoost")
    feature_count: int   = Field(..., example=377)
    threshold:     float = Field(..., example=0.728)
    metrics:       dict  = Field(..., description="Validation metrics from training")
    mlflow_run_id: str   = Field(..., description="MLflow run that produced this model")
