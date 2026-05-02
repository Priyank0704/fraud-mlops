# Fraud Detection MLOps Pipeline

A production-grade machine learning system for real-time credit card fraud detection, built with XGBoost, FastAPI, MLflow, Docker, and GitHub Actions CI/CD.

[![CI/CD Pipeline](https://github.com/Priyank0704/fraud-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Priyank0704/fraud-mlops/actions/workflows/ci-cd.yml)

---

## Live Demo

| Resource | Link |
|---|---|
| **Live API** | https://fraud-detection-api-vtl0.onrender.com/docs |
| **Health Check** | https://fraud-detection-api-vtl0.onrender.com/health |
| **Docker Image** | `docker pull priyank0704/fraud-detection-api:latest` |

---

## Project Overview

This project demonstrates a complete MLOps pipeline — from raw data ingestion to a deployed, monitored inference service. It was built to showcase production ML engineering skills relevant to Canadian fintech, banking, and insurance industries.

**Business Problem:** Financial institutions lose billions annually to fraudulent transactions. This system predicts fraud probability in real time with explainable AI, enabling analysts to act on flagged transactions with full reasoning.

**Dataset:** IEEE-CIS Fraud Detection (590,540 transactions, 433 features, 3.5% fraud rate) from Vesta Corporation via Kaggle.

---

## Architecture

```
Raw Data (CSV)
     │
     ▼
Feature Engineering          ← time features, card aggregations,
     │                          email matching, target encoding
     ▼
Preprocessing Pipeline       ← sklearn Pipeline (impute + scale)
     │                          saved as preprocessor.pkl
     ▼
XGBoost Training             ← Optuna hyperparameter search (20 trials)
     │                          MLflow experiment tracking
     ▼
Model Registry               ← MLflow Model Registry
     │                          best_model.pkl + model_meta.json
     ▼
FastAPI Inference Service    ← POST /predict → fraud probability
     │                          + SHAP explanations per prediction
     ▼
Docker Container             ← containerized, reproducible deployment
     │
     ▼
GitHub Actions CI/CD         ← test → build → push → deploy
     │
     ▼
Render (Cloud Hosting)       ← live public API
     │
     ▼
Evidently Monitoring         ← data drift + model performance reports
```

---

## Model Performance

| Metric | Value |
|---|---|
| AUC-ROC | 0.9676 |
| AUC-PR | 0.8208 |
| F1 Score | 0.7710 |
| Decision Threshold | 0.728 |
| Training samples | 472,432 |
| Validation samples | 118,108 |

> **Why AUC-PR over AUC-ROC?** With only 3.5% fraud, AUC-ROC is misleading — a model predicting all legitimate scores ~0.75. AUC-PR is the honest metric for imbalanced classification.

---

## Key Engineering Decisions

**Threshold tuning:** Default 0.5 threshold is wrong for imbalanced data. We sweep the precision-recall curve on the validation set and select the threshold that maximises F1 (0.728 in this case).

**scale_pos_weight:** Instead of SMOTE oversampling, we use XGBoost's built-in class weighting (neg/pos ratio = 27.58). Cleaner, faster, and easier to explain.

**SHAP explanations:** Every prediction returns the top 5 features driving the score. A fraud probability of 0.87 is not actionable — "flagged because amount is 40x card average and email domains don't match" is.

**scikit-learn Pipeline:** All preprocessing (imputation, scaling, encoding) is wrapped in a fitted Pipeline object. The same object used at training time is loaded at inference time — eliminating training-serving skew.

---

## Repository Structure

```
fraud-mlops/
├── src/
│   ├── features/
│   │   ├── feature_engineering.py   # time, amount, card, email features
│   │   ├── preprocessing.py         # sklearn Pipeline + TargetEncoder
│   │   └── build_features.py        # orchestrates full feature pipeline
│   ├── train/
│   │   └── train_model.py           # Optuna + XGBoost + MLflow
│   ├── api/
│   │   ├── main.py                  # FastAPI app with lifespan
│   │   └── schemas.py               # Pydantic request/response models
│   ├── monitoring/
│   │   └── monitor.py               # Evidently drift detection
│   └── utils/
│       ├── config.py                # central configuration
│       └── data_loader.py           # load/save utilities
├── tests/
│   └── unit/
│       └── test_api.py              # 19 pytest unit tests
├── scripts/
│   └── create_mock_artefacts.py     # CI/CD mock model generation
├── notebooks/
│   └── 01_eda.py                    # exploratory data analysis
├── .github/
│   └── workflows/
│       └── ci-cd.yml                # GitHub Actions pipeline
├── Dockerfile                       # python:3.11-slim, non-root user
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.11
- Docker Desktop
- Kaggle account (for dataset)

### Local Setup

```bash
# Clone repo
git clone https://github.com/Priyank0704/fraud-mlops.git
cd fraud-mlops

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download dataset
kaggle competitions download -c ieee-fraud-detection -p data/raw/
tar -xf data/raw/ieee-fraud-detection.zip -C data/raw/
```

### Run the Pipeline

```bash
# Step 1: Feature engineering (~5 minutes)
python -m src.features.build_features

# Step 2: Train model with MLflow tracking (~25 minutes)
python -m src.train.train_model

# Step 3: View MLflow experiments
mlflow ui
# Open http://127.0.0.1:5000

# Step 4: Start inference API
python -m src.api.main
# Open http://127.0.0.1:8000/docs
```

### Run with Docker

```bash
docker-compose up --build
# API available at http://127.0.0.1:8000
```

### Run Tests

```bash
pytest tests/ -v
# 19 tests, ~10 seconds
```

---

## API Reference

### POST /predict

Predict fraud probability for a single transaction.

**Request:**
```json
{
  "TransactionDT": 86400.0,
  "TransactionAmt": 2999.99,
  "ProductCD": "W",
  "card4": "visa",
  "card6": "credit",
  "P_emaildomain": "gmail.com",
  "R_emaildomain": "yahoo.com"
}
```

**Response:**
```json
{
  "fraud_probability": 0.8734,
  "is_fraud": true,
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
```

### GET /health
```json
{"status": "healthy", "model_loaded": true, "threshold": 0.728}
```

### GET /model-info
```json
{
  "model_type": "XGBoost",
  "feature_count": 377,
  "threshold": 0.728,
  "metrics": {"auc_roc": 0.9676, "auc_pr": 0.8208, "f1": 0.771}
}
```

---

## CI/CD Pipeline

Every push to `main` automatically:

1. **Runs 19 pytest tests** — blocks deployment if any fail
2. **Builds Docker image** — tagged with git commit SHA
3. **Pushes to Docker Hub** — `priyank0704/fraud-detection-api:latest`
4. **Deploys to Render** — live public API updated automatically

---

## Tech Stack

| Category | Tools |
|---|---|
| Modelling | XGBoost, scikit-learn, Optuna |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| API | FastAPI, Pydantic, Uvicorn |
| Containerisation | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI |
| Cloud | Render, Docker Hub |
| Testing | pytest, pytest-cov |

---

## Author

**Priyank** — Post Graduate in Big Data Analytics (2024) and Artificial Intelligence (2025)

- GitHub: [@Priyank0704](https://github.com/Priyank0704)

---

## Acknowledgements

- Dataset: [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) by Vesta Corporation
- MIMIC-IV: PhysioNet / Beth Israel Deaconess Medical Center
