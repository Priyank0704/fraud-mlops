"""
scripts/create_mock_artefacts.py

Creates minimal model artefacts for CI/CD pipeline.
Used in GitHub Actions so tests and Docker build
don't need the real 100MB trained model.
"""

import json
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURE_NAMES = [f"feature_{i}" for i in range(377)]

# 1. model_meta.json
meta = {
    "threshold": 0.728,
    "feature_names": FEATURE_NAMES,
    "metrics": {"auc_roc": 0.9676, "auc_pr": 0.8208, "f1": 0.771},
    "run_id": "ci-mock-run",
}
pathlib.Path("models/model_meta.json").write_text(json.dumps(meta))
print("Created models/model_meta.json")

# 2. best_model.pkl — tiny trained XGBoost
X = pd.DataFrame(np.zeros((10, 377)), columns=FEATURE_NAMES)
y = np.array([0] * 9 + [1])
mock_model = XGBClassifier(n_estimators=1, max_depth=1, random_state=42)
mock_model.fit(X, y)
joblib.dump(mock_model, "models/best_model.pkl")
print("Created models/best_model.pkl")

# 3. preprocessor.pkl
artefacts = {
    "preprocessor": Pipeline([("scaler", StandardScaler())]),
    "target_encoder": None,
    "drop_cols": [],
    "col_types": {
        "numeric": FEATURE_NAMES,
        "low_card_cat": [],
    },
}
joblib.dump(artefacts, "models/preprocessor.pkl")
print("Created models/preprocessor.pkl")
print("All mock artefacts created successfully")
