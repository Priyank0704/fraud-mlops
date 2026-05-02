"""
scripts/create_mock_artefacts.py
Only creates mock artefacts if real ones don't already exist.
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

# Only create if missing — real files take priority
if not pathlib.Path("models/model_meta.json").exists():
    meta = {
        "threshold": 0.728,
        "feature_names": FEATURE_NAMES,
        "metrics": {"auc_roc": 0.9676, "auc_pr": 0.8208, "f1": 0.771},
        "run_id": "ci-mock-run",
    }
    pathlib.Path("models/model_meta.json").write_text(json.dumps(meta))
    print("Created mock models/model_meta.json")
else:
    print("Real model_meta.json found — skipping mock creation")

if not pathlib.Path("models/best_model.pkl").exists():
    X = pd.DataFrame(np.zeros((10, 377)), columns=FEATURE_NAMES)
    y = np.array([0] * 9 + [1])
    mock_model = XGBClassifier(n_estimators=1, max_depth=1, random_state=42)
    mock_model.fit(X, y)
    joblib.dump(mock_model, "models/best_model.pkl")
    print("Created mock models/best_model.pkl")
else:
    print("Real best_model.pkl found — skipping mock creation")

if not pathlib.Path("models/preprocessor.pkl").exists():
    from src.features.preprocessing import TargetEncoder
    artefacts = {
        "preprocessor": Pipeline([("scaler", StandardScaler())]),
        "target_encoder": TargetEncoder(cols=[]),
        "drop_cols": [],
        "col_types": {
            "numeric": FEATURE_NAMES,
            "low_card_cat": [],
        },
    }
    joblib.dump(artefacts, "models/preprocessor.pkl")
    print("Created mock models/preprocessor.pkl")
else:
    print("Real preprocessor.pkl found — skipping mock creation")

print("Artefact check complete")