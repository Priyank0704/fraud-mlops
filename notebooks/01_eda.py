"""
notebooks/01_eda.py  (run as a script or paste into Jupyter)

PURPOSE:
  Understand the data before building anything.
  The three questions you MUST answer before modelling:
    1. How severe is the class imbalance?
    2. Which features have too much missingness to be useful?
    3. Are there any obvious leakage features?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.data_loader import load_raw

df = load_raw("train")

# ── 1. CLASS IMBALANCE ───────────────────────────────────────
print("\n── Target distribution ──")
vc = df["isFraud"].value_counts()
print(vc)
print(f"Fraud rate: {vc[1] / len(df) * 100:.2f}%")
# Expect: ~3.5% fraud — severe imbalance
# Implication: NEVER use accuracy as your metric. Use AUC-PR.

# ── 2. MISSINGNESS AUDIT ─────────────────────────────────────
print("\n── Top 20 columns by missingness ──")
miss = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print(miss.head(20))
# Many V-columns (V1-V339) will be >70% missing.
# Strategy: drop columns >80% missing, impute the rest.

# ── 3. TRANSACTION AMOUNT DISTRIBUTION ───────────────────────
print("\n── TransactionAmt stats ──")
print(df["TransactionAmt"].describe())
# Expect heavy right skew — log-transform this feature.

# ── 4. TIME DISTRIBUTION ─────────────────────────────────────
# TransactionDT is seconds elapsed from a reference point (not real timestamps)
# We'll convert to hour-of-day and day-of-week as proxy cyclic features
df["hour"] = (df["TransactionDT"] // 3600) % 24
df["day"]  = (df["TransactionDT"] // (3600 * 24)) % 7

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df.groupby("hour")["isFraud"].mean().plot(ax=axes[0], title="Fraud rate by hour of day")
df.groupby("day")["isFraud"].mean().plot(ax=axes[1], title="Fraud rate by day of week")
plt.tight_layout()
plt.savefig("reports/fraud_time_patterns.png", dpi=150)
print("Saved: reports/fraud_time_patterns.png")

# ── 5. TOP CATEGORICAL FEATURES ──────────────────────────────
for col in ["ProductCD", "card4", "card6", "P_emaildomain"]:
    if col in df.columns:
        print(f"\n── {col} fraud rates ──")
        print(
            df.groupby(col)["isFraud"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
            .head(10)
        )

# ── 6. LEAKAGE CHECK ─────────────────────────────────────────
# Any feature that is determined AFTER the fraud event is leakage.
# In IEEE-CIS: TransactionID is safe (it's the key, not a feature).
# The V-columns are Vesta's anonymized features — safe.
# Watch out for: any column you engineered using the target variable
# on the full dataset (must be computed only on training fold).

print("\n✓ EDA complete. Proceed to feature engineering.")
