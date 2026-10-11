"""Supervised Random Forest on the same features, using the synthetic labels.

Run after train_model.py:   python train_supervised.py

Differences from the Isolation Forest (for interview):
- This model USES labels (supervised), so we must hold out a TEST set it never trains on.
- Labels are the target (y); they are never in the feature columns (X).
- class_weight="balanced" because only ~4.7% of claims are suspicious.
- Caveat: a random split puts claims from the same provider/member in both train and test,
  so provider/member features can make results look better than on brand-new providers.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
DATA, MODELS, OUT = BASE / "data", BASE / "models", BASE / "outputs"
MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
SEED = 42

FEATURES = [
    "billed_amount", "cost_ratio", "missing_auth", "dup_within_1_day", "days_to_submit",
    "provider_claims", "provider_members", "claims_per_member",
    "provider_avg_cost_ratio", "top_procedure_share",
    "member_claims", "member_providers", "max_claims_30d",
]

feat = pd.read_csv(DATA / "claim_features.csv")
labels = pd.read_csv(DATA / "claim_labels.csv")
df = feat.merge(labels, on="claim_id", how="left")
X, y = df[FEATURES], df["is_suspicious"]
assert not any("suspicious" in c or "pattern" in c for c in FEATURES), "label in features!"

# Stratified split keeps ~4.7% suspicious in both train and test.
idx_train, idx_test = train_test_split(df.index, test_size=0.3, stratify=y, random_state=SEED)
X_train, X_test, y_train, y_test = X.loc[idx_train], X.loc[idx_test], y.loc[idx_train], y.loc[idx_test]
print(f"Train: {len(X_train):,} claims ({int(y_train.sum())} suspicious) | Test: {len(X_test):,} claims ({int(y_test.sum())} suspicious)")

model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                               random_state=SEED, n_jobs=-1)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]      # probability that a claim is suspicious
pred = (proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
prev = float(y_test.mean())
metrics = {
    "test_claims": int(len(y_test)), "test_suspicious": int(y_test.sum()),
    "threshold": 0.5,
    "precision": round(float(precision_score(y_test, pred)), 4),
    "recall": round(float(recall_score(y_test, pred)), 4),
    "f1": round(float(f1_score(y_test, pred)), 4),
    "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
    "pr_auc_random_baseline": round(prev, 4),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}
print("\n=== Random Forest, TEST set only (claims it never trained on) ===")
for k, v in metrics.items():
    print(f"{k:24s} {v}")

# Same test rows, Isolation Forest, for a fair comparison.
scored = pd.read_csv(OUT / "scored_claims.csv").set_index("claim_id")
if_scores = scored.loc[df.loc[idx_test, "claim_id"], "anomaly_score"].values
print(f"\nIsolation Forest PR-AUC on the SAME test rows: {average_precision_score(y_test, if_scores):.4f}")

res = df.loc[idx_test, ["pattern_type"]].copy()
res["y"], res["caught"] = y_test.values, pred
pp = res[res.y == 1].groupby("pattern_type").agg(planted=("caught", "size"), caught=("caught", "sum"))
pp["recall"] = (pp.caught / pp.planted).round(3)
print("\n=== Recall per planted pattern (test set) ===")
print(pp.to_string())

imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False).round(3)
print("\n=== Feature importance (which features the forest relied on) ===")
print(imp.to_string())

joblib.dump(model, MODELS / "random_forest.joblib")
(OUT / "metrics_supervised.json").write_text(json.dumps(
    {"overall": metrics, "per_pattern": pp.reset_index().to_dict("records"),
     "feature_importance": imp.to_dict()}, indent=2))
print("\nSaved: models/random_forest.joblib, outputs/metrics_supervised.json")
