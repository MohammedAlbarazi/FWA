"""Train and evaluate the Isolation Forest on claim_features.csv.

Run from the folder that contains `data/`:   python train_model.py

Design notes (for interview):
- Isolation Forest is UNSUPERVISED: it never sees labels. Labels are loaded only AFTER scoring, to grade it.
- Tree-based, so features need no scaling.
- Identifier columns (claim_id, member_id, provider_id, service_date) are NOT model inputs.
- We flag the top FLAG_RATE of claims by anomaly score (a review-capacity choice),
  and also report PR-AUC, which does not depend on any threshold.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

BASE = Path(__file__).parent
DATA, MODELS, OUT = BASE / "data", BASE / "models", BASE / "outputs"
MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SEED = 42
FLAG_RATE = 0.05   # flag the top 5% most anomalous claims for investigator review

FEATURES = [
    "billed_amount", "cost_ratio", "missing_auth", "dup_within_1_day", "days_to_submit",
    "provider_claims", "provider_members", "claims_per_member",
    "provider_avg_cost_ratio", "top_procedure_share",
    "member_claims", "member_providers", "max_claims_30d",
]

# ----------------------------------------------------------------- load
df = pd.read_csv(DATA / "claim_features.csv")
missing = [c for c in FEATURES if c not in df.columns]
assert not missing, f"missing feature columns: {missing}"
assert not any("suspicious" in c or "pattern" in c for c in df.columns), "label column leaked into features!"
X = df[FEATURES]
print(f"Loaded {len(df):,} claims, {len(FEATURES)} features")

# ---------------------------------------------------------------- train
model = IsolationForest(n_estimators=200, contamination="auto", random_state=SEED, n_jobs=-1)
model.fit(X)

# score_samples: higher = more NORMAL. We flip the sign so higher = more anomalous.
raw = -model.score_samples(X)
lo, hi = float(raw.min()), float(raw.max())
anomaly_score = (raw - lo) / (hi - lo)          # 0..1, used later by the API
threshold = float(np.quantile(anomaly_score, 1 - FLAG_RATE))
flagged = (anomaly_score >= threshold).astype(int)

# ------------------------------------------------------------- evaluate
labels = pd.read_csv(DATA / "claim_labels.csv")
res = df[["claim_id"]].copy()
res["anomaly_score"] = anomaly_score
res["flagged"] = flagged
res = res.merge(labels, on="claim_id", how="left")
y = res["is_suspicious"].values

prevalence = y.mean()
pr_auc = average_precision_score(y, anomaly_score)
tn, fp, fn, tp = confusion_matrix(y, flagged).ravel()
metrics = {
    "claims": int(len(y)), "planted_suspicious": int(y.sum()), "prevalence": round(float(prevalence), 4),
    "flag_rate": FLAG_RATE, "flagged_claims": int(flagged.sum()),
    "precision": round(float(precision_score(y, flagged)), 4),
    "recall": round(float(recall_score(y, flagged)), 4),
    "f1": round(float(f1_score(y, flagged)), 4),
    "pr_auc": round(float(pr_auc), 4),
    "pr_auc_random_baseline": round(float(prevalence), 4),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}
print("\n=== Overall (threshold = top {:.0%} by anomaly score) ===".format(FLAG_RATE))
for k, v in metrics.items():
    print(f"{k:24s} {v}")

per_pattern = (res[res.is_suspicious == 1]
               .groupby("pattern_type")
               .agg(planted=("flagged", "size"), caught=("flagged", "sum")))
per_pattern["recall"] = (per_pattern.caught / per_pattern.planted).round(3)
print("\n=== Recall per planted pattern ===")
print(per_pattern.to_string())

# ----------------------------------------------------------------- save
joblib.dump(model, MODELS / "isolation_forest.joblib")
meta = {"features": FEATURES, "score_min": lo, "score_max": hi,
        "flag_threshold": threshold, "flag_rate": FLAG_RATE, "seed": SEED,
        "sklearn_trained_with": __import__("sklearn").__version__}
(MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2))
(OUT / "metrics.json").write_text(json.dumps({"overall": metrics,
                                              "per_pattern": per_pattern.reset_index().to_dict("records")}, indent=2))
res[["claim_id", "anomaly_score", "flagged"]].to_csv(OUT / "scored_claims.csv", index=False)
print("\nSaved: models/isolation_forest.joblib, models/model_meta.json, outputs/metrics.json, outputs/scored_claims.csv")
