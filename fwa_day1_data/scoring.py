"""Combined FWA risk score = rules + Isolation Forest + Random Forest.

  python scoring.py        -> scores every claim, prints evaluation, writes outputs/risk_scores.csv
  from scoring import score_batch   -> used by the FastAPI service (one claim = a 1-row DataFrame)

Design (for interview):
  risk_score (0-100) = 100 * ( W_RF * rf_probability          # supervised: knows the planted patterns
                              + W_IF * anomaly_percentile     # unsupervised: catches the unusual / unknown
                              + W_RULES * rule_points/60 )    # explicit, explainable checks (capped at 60 points)
  The score only RANKS claims for investigator review. It never says "fraud".
  Weights are a judgment call: the supervised model is the strongest on synthetic data, but we do not
  fully trust it on real data (planted patterns are easy), so it does not get all the weight.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rules import evaluate_claim

BASE = Path(__file__).parent
DATA, MODELS, OUT = BASE / "data", BASE / "models", BASE / "outputs"

W_RF, W_IF, W_RULES = 0.40, 0.25, 0.35
RULE_POINTS_CAP = 60
LEVELS = [("HIGH", 60), ("MEDIUM", 35), ("LOW", 0)]       # minimum risk_score for each level


def load_artifacts():
    meta = json.loads((MODELS / "model_meta.json").read_text())
    return (joblib.load(MODELS / "isolation_forest.joblib"),
            joblib.load(MODELS / "random_forest.joblib"), meta)


def risk_level(score: float) -> str:
    return next(name for name, floor in LEVELS if score >= floor)


def score_batch(df: pd.DataFrame, artifacts=None, rf_proba=None, anomaly_ref=None) -> pd.DataFrame:
    """Score a DataFrame of claim features. rf_proba lets the evaluation pass out-of-fold probabilities."""
    iso, rf, meta = artifacts or load_artifacts()
    X = df[meta["features"]]
    raw = -iso.score_samples(X)                                   # higher = more anomalous
    ref = np.sort(np.asarray(anomaly_ref if anomaly_ref is not None else meta["anomaly_reference"]))
    anomaly_pct = np.searchsorted(ref, raw, side="right") / len(ref)   # 0..1 percentile vs training claims
    rf_p = rf.predict_proba(X)[:, 1] if rf_proba is None else np.asarray(rf_proba)

    hits = [evaluate_claim(r) for r in X.to_dict("records")]
    points = np.array([sum(h["points"] for h in hs) for hs in hits])
    score = 100 * (W_RF * rf_p + W_IF * anomaly_pct + W_RULES * np.minimum(points, RULE_POINTS_CAP) / RULE_POINTS_CAP)

    out = pd.DataFrame({"anomaly_score": np.round(anomaly_pct, 4), "rf_probability": np.round(rf_p, 4),
                        "rule_points": points, "risk_score": np.round(score, 1)}, index=df.index)
    out["risk_level"] = out.risk_score.map(risk_level)
    out["triggered_rules"] = hits
    return out


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    iso, rf, meta = load_artifacts()
    feat = pd.read_csv(DATA / "claim_features.csv")
    labels = pd.read_csv(DATA / "claim_labels.csv")
    df = feat.merge(labels, on="claim_id", how="left")
    y = df.is_suspicious.values

    # Reference distribution so the API can turn a raw anomaly score into a percentile.
    raw_all = -iso.score_samples(df[meta["features"]])
    meta["anomaly_reference"] = [float(v) for v in np.quantile(raw_all, np.linspace(0, 1, 1001))]
    (MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2))

    # Out-of-fold RF probabilities: each claim is scored by a model that never trained on it.
    rf_cv = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
    oof = cross_val_predict(rf_cv, df[meta["features"]], y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            method="predict_proba")[:, 1]

    res = score_batch(df, (iso, rf, meta), rf_proba=oof)
    res["claim_id"], res["is_suspicious"], res["pattern_type"] = df.claim_id, y, df.pattern_type

    print("=== PR-AUC (ranking quality; random baseline = %.4f) ===" % y.mean())
    for name, s in [("Isolation Forest only", res.anomaly_score), ("Random Forest only (out-of-fold)", res.rf_probability),
                    ("Rules only", res.rule_points), ("COMBINED risk score", res.risk_score)]:
        print(f"{name:34s}{average_precision_score(y, s):.4f}")

    print("\n=== Risk levels ===")
    print(f"{'level':8s}{'claims':>8s}{'share':>8s}{'planted':>9s}{'precision':>11s}")
    for lvl in ("HIGH", "MEDIUM", "LOW"):
        m = res.risk_level == lvl
        print(f"{lvl:8s}{int(m.sum()):8d}{m.mean():8.1%}{int(res.is_suspicious[m].sum()):9d}{res.is_suspicious[m].mean():11.2f}")
    top = res.risk_level.isin(["HIGH", "MEDIUM"])
    print(f"\nHIGH+MEDIUM: {int(top.sum())} claims ({top.mean():.1%}), "
          f"precision {res.is_suspicious[top].mean():.2f}, recall {res.is_suspicious[top].sum() / y.sum():.2f}")

    print("\n=== Where each planted pattern lands (share of its claims) ===")
    pt = pd.crosstab(res.pattern_type[y == 1], res.risk_level[y == 1], normalize="index").round(2)
    print(pt.reindex(columns=["HIGH", "MEDIUM", "LOW"]).to_string())

    res["triggered_rules"] = res.triggered_rules.map(lambda hs: ";".join(h["rule"] for h in hs))
    res[["claim_id", "anomaly_score", "rf_probability", "rule_points", "risk_score", "risk_level", "triggered_rules"]] \
        .to_csv(OUT / "risk_scores.csv", index=False)
    print("\nSaved outputs/risk_scores.csv and updated models/model_meta.json")
