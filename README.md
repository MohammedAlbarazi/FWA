# Healthcare FWA Detection & Investigation Platform

A portfolio project that scores **synthetic** health-insurance claims for Fraud, Waste & Abuse (FWA) risk and presents them to an investigator. Built end to end: SQL feature engineering in BigQuery, rules + anomaly detection + a supervised model, graph analysis, a FastAPI scoring service, automated tests and a Next.js dashboard.

> **Principle:** an anomaly is not confirmed fraud. The system outputs a risk score, an anomaly score, the rules that fired and the evidence. Every output is a *case for investigator review*.

All data is synthetic. No real patient data is used. Results below show the pipeline works on planted patterns; they are not evidence of performance on real claims.

## What is built, and what is not

| Area | Status |
|---|---|
| Synthetic data generator (members, providers, doctors, procedures, authorizations, claims) with planted FWA patterns | Done |
| BigQuery (sandbox) tables and SQL feature engineering | Done |
| Isolation Forest (unsupervised anomaly detection) | Done |
| Random Forest (supervised, synthetic labels) | Done |
| Rule engine (7 explainable rules with evidence) | Done |
| Combined risk score and risk levels | Done |
| Graph analytics (NetworkX) for provider collusion | Done |
| FastAPI scoring service (`POST /score-claim`) + dashboard data endpoints | Done |
| Automated tests (pytest, 14 tests) | Done |
| Next.js investigation dashboard | Done |
| **Vertex AI Model Registry / endpoint** | **Not deployed.** Needs a billing account and a US$30 prepayment; I chose not to. See "Path to Vertex AI". |
| Cloud Storage, Cloud Run, IAM/service accounts, logging/monitoring | Not done |
| OCR, LLM-written explanations, automated retraining, CI/CD | Not done |

## Architecture (as built)

```
generate_data.py  ->  7 CSV files (synthetic)
        |
        v
BigQuery sandbox (dataset Fwa)  -- SQL: claim, provider and member features -->  claim_features (19,660 rows)
        |  exported to data/claim_features.csv
        v
train_model.py        Isolation Forest  (unsupervised, 13 features)
train_supervised.py   Random Forest     (70/30 stratified split)
rules.py              7 rules with evidence text
scoring.py            combined risk score = 0.40 RF + 0.25 anomaly percentile + 0.35 rules
graph_analysis.py     provider-member graph, clusters, collusion signal
        |
        v
api.py + dashboard_api.py   FastAPI
        |
        v
dashboard/            Next.js investigation dashboard
```

Everything runs locally except BigQuery. Models were trained locally to keep cloud cost at about zero.

## Data

Seeded generator (`generate_data.py`, seed 42): 2,000 members, 100 providers, 250 doctors, 40 procedures, 3,347 authorizations, 19,660 claims. **932 claims (4.74%)** carry a planted pattern, stored in a separate answer-key file that is never used as a feature:

| Pattern | Claims |
|---|---|
| Abnormal billing (upcoding providers) | 272 |
| Service overutilization | 180 |
| Procedure frequency spike | 180 |
| Provider/member collusion ring (4 providers, 25 members) | 150 |
| Duplicate claims | 150 |

## Features (SQL in BigQuery)

13 features per claim, built with CTEs, window functions and joins: `billed_amount`, `cost_ratio` (billed vs normal cost for the procedure), `missing_auth`, `dup_within_1_day`, `days_to_submit`, provider-level (`provider_claims`, `provider_members`, `claims_per_member`, `provider_avg_cost_ratio`, `top_procedure_share`) and member-level (`member_claims`, `member_providers`, `max_claims_30d`).

## Results (my runs, synthetic data)

The data is imbalanced (4.74% suspicious), so accuracy is not used. A model that flags nothing would score about 95% accuracy.

| Method | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Isolation Forest (top 5% flagged, all claims) | 0.39 | 0.41 | 0.40 | 0.42 |
| Random Forest (30% held-out test, 5,898 claims) | 0.64 | 0.92 | 0.75 | 0.85 |
| Rules (any rule fires) | 0.17 | 0.79 | n/a | n/a |

Random baseline PR-AUC is 0.047. Recall by planted pattern is in `train_model.py` / `train_supervised.py` output; Isolation Forest finds abnormal billing well (0.95) but misses frequency spikes (0.01), which is why rules and a supervised model are combined.

**Combined score** (out-of-fold Random Forest probabilities, all 19,660 claims):

| Level | Claims | Share | Precision |
|---|---|---|---|
| HIGH (score >= 60) | 1,002 | 5.1% | 0.66 |
| MEDIUM (35-60) | 1,192 | 6.1% | 0.22 |
| LOW | 17,466 | | |

HIGH + MEDIUM is 11.2% of claims with recall 0.99 and precision 0.42. In ranking terms (PR-AUC) the combined score is 0.77, **below the Random Forest alone (0.85)**. On synthetic data with easy planted patterns the supervised model is strongest. The blend is kept for explainability (every score comes with rules and evidence) and so that the system does not depend only on a supervised model that would need real labels.

**Graph analysis:** providers are linked when they share far more members than chance, then clustered. A cluster is flagged only if its shared members are also billed above normal cost. It found all 4 planted collusion providers and also flagged 2 providers that are not planted (false positives). Large same-city providers share many members but were correctly not flagged because their cost ratio is about 1.0.

## Limitations (please read)

- Synthetic data with planted patterns. Strong scores show the pipeline works, not that it would catch real fraud.
- Random Forest numbers are probably optimistic: the same providers and members appear in train and test, and provider-level features act like fingerprints.
- Rule thresholds, level cut-offs (35, 60) and score weights were chosen after looking at this data. Real use needs tuning with investigators and re-measuring false alarms.
- Isolation Forest is scored on the same data it was trained on.
- Rules such as high claim frequency also fire for chronic patients. A flag means "worth a look".
- `POST /score-claim` uses the trained (in-sample) Random Forest for claims it was trained on, so a claim's live score can differ slightly from the dashboard's batch score (for example 93.4 vs 93.3).
- BigQuery sandbox tables expire after 60 days.

## Run it

```
pip install pandas numpy scikit-learn joblib networkx matplotlib fastapi uvicorn pytest httpx
python generate_data.py          # synthetic data (features come from the BigQuery SQL export)
python train_model.py            # Isolation Forest
python train_supervised.py       # Random Forest
python rules.py                  # rule evaluation
python scoring.py                # combined score -> outputs/risk_scores.csv
python graph_analysis.py         # provider network
python -m pytest -q              # 14 tests
uvicorn api:app --reload         # API on :8000, docs at /docs
cd dashboard && npm install && npm run dev    # dashboard on :3000
```

Example: `POST /score-claim` with `{"claim_id": "C000100"}` returns `risk_score`, `risk_level`, `anomaly_score`, `rf_probability`, `triggered_rules` (with evidence) and a note that the output is not a determination of fraud.

## Path to Vertex AI (design, not implemented)

The scoring logic is isolated in `scoring.load_artifacts()` / `score_batch()`. To move to Vertex AI: export the trained model to Cloud Storage, upload it to the Vertex AI Model Registry, deploy it to a small endpoint, and change the model call inside `scoring.py` to call that endpoint. The API contract and dashboard would not change. Endpoints are billed per hour while deployed, so the endpoint should be undeployed after any demo. The scikit-learn pre-built container may need a custom prediction step to return anomaly scores; that is unverified.

## Project files

`generate_data.py` · `train_model.py` · `train_supervised.py` · `rules.py` · `scoring.py` · `graph_analysis.py` · `api.py` · `dashboard_api.py` · `test_api.py` · `dashboard/` · `INTERVIEW_NOTES.md`
