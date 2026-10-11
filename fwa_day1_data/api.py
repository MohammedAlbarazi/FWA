"""FWA scoring service (FastAPI).

Run:   uvicorn api:app --reload        then open http://127.0.0.1:8000/docs

POST /score-claim   score one claim, by claim_id (looked up in claim_features.csv) or by raw features
GET  /health        liveness check
GET  /summary /claims /claims/{id} /providers /alerts /graph   read-only data for the dashboard (dashboard_api.py)

Design choices (for interview):
- Models and the lookup table are loaded ONCE at startup, not on every request (fast responses).
- Pydantic validates every request, so bad input gets a clear 422 instead of a crash.
- The response always carries the reasons (triggered rules + evidence) next to the score, and a
  reminder that this is a risk score for investigator review, never a fraud verdict.
- scoring.py is the single source of truth, so the API, batch run and tests cannot disagree.
- Today the models load from local files. After a Vertex AI deployment, only load_artifacts()/the
  predict call inside scoring.py would change; this API contract stays the same.
"""
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from dashboard_api import load_dashboard_data, router as dashboard_router
from scoring import load_artifacts, score_batch

DATA = Path(__file__).parent / "data"
DISCLAIMER = "Risk score for investigator review. Not a determination of fraud."


class ClaimFeatures(BaseModel):
    billed_amount: float = Field(ge=0)
    cost_ratio: float = Field(ge=0)
    missing_auth: int = Field(ge=0, le=1)
    dup_within_1_day: int = Field(ge=0)
    days_to_submit: int = Field(ge=0)
    provider_claims: int = Field(ge=0)
    provider_members: int = Field(ge=0)
    claims_per_member: float = Field(ge=0)
    provider_avg_cost_ratio: float = Field(ge=0)
    top_procedure_share: float = Field(ge=0, le=1)
    member_claims: int = Field(ge=0)
    member_providers: int = Field(ge=0)
    max_claims_30d: int = Field(ge=0)


class ScoreRequest(BaseModel):
    claim_id: str | None = Field(default=None, examples=["C000100"])
    features: ClaimFeatures | None = None

    @model_validator(mode="after")
    def need_id_or_features(self):
        if self.claim_id is None and self.features is None:
            raise ValueError("Provide claim_id or features")
        return self


class RuleHit(BaseModel):
    rule: str
    severity: str
    pattern: str
    evidence: str


class ScoreResponse(BaseModel):
    claim_id: str | None
    risk_score: float
    risk_level: str
    anomaly_score: float
    rf_probability: float
    triggered_rules: list[RuleHit]
    note: str = DISCLAIMER


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = load_artifacts()
    feats = pd.read_csv(DATA / "claim_features.csv")
    app.state.features = feats.set_index("claim_id")
    app.state.dash = load_dashboard_data(app.state.features, app.state.artifacts)   # data for the dashboard
    yield


app = FastAPI(title="FWA Claim Scoring API", version="0.2.0", lifespan=lifespan)
# The Next.js dashboard runs on another port (3000), so the browser needs CORS permission to call this API.
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(dashboard_router)


@app.get("/health")
def health():
    return {"status": "ok", "claims_loaded": len(app.state.features)}


@app.post("/score-claim", response_model=ScoreResponse)
def score_claim(req: ScoreRequest):
    if req.features is not None:
        row = pd.DataFrame([req.features.model_dump()])
    else:
        if req.claim_id not in app.state.features.index:
            raise HTTPException(status_code=404, detail=f"Unknown claim_id {req.claim_id}")
        row = app.state.features.loc[[req.claim_id]].reset_index(drop=True)

    res = score_batch(row, app.state.artifacts).iloc[0]
    return ScoreResponse(
        claim_id=req.claim_id,
        risk_score=float(res["risk_score"]),
        risk_level=str(res["risk_level"]),
        anomaly_score=float(res["anomaly_score"]),
        rf_probability=float(res["rf_probability"]),
        triggered_rules=[RuleHit(rule=h["rule"], severity=h["severity"], pattern=h["pattern"],
                                 evidence=h["evidence"]) for h in res["triggered_rules"]],
    )
