"""Read-only endpoints that feed the investigation dashboard.

Data sources (all produced by earlier steps, nothing is invented here):
  outputs/risk_scores.csv            batch risk scores from `python scoring.py` (out-of-fold RF probabilities)
  data/claim_features.csv            claim features from BigQuery
  outputs/graph_*.csv                provider network from `python graph_analysis.py` (optional)

Design note (for interview): the dashboard shows precomputed batch scores so the page is fast and the numbers
match the evaluation. POST /score-claim (api.py) stays the live scoring endpoint for a single claim.
"""
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from rules import evaluate_claim

BASE = Path(__file__).parent
OUT = BASE / "outputs"
router = APIRouter()

NOTE = "Risk scores rank claims for investigator review. They are not a determination of fraud."


def _read(name: str) -> pd.DataFrame | None:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


def load_dashboard_data(features: pd.DataFrame, artifacts) -> dict:
    """Called once at startup. `features` is claim_features indexed by claim_id."""
    scores = _read("risk_scores.csv")
    source = "outputs/risk_scores.csv (batch, out-of-fold)"
    if scores is None:  # fallback so the API still works if scoring.py was not run
        from scoring import score_batch
        res = score_batch(features.reset_index(), artifacts)
        res["claim_id"] = features.index
        res["triggered_rules"] = res.triggered_rules.map(lambda hs: ";".join(h["rule"] for h in hs))
        scores = res[["claim_id", "anomaly_score", "rf_probability", "rule_points", "risk_score", "risk_level", "triggered_rules"]]
        source = "computed at startup (run `python scoring.py` for out-of-fold scores)"
    scores["triggered_rules"] = scores.triggered_rules.fillna("")
    claims = features.reset_index().merge(scores, on="claim_id", how="inner")
    claims["service_date"] = claims.service_date.astype(str)
    claims = claims.sort_values("risk_score", ascending=False).reset_index(drop=True)

    signals = _read("graph_provider_signals.csv")
    edges = _read("graph_edges.csv")
    clusters = _read("graph_clusters.csv")

    prov = claims.groupby("provider_id").agg(
        claims=("claim_id", "count"), members=("provider_members", "first"),
        high_claims=("risk_level", lambda s: int((s == "HIGH").sum())),
        avg_risk=("risk_score", "mean"), avg_cost_ratio=("provider_avg_cost_ratio", "first")).reset_index()
    prov["high_share"] = (prov.high_claims / prov.claims).round(3)
    prov["avg_risk"] = prov.avg_risk.round(1)
    prov["avg_cost_ratio"] = prov.avg_cost_ratio.round(2)
    prov["in_flagged_cluster"] = 0
    prov["cluster"] = None
    if signals is not None:
        s = signals.set_index("provider_id")
        prov["in_flagged_cluster"] = prov.provider_id.map(s.in_flagged_cluster).fillna(0).astype(int)
        prov["cluster"] = prov.provider_id.map(s.cluster).astype("object").where(lambda x: x.notna(), None)
    prov = prov.sort_values(["in_flagged_cluster", "high_claims"], ascending=False).reset_index(drop=True)

    return dict(claims=claims, providers=prov, edges=edges, clusters=clusters, source=source)


def _d(request: Request) -> dict:
    return request.app.state.dash


def _claim_item(row) -> dict:
    return dict(claim_id=row.claim_id, member_id=row.member_id, provider_id=row.provider_id,
                service_date=row.service_date, billed_amount=round(float(row.billed_amount), 2),
                cost_ratio=round(float(row.cost_ratio), 2), risk_score=float(row.risk_score),
                risk_level=row.risk_level, anomaly_score=float(row.anomaly_score),
                rf_probability=float(row.rf_probability),
                rules=[r for r in row.triggered_rules.split(";") if r])


@router.get("/summary")
def summary(request: Request):
    d = _d(request)
    c, p = d["claims"], d["providers"]
    levels = c.risk_level.value_counts()
    edges5 = list(range(0, 100, 5)) + [100.001]                 # 20 bins of width 5, so the 35 and 60 cut-offs fall on bin edges
    hist = pd.cut(c.risk_score, bins=edges5, right=False, include_lowest=True).value_counts().sort_index()
    hist_out = [dict(label=f"{b}-{b + 5}", start=b, count=int(n)) for b, n in zip(range(0, 100, 5), hist.values)]
    rule_counts = c.triggered_rules.str.split(";").explode()
    rule_counts = rule_counts[rule_counts != ""].value_counts()
    return dict(total_claims=len(c), high=int(levels.get("HIGH", 0)), medium=int(levels.get("MEDIUM", 0)),
                low=int(levels.get("LOW", 0)), providers=len(p),
                flagged_providers=int(p.in_flagged_cluster.sum()), alerts=len(_alerts(d)),
                risk_histogram=hist_out, rule_counts=[dict(rule=k, count=int(v)) for k, v in rule_counts.items()],
                score_source=d["source"], note=NOTE)


@router.get("/claims")
def list_claims(request: Request, level: str | None = Query(None, pattern="^(HIGH|MEDIUM|LOW)$"),
                provider_id: str | None = None, member_id: str | None = None,
                limit: int = Query(25, ge=1, le=200), offset: int = Query(0, ge=0)):
    c = _d(request)["claims"]
    if level:
        c = c[c.risk_level == level]
    if provider_id:
        c = c[c.provider_id == provider_id]
    if member_id:
        c = c[c.member_id == member_id]
    page = c.iloc[offset: offset + limit]
    return dict(total=len(c), items=[_claim_item(r) for r in page.itertuples()])


@router.get("/claims/{claim_id}")
def claim_detail(claim_id: str, request: Request):
    d = _d(request)
    c = d["claims"]
    hit = c[c.claim_id == claim_id]
    if hit.empty:
        raise HTTPException(status_code=404, detail=f"Unknown claim_id {claim_id}")
    row = hit.iloc[0]
    item = _claim_item(hit.itertuples().__next__())
    hits = evaluate_claim(row)                              # rules with evidence text
    prov = d["providers"].set_index("provider_id").loc[row.provider_id]
    peers = []
    if d["edges"] is not None and prov.in_flagged_cluster:
        e = d["edges"]
        peers = sorted(set(e[e.provider_a == row.provider_id].provider_b) | set(e[e.provider_b == row.provider_id].provider_a))
    return dict(**item, triggered_rules=[dict(rule=h["rule"], severity=h["severity"], pattern=h["pattern"],
                                              evidence=h["evidence"]) for h in hits],
                relationships=dict(member_providers=int(row.member_providers), member_claims=int(row.member_claims),
                                   provider_claims=int(row.provider_claims), provider_members=int(row.provider_members),
                                   provider_avg_cost_ratio=float(row.provider_avg_cost_ratio),
                                   provider_in_flagged_cluster=bool(prov.in_flagged_cluster),
                                   linked_providers=peers),
                note=NOTE)


@router.get("/providers")
def providers(request: Request):
    return dict(items=_d(request)["providers"].where(lambda x: x.notna(), None).to_dict("records"))


def _alerts(d: dict) -> list[dict]:
    out = []
    cl = d["clusters"]
    if cl is not None:
        for r in cl[cl.flagged].itertuples():
            out.append(dict(type="Possible provider collusion", severity="HIGH", providers=r.providers.split(";"),
                            detail=f"{r.size} providers share about {r.avg_shared_members:.0f} members each and bill "
                                   f"them at {r.avg_cost_ratio:.2f}x the normal cost on average."))
    p = d["providers"]
    for r in p[p.avg_cost_ratio >= 1.3].itertuples():
        out.append(dict(type="Provider billing outlier", severity="MEDIUM", providers=[r.provider_id],
                        detail=f"Bills {r.avg_cost_ratio:.2f}x the normal cost on average across {r.claims} claims."))
    dups = int((d["claims"].dup_within_1_day >= 1).sum())
    if dups:
        out.append(dict(type="Duplicate claims", severity="HIGH", providers=[],
                        detail=f"{dups} claims have the same member, provider and procedure within 1 day of another claim."))
    return out


@router.get("/alerts")
def alerts(request: Request):
    return dict(items=_alerts(_d(request)), note=NOTE)


@router.get("/graph")
def graph(request: Request):
    d = _d(request)
    e, p = d["edges"], d["providers"].set_index("provider_id")
    if e is None:
        return dict(nodes=[], edges=[], available=False)
    ids = sorted(set(e.provider_a) | set(e.provider_b))
    nodes = [dict(id=i, flagged=bool(p.loc[i].in_flagged_cluster), cluster=None if pd.isna(p.loc[i].cluster) else int(p.loc[i].cluster),
                  avg_cost_ratio=float(p.loc[i].avg_cost_ratio)) for i in ids]
    edges = [dict(source=r.provider_a, target=r.provider_b, shared=int(r.shared), lift=float(r.lift), cost=float(r.cost))
             for r in e.itertuples()]
    return dict(nodes=nodes, edges=edges, available=True,
                note="Providers are linked when they share far more members than chance. Flagged = linked cluster whose "
                     "shared members are also billed above normal cost. A lead for investigators, not proof.")
