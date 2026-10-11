"""Tests for the rules and the scoring API. Run from fwa_day1_data:  python -m pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
from fastapi.testclient import TestClient

from api import app
from rules import evaluate_claim

GOOD = dict(billed_amount=500.0, cost_ratio=1.0, missing_auth=0, dup_within_1_day=0, days_to_submit=3,
            provider_claims=150, provider_members=70, claims_per_member=2.1, provider_avg_cost_ratio=1.0,
            top_procedure_share=0.15, member_claims=9, member_providers=3, max_claims_30d=2)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_score_by_claim_id_has_expected_shape(client):
    r = client.post("/score-claim", json={"claim_id": "C000100"})
    assert r.status_code == 200
    body = r.json()
    for key in ("claim_id", "risk_score", "risk_level", "anomaly_score", "triggered_rules", "note"):
        assert key in body
    assert 0 <= body["risk_score"] <= 100
    assert 0 <= body["anomaly_score"] <= 1
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert "not a determination of fraud" in body["note"].lower()


def test_unknown_claim_returns_404(client):
    assert client.post("/score-claim", json={"claim_id": "NOPE"}).status_code == 404


def test_request_without_id_or_features_is_rejected(client):
    assert client.post("/score-claim", json={}).status_code == 422


def test_invalid_features_are_rejected(client):
    bad = dict(GOOD, top_procedure_share=3.0)
    assert client.post("/score-claim", json={"features": bad}).status_code == 422


def test_ordinary_claim_scores_lower_than_suspicious_one(client):
    suspicious = dict(GOOD, cost_ratio=2.4, provider_avg_cost_ratio=1.6, dup_within_1_day=1, member_providers=14)
    low = client.post("/score-claim", json={"features": GOOD}).json()
    high = client.post("/score-claim", json={"features": suspicious}).json()
    assert high["risk_score"] > low["risk_score"]
    assert "ABNORMAL_BILLING" in {h["rule"] for h in high["triggered_rules"]}


def test_rules_fire_with_evidence():
    hits = evaluate_claim(dict(GOOD, cost_ratio=2.0, dup_within_1_day=1))
    names = {h["rule"] for h in hits}
    assert {"ABNORMAL_BILLING", "DUPLICATE_CLAIM"} <= names
    assert all(h["evidence"] for h in hits)


def test_normal_claim_triggers_no_rules():
    assert evaluate_claim(GOOD) == []


# ---- dashboard endpoints ----
def test_summary_counts_add_up(client):
    s = client.get("/summary").json()
    assert s["high"] + s["medium"] + s["low"] == s["total_claims"]
    assert sum(b["count"] for b in s["risk_histogram"]) == s["total_claims"]
    assert "not a determination of fraud" in s["note"].lower()


def test_claims_list_filters_and_sorts(client):
    r = client.get("/claims", params={"level": "HIGH", "limit": 5}).json()
    assert len(r["items"]) == 5 and r["total"] >= 5
    assert all(i["risk_level"] == "HIGH" for i in r["items"])
    scores = [i["risk_score"] for i in r["items"]]
    assert scores == sorted(scores, reverse=True)


def test_claims_list_rejects_bad_level(client):
    assert client.get("/claims", params={"level": "EXTREME"}).status_code == 422


def test_claim_detail_has_evidence_and_relationships(client):
    d = client.get("/claims/C000100").json()
    assert d["claim_id"] == "C000100"
    assert all(h["evidence"] for h in d["triggered_rules"])
    assert "member_providers" in d["relationships"]
    assert "not a determination of fraud" in d["note"].lower()


def test_claim_detail_unknown_is_404(client):
    assert client.get("/claims/NOPE").status_code == 404


def test_providers_alerts_and_graph_respond(client):
    assert len(client.get("/providers").json()["items"]) > 0
    alerts = client.get("/alerts").json()
    assert all({"type", "severity", "detail"} <= set(a) for a in alerts["items"])
    g = client.get("/graph").json()
    ids = {n["id"] for n in g["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in g["edges"])
