"""FWA rule engine: explicit, explainable checks on one claim's feature row.

Used two ways:
  * imported by the scoring API:   from rules import evaluate_claim
  * run directly to evaluate the rules against the answer key:   python rules.py

A rule firing means "worth a look", NOT "fraud". Each rule returns evidence an investigator can read.
Thresholds were chosen from the synthetic data (see the numbers printed by `python rules.py`);
in a real deployment they would be tuned with investigators and re-measured for false alarms.
"""
from dataclasses import dataclass
from typing import Callable, Mapping

SEVERITY_POINTS = {"HIGH": 30, "MEDIUM": 20, "LOW": 10}   # used later by the risk score


@dataclass(frozen=True)
class Rule:
    name: str
    severity: str                                   # HIGH / MEDIUM / LOW
    pattern: str                                    # which FWA pattern it targets
    check: Callable[[Mapping], bool]
    evidence: Callable[[Mapping], str]


RULES = [
    Rule("DUPLICATE_CLAIM", "HIGH", "duplicate claims",
         lambda r: r["dup_within_1_day"] >= 1,
         lambda r: f"{int(r['dup_within_1_day'])} other claim(s) with the same member, provider and procedure within 1 day"),
    Rule("ABNORMAL_BILLING", "HIGH", "abnormal billing",
         lambda r: r["cost_ratio"] >= 1.5,
         lambda r: f"Billed {r['cost_ratio']:.2f}x the normal cost for this procedure"),
    Rule("PROVIDER_COST_OUTLIER", "MEDIUM", "unusual provider behavior",
         lambda r: r["provider_avg_cost_ratio"] >= 1.3,
         lambda r: f"Provider bills {r['provider_avg_cost_ratio']:.2f}x the normal cost on average"),
    Rule("MEMBER_PROVIDER_HOPPING", "MEDIUM", "suspicious member/provider relationships",
         lambda r: r["member_providers"] >= 12,
         lambda r: f"Member visited {int(r['member_providers'])} different providers"),
    Rule("PROVIDER_PROCEDURE_CONCENTRATION", "LOW", "unusual procedure frequency",
         lambda r: r["top_procedure_share"] >= 0.35,
         lambda r: f"{r['top_procedure_share']:.0%} of this provider's claims are one procedure"),
    Rule("MEMBER_HIGH_FREQUENCY", "LOW", "service overutilization",
         lambda r: r["max_claims_30d"] >= 7,
         lambda r: f"Member had up to {int(r['max_claims_30d'])} claims within 30 days (chronic patients can also trigger this)"),
    Rule("MISSING_AUTHORIZATION", "LOW", "abnormal billing / pre-authorization",
         lambda r: r["missing_auth"] == 1,
         lambda r: "Procedure requires pre-authorization but none is on file"),
]


def evaluate_claim(row: Mapping) -> list[dict]:
    """Return the rules that fire for one claim, with severity and evidence."""
    return [{"rule": rule.name, "severity": rule.severity, "pattern": rule.pattern,
             "points": SEVERITY_POINTS[rule.severity], "evidence": rule.evidence(row)}
            for rule in RULES if rule.check(row)]


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    data = Path(__file__).parent / "data"
    df = pd.read_csv(data / "claim_features.csv").merge(pd.read_csv(data / "claim_labels.csv"), on="claim_id")
    y = df["is_suspicious"]
    fired = {r.name: df.apply(r.check, axis=1) for r in RULES}

    print(f"{'rule':34s}{'sev':8s}{'fires':>7s}{'precision':>11s}{'recall':>8s}")
    for r in RULES:
        m = fired[r.name]
        tp = int((m & (y == 1)).sum())
        print(f"{r.name:34s}{r.severity:8s}{int(m.sum()):7d}{tp / max(int(m.sum()), 1):11.2f}{tp / int(y.sum()):8.2f}")

    any_rule = pd.concat(fired, axis=1).any(axis=1)
    print("\nShare of each planted pattern caught by AT LEAST ONE rule:")
    cov = any_rule[y == 1].groupby(df.loc[y == 1, "pattern_type"]).mean().round(3)
    print(cov.to_string())
    print(f"\nClaims with at least one rule fired: {int(any_rule.sum())} of {len(df)} "
          f"(precision {(any_rule & (y == 1)).sum() / any_rule.sum():.2f}, recall {(any_rule & (y == 1)).sum() / y.sum():.2f})")
    ex = df[df.pattern_type == "abnormal_billing"].iloc[0]
    print("\nExample (one abnormal_billing claim):", ex["claim_id"])
    for hit in evaluate_claim(ex):
        print(f"  [{hit['severity']}] {hit['rule']}: {hit['evidence']}")
