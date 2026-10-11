"use client";
import { useEffect, useState } from "react";
import { get } from "./api";
import { Badge, ClaimDrawer, NetworkGraph, RiskHistogram, fmt } from "./components";

const PAGE = 12;

export default function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [providers, setProviders] = useState([]);
  const [graph, setGraph] = useState(null);
  const [error, setError] = useState(null);

  const [level, setLevel] = useState("HIGH");
  const [providerFilter, setProviderFilter] = useState(null);
  const [offset, setOffset] = useState(0);
  const [claims, setClaims] = useState({ total: 0, items: [] });
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    Promise.all([get("/summary"), get("/alerts"), get("/providers"), get("/graph")])
      .then(([s, a, p, g]) => { setSummary(s); setAlerts(a.items); setProviders(p.items); setGraph(g); })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    const q = new URLSearchParams({ limit: PAGE, offset });
    if (level !== "ALL") q.set("level", level);
    if (providerFilter) q.set("provider_id", providerFilter);
    get(`/claims?${q}`).then(setClaims).catch((e) => setError(e.message));
  }, [level, providerFilter, offset]);

  const pickProvider = (id) => { setProviderFilter(id); setLevel("ALL"); setOffset(0); };
  const pickLevel = (l) => { setLevel(l); setOffset(0); };

  if (error)
    return (
      <div className="wrap">
        <div className="err">
          Could not reach the API ({error}). Start it in the project folder with <span className="mono">uvicorn api:app --reload</span>, then refresh.
        </div>
      </div>
    );
  if (!summary || !graph) return <div className="wrap"><p className="sub">Loading…</p></div>;

  const pct = ((summary.high / summary.total_claims) * 100).toFixed(1);
  const topProviders = providers.slice(0, 12);

  return (
    <main className="wrap">
      <div className="top">
        <div>
          <h1>FWA Investigation Dashboard</h1>
          <p className="sub">Claims ranked by a combined risk score (rules, anomaly detection, supervised model). A high score is a lead for an investigator to review. It is not a finding of fraud.</p>
        </div>
        <span className="pill">Synthetic data · {summary.score_source}</span>
      </div>

      <section className="grid kpis" aria-label="Key numbers">
        <div className="card kpi"><div className="label">Total claims</div><div className="value">{fmt(summary.total_claims)}</div><div className="note">{fmt(summary.medium)} medium · {fmt(summary.low)} low</div></div>
        <div className="card kpi"><div className="label">High-risk claims</div><div className="value">{fmt(summary.high)}</div><div className="note">{pct}% of all claims</div></div>
        <div className="card kpi"><div className="label">Providers</div><div className="value">{fmt(summary.providers)}</div><div className="note">{summary.flagged_providers} in a flagged network cluster</div></div>
        <div className="card kpi"><div className="label">Alerts</div><div className="value">{summary.alerts}</div><div className="note">collusion, billing outliers, duplicates</div></div>
      </section>

      <section className="grid two">
        <div className="card">
          <div className="card-head"><h2>Risk score distribution</h2></div>
          <p className="hint">Number of claims at each risk score. Most claims score low; the few at the right are the review queue.</p>
          <RiskHistogram bins={summary.risk_histogram} />
        </div>
        <div className="card">
          <div className="card-head"><h2>Alerts</h2></div>
          <p className="hint">Patterns that need a human look.</p>
          <div className="scroll">
            {alerts.map((a, i) => (
              <div className="alert" key={i}>
                <div><Badge level={a.severity} /></div>
                <div>
                  <div className="t">{a.type}</div>
                  <div className="d">{a.detail}</div>
                  {a.providers.length > 0 && (
                    <div className="p">{a.providers.map((p) => <span key={p} className="mono">{p}</span>)}</div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="card" style={{ marginBottom: 16 }}>
        <div className="card-head">
          <h2>Suspicious claims</h2>
          <div className="tabs" role="group" aria-label="Filter by risk level">
            {["HIGH", "MEDIUM", "ALL"].map((l) => (
              <button key={l} className="tab" aria-pressed={level === l} onClick={() => pickLevel(l)}>{l === "ALL" ? "All" : l[0] + l.slice(1).toLowerCase()}</button>
            ))}
          </div>
        </div>
        <p className="hint" style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          Click a row for the evidence. Sorted by risk score.
          {providerFilter && <span className="chip">Provider {providerFilter}<button aria-label="Clear provider filter" onClick={() => { setProviderFilter(null); setOffset(0); }}>✕</button></span>}
        </p>
        <div className="tablewrap">
          <table>
            <thead>
              <tr><th>Claim</th><th>Date</th><th>Provider</th><th>Member</th><th className="num">Billed</th><th className="num">Cost vs normal</th><th>Risk</th><th>Level</th><th>Rules triggered</th></tr>
            </thead>
            <tbody>
              {claims.items.map((c) => (
                <tr key={c.claim_id} className={`click ${selected === c.claim_id ? "sel" : ""}`} tabIndex={0}
                    onClick={() => setSelected(c.claim_id)} onKeyDown={(e) => e.key === "Enter" && setSelected(c.claim_id)}>
                  <td className="mono">{c.claim_id}</td>
                  <td className="mono">{c.service_date}</td>
                  <td className="mono">{c.provider_id}</td>
                  <td className="mono">{c.member_id}</td>
                  <td className="num">{fmt(Math.round(c.billed_amount))}</td>
                  <td className="num">{c.cost_ratio.toFixed(2)}x</td>
                  <td><span className="scorebar"><span className="track"><span className="fill" style={{ width: `${c.risk_score}%` }} /></span>{c.risk_score.toFixed(1)}</span></td>
                  <td><Badge level={c.risk_level} /></td>
                  <td className="rules">{c.rules.length ? c.rules.map((r) => r.replaceAll("_", " ").toLowerCase()).join(", ") : "none"}</td>
                </tr>
              ))}
              {claims.items.length === 0 && <tr><td colSpan={9} className="hint">No claims match this filter.</td></tr>}
            </tbody>
          </table>
        </div>
        <div className="pager">
          <span>{claims.total === 0 ? 0 : offset + 1}-{Math.min(offset + PAGE, claims.total)} of {fmt(claims.total)}</span>
          <span style={{ display: "flex", gap: 8 }}>
            <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</button>
            <button disabled={offset + PAGE >= claims.total} onClick={() => setOffset(offset + PAGE)}>Next</button>
          </span>
        </div>
      </section>

      <section className="grid two-b">
        <div className="card">
          <div className="card-head"><h2>Providers to review</h2></div>
          <p className="hint">Flagged network clusters first, then by number of high-risk claims. Click a row to see its claims.</p>
          <div className="tablewrap">
            <table>
              <thead><tr><th>Provider</th><th className="num">Claims</th><th className="num">High-risk</th><th className="num">Avg cost vs normal</th><th>Network</th></tr></thead>
              <tbody>
                {topProviders.map((p) => (
                  <tr key={p.provider_id} className="click" tabIndex={0} onClick={() => pickProvider(p.provider_id)}
                      onKeyDown={(e) => e.key === "Enter" && pickProvider(p.provider_id)}>
                    <td className="mono">{p.provider_id}</td>
                    <td className="num">{fmt(p.claims)}</td>
                    <td className="num">{p.high_claims} ({(p.high_share * 100).toFixed(0)}%)</td>
                    <td className="num">{p.avg_cost_ratio.toFixed(2)}x</td>
                    <td>{p.in_flagged_cluster ? <Badge level="HIGH" /> : <span className="hint">-</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="card">
          <div className="card-head"><h2>Provider relationship network</h2></div>
          <p className="hint">Providers are linked when they share far more members than chance would give. A cluster is flagged only if those members are also billed above normal cost.</p>
          <NetworkGraph graph={graph} onPickProvider={pickProvider} />
        </div>
      </section>

      <p className="footer">Prototype built on synthetic claims. Scores come from batch scoring (out-of-fold supervised probabilities); live single-claim scoring is POST /score-claim on the API.</p>
      {selected && <ClaimDrawer claimId={selected} onClose={() => setSelected(null)} onPickProvider={pickProvider} />}
    </main>
  );
}
