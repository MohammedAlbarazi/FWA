"use client";
import { useEffect, useMemo, useState } from "react";
import { get } from "./api";

export const fmt = (n) => Number(n).toLocaleString("en-US");

export function Badge({ level }) {
  return <span className={`badge ${level}`}>{level}</span>;
}

/* ---------- Risk score histogram (20 bins of width 5; cut-offs 35 and 60 fall on bin edges) ---------- */
export function RiskHistogram({ bins }) {
  const [hover, setHover] = useState(null);
  const [asTable, setAsTable] = useState(false);
  const W = 640, H = 250, L = 46, R = 10, T = 14, B = 44;
  const max = Math.max(...bins.map((b) => b.count), 1);
  const step = Math.pow(10, Math.floor(Math.log10(max)));
  const top = Math.ceil(max / step) * step;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round((top * f) / 10) * 10);
  const bw = (W - L - R) / bins.length;
  const y = (v) => T + (H - T - B) * (1 - v / top);
  const x = (score) => L + (score / 100) * (W - L - R);

  if (asTable)
    return (
      <div>
        <div className="tablewrap" style={{ maxHeight: 260, overflowY: "auto" }}>
          <table>
            <thead><tr><th>Risk score</th><th className="num">Claims</th></tr></thead>
            <tbody>{bins.map((b) => <tr key={b.label}><td>{b.label}</td><td className="num">{fmt(b.count)}</td></tr>)}</tbody>
          </table>
        </div>
        <button className="linkbtn" onClick={() => setAsTable(false)}>Show chart</button>
      </div>
    );

  return (
    <div style={{ position: "relative" }}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Number of claims at each risk score, in bins of 5 points">
        {ticks.map((t) => (
          <g key={t}>
            <line className="axis" x1={L} x2={W - R} y1={y(t)} y2={y(t)} />
            <text x={L - 6} y={y(t) + 4} textAnchor="end">{fmt(t)}</text>
          </g>
        ))}
        {bins.map((b, i) => {
          const h = y(0) - y(b.count);
          const bx = L + i * bw + 1;
          return (
            <g key={b.label} onMouseEnter={() => setHover({ b, cx: L + i * bw + bw / 2, cy: y(b.count) })}
               onMouseLeave={() => setHover(null)}>
              <rect x={L + i * bw} y={T} width={bw} height={H - T - B} fill="transparent" />
              <rect x={bx} y={y(b.count)} width={bw - 2} height={Math.max(h, b.count ? 1 : 0)} rx="3"
                    fill={hover && hover.b === b ? "var(--ink)" : "var(--bar)"} />
            </g>
          );
        })}
        {[35, 60].map((c) => (
          <line key={c} className="band" x1={x(c)} x2={x(c)} y1={T} y2={H - B + 6} />
        ))}
        <text x={(x(0) + x(35)) / 2} y={H - 10} textAnchor="middle">LOW (&lt;35)</text>
        <text x={(x(35) + x(60)) / 2} y={H - 10} textAnchor="middle">MEDIUM (35-60)</text>
        <text x={(x(60) + x(100)) / 2} y={H - 10} textAnchor="middle">HIGH (60+)</text>
        {[0, 20, 40, 60, 80, 100].map((s) => (
          <text key={s} x={x(s)} y={H - B + 18} textAnchor="middle">{s}</text>
        ))}
      </svg>
      {hover && (
        <div className="tip" style={{ left: `${(hover.cx / W) * 100}%`, top: `${(hover.cy / H) * 100}%` }}>
          Score {hover.b.label}: {fmt(hover.b.count)} claims
        </div>
      )}
      <button className="linkbtn" onClick={() => setAsTable(true)}>Show as table</button>
    </div>
  );
}

/* ---------- Provider network: providers linked when they share far more members than chance ---------- */
export function NetworkGraph({ graph, onPickProvider }) {
  const [hover, setHover] = useState(null);
  const W = 620, H = 460;
  const pos = useMemo(() => {
    const groups = {};
    graph.nodes.forEach((n) => (groups[n.cluster ?? "x"] ||= []).push(n));
    const keys = Object.keys(groups);
    const out = {};
    keys.forEach((k, gi) => {
      const a = (2 * Math.PI * gi) / keys.length - Math.PI / 2;
      const cx = W / 2 + Math.cos(a) * 235, cy = H / 2 + Math.sin(a) * 160;
      const members = groups[k];
      const r = members.length === 1 ? 0 : 20 + members.length * 5;
      members.forEach((n, i) => {
        const b = (2 * Math.PI * i) / members.length - Math.PI / 2;
        out[n.id] = { x: cx + Math.cos(b) * r, y: cy + Math.sin(b) * r };
      });
    });
    return out;
  }, [graph]);

  if (!graph.available)
    return <p className="hint">Run <span className="mono">python graph_analysis.py</span> to create the network data, then restart the API.</p>;

  const maxShared = Math.max(...graph.edges.map((e) => e.shared), 1);
  return (
    <div>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Network of providers that share many members. Flagged providers are drawn with a red ring.">
        {graph.edges.map((e, i) => {
          const a = pos[e.source], b = pos[e.target];
          const hot = hover && (hover.id === e.source || hover.id === e.target);
          return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={hot ? "var(--ink)" : "var(--muted)"}
                       strokeOpacity={hot ? 0.9 : 0.4} strokeWidth={0.8 + (e.shared / maxShared) * 3} />;
        })}
        {graph.nodes.map((n) => {
          const p = pos[n.id];
          return (
            <g key={n.id} tabIndex={0} style={{ cursor: "pointer" }} onClick={() => onPickProvider(n.id)}
               onKeyDown={(e) => e.key === "Enter" && onPickProvider(n.id)}
               onMouseEnter={() => setHover(n)} onMouseLeave={() => setHover(null)}
               onFocus={() => setHover(n)} onBlur={() => setHover(null)}>
              <circle cx={p.x} cy={p.y} r={15} fill="var(--surface)" stroke={n.flagged ? "var(--high)" : "var(--bar)"}
                      strokeWidth={n.flagged ? 3 : 1.5} />
              <text x={p.x} y={p.y + 3} textAnchor="middle" style={{ fontSize: 9, fontWeight: n.flagged ? 700 : 400 }}>{n.id}</text>
            </g>
          );
        })}
      </svg>
      <div className="legend">
        <span><span className="dot" style={{ border: "3px solid var(--high)", background: "var(--surface)" }} />Flagged cluster</span>
        <span><span className="dot" style={{ border: "1.5px solid var(--bar)", background: "var(--surface)" }} />Linked, not flagged</span>
        <span>Line thickness = shared members. Click a provider to filter claims.</span>
      </div>
      <p className="hint" style={{ minHeight: 18, margin: "6px 0 0" }}>
        {hover ? `${hover.id}: ${hover.flagged ? "in a flagged cluster" : "not flagged"}, cluster ${hover.cluster}, bills ${hover.avg_cost_ratio.toFixed(2)}x normal cost on average`
               : "Hover a provider for details."}
      </p>
    </div>
  );
}

/* ---------- Claim detail drawer with the investigator review step ---------- */
const STATUSES = ["Needs investigation", "Request more info", "Dismiss as normal"];

export function ClaimDrawer({ claimId, onClose, onPickProvider }) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [review, setReview] = useState({ status: null, note: "" });

  useEffect(() => {
    setD(null); setErr(null);
    get(`/claims/${claimId}`).then(setD).catch((e) => setErr(e.message));
    try {
      const all = JSON.parse(localStorage.getItem("fwa-review") || "{}");
      setReview(all[claimId] || { status: null, note: "" });
    } catch { setReview({ status: null, note: "" }); }
  }, [claimId]);

  useEffect(() => {
    const h = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);

  const save = (next) => {
    setReview(next);
    try {
      const all = JSON.parse(localStorage.getItem("fwa-review") || "{}");
      all[claimId] = next;
      localStorage.setItem("fwa-review", JSON.stringify(all));
    } catch { /* storage can be blocked; the review then lasts only for this page view */ }
  };

  return (
    <>
      <div className="drawer-bg" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label={`Claim ${claimId}`}>
        <div className="dhead">
          <div>
            <h3 className="mono">{claimId}</h3>
            {d && <div className="hint" style={{ margin: "4px 0 0" }}>{d.service_date} · member {d.member_id} · provider {d.provider_id}</div>}
          </div>
          <button className="close" onClick={onClose} aria-label="Close">✕</button>
        </div>
        {err && <div className="err" style={{ marginTop: 16 }}>{err}</div>}
        {!d && !err && <p className="hint">Loading…</p>}
        {d && (
          <>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 16 }}>
              <span className="bigscore">{d.risk_score.toFixed(1)}</span><Badge level={d.risk_level} />
            </div>
            <div className="stats">
              <div className="stat"><div className="k">Anomaly percentile</div><div className="v">{(d.anomaly_score * 100).toFixed(1)}%</div></div>
              <div className="stat"><div className="k">Supervised model</div><div className="v">{(d.rf_probability * 100).toFixed(1)}%</div></div>
              <div className="stat"><div className="k">Cost vs normal</div><div className="v">{d.cost_ratio.toFixed(2)}x</div></div>
            </div>

            <h4>Why it was flagged ({d.triggered_rules.length} rule{d.triggered_rules.length === 1 ? "" : "s"})</h4>
            {d.triggered_rules.length === 0 && <p className="hint">No rules fired. The score comes from the models only.</p>}
            {d.triggered_rules.map((r) => (
              <div className="rule" key={r.rule}>
                <div className="rn"><span className="badge LOW" style={r.severity === "HIGH" ? { color: "var(--high)", background: "var(--high-bg)" } : r.severity === "MEDIUM" ? { color: "var(--med)", background: "var(--med-bg)" } : undefined}>{r.severity}</span>{r.rule}</div>
                <div className="ev">{r.evidence}</div>
              </div>
            ))}

            <h4>Member and provider relationships</h4>
            <dl className="kv">
              <dt>Member visited providers</dt><dd>{d.relationships.member_providers}</dd>
              <dt>Member total claims</dt><dd>{d.relationships.member_claims}</dd>
              <dt>Provider total claims</dt><dd>{fmt(d.relationships.provider_claims)}</dd>
              <dt>Provider distinct members</dt><dd>{d.relationships.provider_members}</dd>
              <dt>Provider avg cost vs normal</dt><dd>{d.relationships.provider_avg_cost_ratio.toFixed(2)}x</dd>
              <dt>Provider in flagged cluster</dt><dd>{d.relationships.provider_in_flagged_cluster ? "Yes" : "No"}</dd>
            </dl>
            {d.relationships.linked_providers.length > 0 && (
              <p className="hint" style={{ marginTop: 8 }}>
                Linked providers (share many members): {d.relationships.linked_providers.join(", ")}
              </p>
            )}
            <button className="linkbtn" onClick={() => { onPickProvider(d.provider_id); onClose(); }}>
              Show all claims from {d.provider_id}
            </button>

            <h4>Investigator review</h4>
            <div className="tabs">
              {STATUSES.map((s) => (
                <button key={s} className="tab" aria-pressed={review.status === s} onClick={() => save({ ...review, status: s })}>{s}</button>
              ))}
            </div>
            <textarea style={{ marginTop: 8 }} placeholder="Notes for the case file…" value={review.note}
                      onChange={(e) => save({ ...review, note: e.target.value })} />
            <div className="hint">Saved in this browser only (a prototype of the review step).</div>

            <div className="disclaimer">{d.note}</div>
          </>
        )}
      </aside>
    </>
  );
}
