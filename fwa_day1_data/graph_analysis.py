"""Graph analytics for provider collusion (NetworkX).

Run:   python graph_analysis.py        (optional chart needs: pip install matplotlib)

Idea: build a graph of providers and the members they bill. Providers that share an unusually
large set of members are linked. Collusion shows up as a tight cluster of providers sharing the
same members AND billing those members above the normal cost.

Why two signals? Shared members alone also flag large honest providers in the same big city
(they overlap by chance). So a cluster is flagged only if its shared members are ALSO billed
above normal cost. A flagged cluster is a lead for investigators, not proof of collusion.
Thresholds were chosen from the synthetic data.
"""
from pathlib import Path
import itertools

import networkx as nx
import pandas as pd

BASE = Path(__file__).parent
DATA, OUT = BASE / "data", BASE / "outputs"
OUT.mkdir(exist_ok=True)

MIN_SHARED = 12        # providers must share at least this many members to be linked
MIN_LIFT = 3.0         # ...and share >= 3x more members than chance would give
MIN_COST_RATIO = 1.2   # cluster flagged if shared members are billed >= 1.2x normal cost
MIN_CLUSTER = 3        # a "ring" needs at least 3 providers

claims = pd.read_csv(DATA / "claim_features.csv", usecols=["claim_id", "member_id", "provider_id", "cost_ratio"])
n_members_total = claims.member_id.nunique()

# 1) Bipartite graph: provider <-> member, edge weight = number of claims
B = nx.Graph()
for (p, m), g in claims.groupby(["provider_id", "member_id"]):
    B.add_edge(("P", p), ("M", m), claims=len(g))
print(f"Bipartite graph: {len({n for n in B if n[0]=='P'})} providers, "
      f"{len({n for n in B if n[0]=='M'})} members, {B.number_of_edges()} provider-member links")

# 2) Provider projection: link providers that share many more members than chance
members_of = claims.groupby("provider_id").member_id.apply(set)
G = nx.Graph()
G.add_nodes_from(members_of.index)
for a, b in itertools.combinations(members_of.index, 2):
    shared = members_of[a] & members_of[b]
    if len(shared) < MIN_SHARED:
        continue
    expected = len(members_of[a]) * len(members_of[b]) / n_members_total
    lift = len(shared) / expected
    if lift < MIN_LIFT:
        continue
    sub = claims[claims.member_id.isin(shared) & claims.provider_id.isin([a, b])]
    G.add_edge(a, b, shared=len(shared), lift=round(lift, 2), cost=round(float(sub.cost_ratio.mean()), 3))
print(f"Provider graph: {G.number_of_edges()} strong links between providers")

# 3) Communities (clusters) among linked providers, then judge each cluster
linked = G.subgraph([n for n in G if G.degree(n) > 0])
comms = [set(c) for c in nx.community.louvain_communities(linked, weight="shared", seed=42)]
rows = []
for i, c in enumerate(sorted(comms, key=len, reverse=True), start=1):
    if len(c) < 2:
        continue
    sub = G.subgraph(c)
    costs = [d["cost"] for _, _, d in sub.edges(data=True)]
    rows.append(dict(cluster=i, providers=sorted(c), size=len(c), links=sub.number_of_edges(),
                     density=round(nx.density(sub), 2), avg_shared_members=round(sum(d["shared"] for *_, d in sub.edges(data=True)) / max(sub.number_of_edges(), 1), 1),
                     avg_cost_ratio=round(sum(costs) / max(len(costs), 1), 2)))
cl = pd.DataFrame(rows)
cl["flagged"] = (cl["size"] >= MIN_CLUSTER) & (cl.avg_cost_ratio >= MIN_COST_RATIO) & (cl.density >= 0.5)
print("\n=== Provider clusters (providers sharing many members) ===")
print(cl.to_string(index=False))

# 4) Other graph measures: centrality and member concentration
deg = nx.degree_centrality(G)
btw = nx.betweenness_centrality(G, weight=None)
top_share = (claims.groupby(["provider_id", "member_id"]).size().rename("n").reset_index()
             .sort_values(["provider_id", "n"], ascending=[True, False]).groupby("provider_id").head(10)
             .groupby("provider_id").n.sum() / claims.groupby("provider_id").size())
signals = pd.DataFrame({"provider_id": list(G.nodes)})
cluster_of = {p: int(r.cluster) for _, r in cl.iterrows() for p in r.providers}
flagged_providers = {p for _, r in cl[cl.flagged].iterrows() for p in r.providers}
signals["cluster"] = signals.provider_id.map(cluster_of)
signals["in_flagged_cluster"] = signals.provider_id.isin(flagged_providers).astype(int)
signals["degree_centrality"] = signals.provider_id.map(deg).round(3)
signals["betweenness"] = signals.provider_id.map(btw).round(4)
signals["top10_member_claim_share"] = signals.provider_id.map(top_share).round(3)
signals.to_csv(OUT / "graph_provider_signals.csv", index=False)
# Saved for the dashboard (provider network view and collusion alerts)
pd.DataFrame([dict(provider_a=a, provider_b=b, **d) for a, b, d in G.edges(data=True)]).to_csv(OUT / "graph_edges.csv", index=False)
cl_out = cl.copy()
cl_out["providers"] = cl_out.providers.map(";".join)
cl_out.to_csv(OUT / "graph_clusters.csv", index=False)
print("\nProviders with highest degree centrality (linked to many providers):")
print(signals.sort_values("degree_centrality", ascending=False).head(5)[["provider_id", "degree_centrality", "in_flagged_cluster"]].to_string(index=False))

# 5) Grade against the answer key (labels are used ONLY here)
lab = pd.read_csv(DATA / "claim_labels.csv")
c2 = pd.read_csv(DATA / "claims.csv", usecols=["claim_id", "provider_id"]).merge(lab, on="claim_id")
ring = set(c2[c2.pattern_type == "collusion_ring"].provider_id)
hit = ring & flagged_providers
print(f"\n=== Check against answer key ===\nPlanted collusion providers: {sorted(ring)}")
print(f"Flagged by graph: {sorted(flagged_providers)}")
print(f"Planted providers found: {len(hit)} of {len(ring)} | flagged providers that are NOT planted: {len(flagged_providers - ring)}")

# 6) Optional picture for the portfolio
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(linked, weight="shared", seed=42, k=0.9)
    colors = ["#d62728" if n in flagged_providers else "#8c8c8c" for n in linked]
    plt.figure(figsize=(9, 7))
    nx.draw_networkx_edges(linked, pos, alpha=0.35, width=[linked[u][v]["shared"] / 10 for u, v in linked.edges])
    nx.draw_networkx_nodes(linked, pos, node_color=colors, node_size=420)
    nx.draw_networkx_labels(linked, pos, font_size=7, font_color="white")
    plt.title("Providers sharing many members (red = flagged cluster: shared members billed above normal cost)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "provider_graph.png", dpi=150)
    print("Saved outputs/provider_graph.png")
except ImportError:
    print("(matplotlib not installed, skipping the chart: pip install matplotlib)")
print("Saved outputs/graph_provider_signals.csv, graph_edges.csv, graph_clusters.csv")
