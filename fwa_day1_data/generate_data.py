"""Synthetic health-insurance claims generator for the FWA project.

Everything here is fake. A fixed seed makes every run identical.
Output (CSV, in ./data):
  members, providers, doctors, procedures, authorizations, claims
  claim_labels  -> ANSWER KEY ONLY (is_suspicious, pattern_type). Never used as a feature.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

N_MEMBERS, N_PROVIDERS, N_DOCTORS, N_PROCS = 2000, 100, 250, 40
N_NORMAL_CLAIMS = 19000
START, END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
CITIES = ["Riyadh", "Jeddah", "Dammam", "Mecca", "Medina", "Khobar", "Abha", "Tabuk"]
CITY_P = [0.32, 0.22, 0.12, 0.09, 0.08, 0.07, 0.05, 0.05]

# ---------------------------------------------------------------- procedures
cats = ["consultation", "lab", "imaging", "physio", "surgery", "dental", "pharmacy", "emergency"]
proc = pd.DataFrame({
    "procedure_code": [f"PR{i:03d}" for i in range(1, N_PROCS + 1)],
    "category": rng.choice(cats, N_PROCS),
})
base = {"consultation": 250, "lab": 120, "imaging": 900, "physio": 200,
        "surgery": 12000, "dental": 400, "pharmacy": 150, "emergency": 1800}
proc["typical_cost"] = [round(base[c] * rng.uniform(0.6, 1.6), 2) for c in proc.category]
proc["requires_auth"] = proc.typical_cost >= 3000
proc.loc[proc.category == "imaging", "requires_auth"] = proc.typical_cost >= 1000
proc["popularity"] = 1 / np.arange(1, N_PROCS + 1) ** 0.6   # a few procedures are common
proc["popularity"] = rng.permutation(proc["popularity"].values)

# ----------------------------------------------------------------- providers
ptypes = ["hospital", "clinic", "lab", "imaging_center", "pharmacy"]
providers = pd.DataFrame({
    "provider_id": [f"P{i:03d}" for i in range(1, N_PROVIDERS + 1)],
    "provider_type": rng.choice(ptypes, N_PROVIDERS, p=[0.2, 0.45, 0.12, 0.1, 0.13]),
    "city": rng.choice(CITIES, N_PROVIDERS, p=CITY_P),
})
providers["name"] = "Provider " + providers.provider_id

# ------------------------------------------------------------------- doctors
doc_prov = np.concatenate([providers.provider_id.values,
                           rng.choice(providers.provider_id, N_DOCTORS - N_PROVIDERS)])
doctors = pd.DataFrame({
    "doctor_id": [f"D{i:03d}" for i in range(1, N_DOCTORS + 1)],
    "provider_id": doc_prov,
    "specialty": rng.choice(["general", "cardiology", "orthopedics", "radiology",
                             "dentistry", "internal_medicine", "emergency"], N_DOCTORS),
})

# ------------------------------------------------------------------- members
age = np.clip(rng.gamma(4, 9, N_MEMBERS), 1, 90).astype(int)
chronic = rng.random(N_MEMBERS) < np.where(age > 55, 0.35, 0.07)   # legit heavy users
members = pd.DataFrame({
    "member_id": [f"M{i:05d}" for i in range(1, N_MEMBERS + 1)],
    "age": age,
    "gender": rng.choice(["F", "M"], N_MEMBERS),
    "city": rng.choice(CITIES, N_MEMBERS, p=CITY_P),
    "chronic_flag": chronic.astype(int),
})

# Each provider offers a subset of procedures; each member has "regular" providers in their city.
prov_procs = {p: rng.choice(proc.procedure_code, rng.integers(10, 21), replace=False)
              for p in providers.provider_id}
pop = proc.set_index("procedure_code").popularity
prov_by_city = providers.groupby("city").provider_id.apply(list).to_dict()
regular = {}
for m, c in zip(members.member_id, members.city):
    pool = prov_by_city.get(c) or providers.provider_id.tolist()
    regular[m] = list(rng.choice(pool, min(len(pool), rng.integers(1, 4)), replace=False))
docs_by_prov = doctors.groupby("provider_id").doctor_id.apply(list).to_dict()
cost = proc.set_index("procedure_code").typical_cost
needs_auth = proc.set_index("procedure_code").requires_auth

rows, auths = [], []


def rand_date():
    return START + pd.Timedelta(days=int(rng.integers(0, (END - START).days + 1)))


def add_claim(member, prov, code, date, amount, pattern="normal", auth=True, doctor=None):
    doctor = doctor or rng.choice(docs_by_prov[prov])
    auth_id = None
    if needs_auth[code] and auth:
        auth_id = f"A{len(auths) + 1:06d}"
        auths.append({"auth_id": auth_id, "member_id": member, "provider_id": prov,
                      "procedure_code": code,
                      "approved_date": date - pd.Timedelta(days=int(rng.integers(1, 15)))})
    rows.append({"member_id": member, "provider_id": prov, "doctor_id": doctor,
                 "procedure_code": code, "service_date": date,
                 "submitted_date": date + pd.Timedelta(days=int(rng.integers(0, 10))),
                 "billed_amount": round(float(amount), 2), "auth_id": auth_id,
                 "_pattern": pattern})


# ------------------------------------------------------------ normal claims
w = np.where(members.chronic_flag == 1, 3.0, 1.0) * np.where(members.age > 55, 1.5, 1.0)
w = w / w.sum()
for m in rng.choice(members.member_id, N_NORMAL_CLAIMS, p=w):
    prov = rng.choice(regular[m]) if rng.random() < 0.8 else rng.choice(providers.provider_id)
    codes = prov_procs[prov]
    pw = pop[codes].values
    code = rng.choice(codes, p=pw / pw.sum())
    add_claim(m, prov, code, rand_date(), cost[code] * rng.lognormal(0, 0.15),
              auth=rng.random() < 0.97)

normal_idx = list(range(len(rows)))
used_prov = set()


def pick_providers(k):
    pool = [p for p in providers.provider_id if p not in used_prov]
    chosen = list(rng.choice(pool, k, replace=False))
    used_prov.update(chosen)
    return chosen


# ---------------------------------------------- 1) duplicate claims (~150)
for i in rng.choice(normal_idx, 150, replace=False):
    r = rows[i]
    d = r["service_date"] + pd.Timedelta(days=int(rng.choice([0, 0, 0, 1])))
    amt = r["billed_amount"] * (1.0 if rng.random() < 0.7 else rng.uniform(0.98, 1.02))
    add_claim(r["member_id"], r["provider_id"], r["procedure_code"], d, amt,
              "duplicate_claim", auth=False, doctor=r["doctor_id"])
    rows[-1]["auth_id"] = r["auth_id"]

# ---------------------------------- 2) abnormal billing / upcoding (4 providers)
for p in pick_providers(4):
    for i in normal_idx:
        if rows[i]["provider_id"] == p and rows[i]["_pattern"] == "normal" and rng.random() < 0.35:
            rows[i]["billed_amount"] = round(rows[i]["billed_amount"] * rng.uniform(1.8, 2.6), 2)
            rows[i]["_pattern"] = "abnormal_billing"

# ------------------------------- 3) member overutilization (15 members x 12)
heavy = rng.choice(members.member_id[members.chronic_flag == 0], 15, replace=False)
for m in heavy:
    code = rng.choice(proc.procedure_code[proc.category.isin(["physio", "consultation", "lab"])])
    start = rand_date() - pd.Timedelta(days=60)
    for _ in range(12):
        prov = rng.choice(providers.provider_id)          # doctor-shopping across providers
        c = code if code in prov_procs[prov] else rng.choice(prov_procs[prov])
        add_claim(m, prov, c, start + pd.Timedelta(days=int(rng.integers(0, 60))),
                  cost[c] * rng.lognormal(0, 0.15), "overutilization")

# -------------------------- 4) unusual procedure frequency (3 providers x 60)
for p in pick_providers(3):
    code = rng.choice(prov_procs[p])
    spike_members = rng.choice(members.member_id, 40, replace=False)
    for _ in range(60):
        add_claim(rng.choice(spike_members), p, code,
                  rand_date(), cost[code] * rng.lognormal(0, 0.15), "procedure_frequency_spike")

# ------------- 5) collusion ring: 4 providers sharing 25 members (~150 claims)
ring_p = pick_providers(4)
ring_m = rng.choice(members.member_id, 25, replace=False)
for _ in range(150):
    p, m = rng.choice(ring_p), rng.choice(ring_m)
    code = rng.choice(prov_procs[p])
    add_claim(m, p, code, rand_date(), cost[code] * rng.uniform(1.25, 1.6),
              "collusion_ring", auth=rng.random() < 0.5)

# ------------------------------------------------------------------ finalize
claims = pd.DataFrame(rows).sort_values(["service_date", "member_id"]).reset_index(drop=True)
claims.insert(0, "claim_id", [f"C{i:06d}" for i in range(1, len(claims) + 1)])  # IDs assigned AFTER shuffle/sort: no leakage
labels = claims[["claim_id", "_pattern"]].rename(columns={"_pattern": "pattern_type"})
labels["is_suspicious"] = (labels.pattern_type != "normal").astype(int)
claims = claims.drop(columns="_pattern")
authorizations = pd.DataFrame(auths)

for name, df in {"members": members, "providers": providers, "doctors": doctors,
                 "procedures": proc.drop(columns="popularity"),
                 "authorizations": authorizations, "claims": claims,
                 "claim_labels": labels[["claim_id", "is_suspicious", "pattern_type"]]}.items():
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name:15s} {len(df):>7,} rows")
print(f"\nsuspicious rate: {labels.is_suspicious.mean():.2%}")
print(labels.pattern_type.value_counts().to_string())
