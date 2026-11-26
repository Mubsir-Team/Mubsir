"""
Counterfactual data for the controlled rotation study (PREREGISTRATION.md §4). Non-destructive: reads the pre-injection
history (behavior_features_r62_v1) and baselines, writes only study_controlled_rotation/data/ and results/.

Cohorts
  H146  primary: dev users with NB12-valid baselines on all 3 channels and integer P50_c and max(1, P95_c - P50_c)
        rng = default_rng(42), hosts in sorted order: start s, then channel permutation pi        (§4.2-4.3)
  R253  secondary S7: the other 107 of the 253 (approximate matching), rng = default_rng(43), same draw order
  S8    secondary: H146 hosts whose attack days are restricted to days with all 3 channels naturally active;
        rng = default_rng(44): start position among such days having >= 30 prior workdays, then pi
Variants per host and L in {6, 12}: ROT, BLK, FIX-E, FIX-U, FIX-H, SIM (H146); ROT, BLK only (R253, S8).
Injection on channel c, attack day t:  x' = max(x, P50_c) + delta * den_c, delta = 1, den_c = max(1, P95_c - P50_c)
  (R253: x' = max(x, ceil(P50_c)) + round(den_c); SIM: x' = max(x, P50_c) + max(1, round(den_c / 3)))
The added count goes to the channel field and its NB10 event-total fields (features.INJECTION_FIELDS).
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent  # studies/<name>/ -> Model/ (moved 2026-11-26)
ROOT = MODEL_DIR.parent
sys.path.insert(0, str(HERE))
from features import compute_features, INJECTION_FIELDS, RAW_DAILY  # noqa: E402

assert hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest() == \
    (HERE / "PREREGISTRATION.sha256").read_text().strip(), "pre-registration changed"
assert json.loads((HERE / "results" / "gate0.json").read_text())["passed"], "Gate 0 has not passed"
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
RES = HERE / "results"
t0 = time.time()
FEATS = json.loads((MODEL_DIR / "model" / "manifest.json").read_text(encoding="utf-8"))["features"]
CHN = ["EMAIL", "USB", "HTTP"]
CNT = {"EMAIL": "external_email_count", "USB": "usb_copy_count", "HTTP": "www_upload_count"}
BASE = {"EMAIL": "email_count", "USB": "usb_copy", "HTTP": "http_upload"}
LS = (6, 12)
DELTA = 1.0

split = pd.read_csv(MODEL_DIR / "config" / "final_split.csv")
um = pd.read_csv(MODEL_DIR / "config" / "user_meta.csv").set_index("user")
dev = set(split.loc[split.part == "dev", "user"])
v1 = pd.read_parquet(ROOT / "Data_engineering" / "data" / "features" / "behavior_features_r62_v1.parquet")
v1["user"] = v1["user"].astype(str)
v1["day"] = pd.to_datetime(v1["day"]).dt.normalize()
v1 = v1[v1["user"].isin(dev)].sort_values(["user", "day"]).reset_index(drop=True)
bm = pd.read_parquet(ROOT / "Data_engineering" / "data" / "features" / "user_baseline_r62_v1.parquet",
                     columns=["user", "email_count_mean_active", "usb_copy_mean_active", "http_upload_mean_active"])
bm["user"] = bm["user"].astype(str)
v1 = v1.merge(bm, on="user", how="left", validate="many_to_one")
hist = {u: g.reset_index(drop=True) for u, g in v1.groupby("user", sort=True)}


def bl(u, ch, q):
    return float(hist[u][f"{BASE[ch]}_{q}"].iloc[0])


def valid(u):
    return all(np.isfinite(bl(u, c, "p50")) and np.isfinite(bl(u, c, "p95")) and bl(u, c, "p50") > 0
               and bl(u, c, "p95") >= bl(u, c, "p50") for c in CHN)


def den(u, c):
    return max(1.0, bl(u, c, "p95") - bl(u, c, "p50"))


def is_int(v):
    return abs(v - round(v)) < 1e-9


hosts253 = sorted(u for u in hist if valid(u))
H146 = [u for u in hosts253 if all(is_int(bl(u, c, "p50")) and is_int(den(u, c)) for c in CHN)]
R107 = [u for u in hosts253 if u not in set(H146)]
assert len(hosts253) == 253 and len(H146) == 146, (len(hosts253), len(H146))


def schedule(variant, L, pi):
    """List over attack days j of {channel: share of delta}."""
    if variant == "ROT":
        return [{CHN[pi[j % 3]]: 1.0} for j in range(L)]
    if variant == "BLK":
        return [{CHN[pi[(3 * j) // L]]: 1.0} for j in range(L)]
    if variant.startswith("FIX-"):
        return [{variant[4:]: 1.0} for _ in range(L)]
    if variant == "SIM":
        return [{c: 1 / 3 for c in CHN} for _ in range(L)]
    raise ValueError(variant)


def make_case(u, cohort, variant, L, days_idx, pi):
    rows = hist[u].copy()
    added_log = []
    for j, (t, assign) in enumerate(zip(days_idx, schedule(variant, L, pi))):
        for ch, share in assign.items():
            x = float(rows.at[t, CNT[ch]])
            p50, d = bl(u, ch, "p50"), den(u, ch)
            if cohort == "R253":
                new = max(x, np.ceil(p50)) + round(d)
            elif variant == "SIM":
                new = max(x, p50) + max(1, round(d / 3))
            else:
                new = max(x, p50) + DELTA * d
            add = new - x
            assert add >= 0 and is_int(add)
            for f in INJECTION_FIELDS[ch]:
                rows.at[t, f] = rows.at[t, f] + add
            added_log.append(dict(day_index=t, channel=ch, added=add, x_before=x))
    feats = compute_features(rows, FEATS)
    case = f"{u}|{cohort}|{variant}|L{L}"
    y = np.zeros(len(rows), dtype=int)
    y[list(days_idx)] = 1
    y10 = pd.Series(y).rolling(10, min_periods=1).max().astype(int).to_numpy()
    span = np.zeros(len(rows), dtype=bool)
    span[days_idx[0]: min(len(rows), days_idx[-1] + 10)] = True
    out = feats.assign(case=case, host=u, cohort=cohort, variant=variant, L=L, y=y, y_10=y10, in_span=span,
                       first_attack_index=days_idx[0], row_index=np.arange(len(rows)))
    for f in ("external_email_count", "usb_copy_count", "www_upload_count"):
        out[f"raw_{f}"] = rows[f].to_numpy()
    return out, added_log


base_feats = {}


def injected_dE(u, case_df, days_idx):
    """Per-channel injected deviation on the attack days = E_today(case) - E_today(original)."""
    if u not in base_feats:
        base_feats[u] = compute_features(hist[u], FEATS)
    b = base_feats[u]
    out = {}
    for c, col in zip(CHN, ("E_email_today", "E_usb_today", "E_http_today")):
        diff = case_df[col].to_numpy() - b[col].to_numpy()
        out[c] = float(diff[list(days_idx)].sum())
        out[f"{c}_outside_attack_days"] = float(np.abs(np.delete(diff, list(days_idx))).sum())
    return out


frames, verify, excluded = [], [], []


def run_cohort(cohort, host_list, seed, variants, strict=False):
    rng = np.random.default_rng(seed)
    for u in host_list:
        n = len(hist[u])
        if strict:
            r = hist[u]
            all3 = np.flatnonzero((r["external_email_count"] > 0) & (r["usb_copy_count"] > 0) & (r["www_upload_count"] > 0))
            all3 = all3[all3 >= 30]
            if len(all3) < min(LS):
                excluded.append(dict(host=u, cohort=cohort, reason=f"only {len(all3)} all-3-active days after day 30"))
                continue
            Lmax = max(L for L in LS if L <= len(all3))      # PREREGISTRATION S8: at least L such days per length
            p = int(rng.integers(0, len(all3) - Lmax + 1))
            pi = rng.permutation(3)
            days_for = {L: list(all3[p:p + L]) for L in LS if L <= len(all3)}
            if len(days_for) < len(LS):
                excluded.append(dict(host=u, cohort=cohort, reason=f"L=12 not possible: {len(all3)} all-3-active days"))
        else:
            if n < 30 + 12 + 10:
                excluded.append(dict(host=u, cohort=cohort, reason=f"only {n} eligible workdays"))
                continue
            s = int(rng.integers(30, n - 12 - 10 + 1))
            pi = rng.permutation(3)
            days_for = {L: list(range(s, s + L)) for L in LS}
        for L in [L for L in LS if L in days_for]:
            per = {}
            for v in variants:
                df, log = make_case(u, cohort, v, L, days_for[L], pi)
                frames.append(df)
                per[v] = injected_dE(u, df, days_for[L])
                raw = pd.DataFrame(log).groupby("channel")["added"].sum().reindex(CHN).fillna(0)
                verify.append(dict(host=u, cohort=cohort, variant=v, L=L, start=days_for[L][0],
                                   pi="".join(CHN[i][0] for i in pi), is_attacker_host=int(um.loc[u, "is_attacker"]),
                                   **{f"dE_{c}": per[v][c] for c in CHN},
                                   dE_total=sum(per[v][c] for c in CHN),
                                   dE_outside=sum(per[v][f"{c}_outside_attack_days"] for c in CHN),
                                   **{f"raw_added_{c}": float(raw[c]) for c in CHN}))


run_cohort("H146", H146, 42, ["ROT", "BLK", "FIX-EMAIL", "FIX-USB", "FIX-HTTP", "SIM"])
run_cohort("R253", R107, 43, ["ROT", "BLK"])
run_cohort("S8", H146, 44, ["ROT", "BLK"], strict=True)
V = pd.DataFrame(verify)

# ---------------------------------------------------------------- verification (§4.4)
checks = []


def check(name, ok, detail=""):
    checks.append(dict(check=name, passed=bool(ok), detail=str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


h = V[V.cohort.isin(["H146", "S8"])]
piv = h[h.variant.isin(["ROT", "BLK"])].pivot_table(index=["cohort", "host", "L"], columns="variant",
                                                    values=[f"dE_{c}" for c in CHN])
gap = max(float((piv[(f"dE_{c}", "ROT")] - piv[(f"dE_{c}", "BLK")]).abs().max()) for c in CHN)
bad_hosts = sorted({k[1] for k in piv.index if max(abs(piv.loc[k, (f"dE_{c}", "ROT")] - piv.loc[k, (f"dE_{c}", "BLK")]) for c in CHN) > 1e-9})
check("ROT vs BLK per-channel injected deviation identical (H146, S8), max |diff|", gap <= 1e-9, f"{gap:.2e}; hosts failing: {len(bad_hosts)}")
exact = h[h.variant.isin(["ROT", "BLK", "FIX-EMAIL", "FIX-USB", "FIX-HTTP"])]
tot_gap = float((exact["dE_total"] - exact["L"] * DELTA).abs().max())
check("total injected deviation = L x delta for ROT/BLK/FIX (H146, S8)", tot_gap <= 1e-9, f"max |diff| {tot_gap:.2e}")
check("each channel gets L/3 of the deviation in ROT and BLK (H146, S8)",
      float((h[h.variant.isin(["ROT", "BLK"])][[f"dE_{c}" for c in CHN]].sub(h[h.variant.isin(["ROT", "BLK"])]["L"] / 3, axis=0)).abs().max().max()) <= 1e-9)
check("no deviation change outside the attack days' own E_today", float(V["dE_outside"].abs().max()) <= 1e-9, f"{V['dE_outside'].abs().max():.2e}")
surv = len(set(H146) - set(bad_hosts) - {e['host'] for e in excluded if e['cohort'] == 'H146'})
check("H146 hosts surviving verification >= 120 (else primary inconclusive)", surv >= 120, surv)

data = pd.concat(frames, ignore_index=True)
data.to_parquet(OUT / "counterfactual_cases.parquet", index=False)
V.to_csv(RES / "data_verification.csv", index=False)
pd.DataFrame(excluded).to_csv(RES / "data_exclusions.csv", index=False)
summ = V.groupby(["cohort", "variant", "L"]).agg(cases=("host", "size"), attacker_hosts=("is_attacker_host", "sum"),
                                                 dE_EMAIL=("dE_EMAIL", "mean"), dE_USB=("dE_USB", "mean"),
                                                 dE_HTTP=("dE_HTTP", "mean"), dE_total_mean=("dE_total", "mean"),
                                                 dE_total_min=("dE_total", "min"), dE_total_max=("dE_total", "max"),
                                                 raw_EMAIL=("raw_added_EMAIL", "mean"), raw_USB=("raw_added_USB", "mean"),
                                                 raw_HTTP=("raw_added_HTTP", "mean")).reset_index()
summ.to_csv(RES / "data_summary.csv", index=False)
json.dump(dict(passed=all(c["passed"] for c in checks), checks=checks, hosts_253=len(hosts253), H146=len(H146),
               H146_surviving=surv, R107=len(R107), excluded=len(excluded), cases=int(data["case"].nunique()),
               rows=int(len(data)), seconds=round(time.time() - t0)),
          open(RES / "data_generation.json", "w"), indent=1)
print(summ.round(3).to_string(index=False))
print(f"cases {data['case'].nunique()}, rows {len(data):,}, excluded {len(excluded)}, {time.time() - t0:.0f}s")
