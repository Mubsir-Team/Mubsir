"""
Gate 0 of the controlled rotation study (PREREGISTRATION.md §3). Any failed check stops the study.

0.1  features.py on the INJECTED counts reproduces all 100 model features of the 995 eligible users
     (max |diff| <= 1e-9, identical NaN pattern, identical keys).
0.2  features.py on the PRE-INJECTION counts reproduces the 100 features of every benign eligible user.
0.3  injected counts = pre-injection counts + count_added on exactly the NB10 fields of every ground-truth channel-day,
     and nothing else changed.
Read-only on all inputs; writes results/gate0.json only.
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
RES = HERE / "results"
RES.mkdir(exist_ok=True)
TOL = 1e-9
t0 = time.time()
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(dict(check=name, passed=bool(ok), detail=str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


man = json.loads((MODEL_DIR / "model" / "manifest.json").read_text(encoding="utf-8"))
FEATS = man["features"]
eligible = set(pd.read_csv(MODEL_DIR / "config" / "eligible_users_995.csv")["user"].astype(str))
um = pd.read_csv(MODEL_DIR / "config" / "user_meta.csv").set_index("user")
benign = {u for u in eligible if um.loc[u, "is_attacker"] == 0}

v1 = pd.read_parquet(ROOT / "Data_engineering" / "data" / "features" / "behavior_features_r62_v1.parquet")
v1["user"] = v1["user"].astype(str)
v1["day"] = pd.to_datetime(v1["day"]).dt.normalize()
v1 = v1[v1["user"].isin(eligible)].sort_values(["user", "day"]).reset_index(drop=True)
bm = pd.read_parquet(ROOT / "Data_engineering" / "data" / "features" / "user_baseline_r62_v1.parquet",
                     columns=["user", "email_count_mean_active", "usb_copy_mean_active", "http_upload_mean_active"])
bm["user"] = bm["user"].astype(str)
base = v1.merge(bm, on="user", how="left", validate="many_to_one")
check("pre-injection rows for the 995 users", base["user"].nunique() == 995 and len(base) == 103395, len(base))

inj = pd.read_parquet(ROOT / "New_Attack_Simulation" / "reports" / "10_Attack_Injection_Pipeline" / "user_day_injected_counts_r62.parquet")
inj["user"] = inj["user"].astype(str)
inj["day"] = pd.to_datetime(inj["day"]).dt.normalize()
inj = base[["user", "day"]].merge(inj, on=["user", "day"], how="left", validate="one_to_one")
check("injected counts available for every eligible row", inj[RAW_DAILY].notna().all().all())

ref = pd.read_parquet(MODEL_DIR / "data" / "model_dataset_995.parquet")
ref["user"] = ref["user"].astype(str)
ref["day"] = pd.to_datetime(ref["day"]).dt.normalize()
ref = ref.sort_values(["user", "day"]).reset_index(drop=True)
check("reference keys identical", (ref["user"].to_numpy() == base["user"].to_numpy()).all()
      and (ref["day"].to_numpy() == base["day"].to_numpy()).all())


def compare(calc, refdf, mask, label):
    worst, worst_f, nan_bad = 0.0, None, []
    per = []
    for f in FEATS:
        a = calc.loc[mask, f].to_numpy(dtype="float64")
        b = refdf.loc[mask, f].to_numpy(dtype="float64")
        na, nb = np.isnan(a), np.isnan(b)
        if not np.array_equal(na, nb):
            nan_bad.append(f)
        d = np.abs(a[~na & ~nb] - b[~na & ~nb])
        m = float(d.max()) if len(d) else 0.0
        per.append((f, m, int((d > TOL).sum())))
        if m > worst:
            worst, worst_f = m, f
    pd.DataFrame(per, columns=["feature", "max_abs_diff", "cells_above_tol"]).to_csv(RES / f"gate0_{label}_per_feature.csv", index=False)
    return worst, worst_f, nan_bad


# ---------------------------------------------------------------- 0.1 injected
rows_inj = base.copy()
for f in RAW_DAILY:
    rows_inj[f] = inj[f].to_numpy()
calc_inj = compute_features(rows_inj, FEATS)
worst, wf, nanbad = compare(calc_inj, ref, np.ones(len(ref), bool), "injected")
check("0.1 injected counts -> 100 features, all 995 users: max |diff| <= 1e-9", worst <= TOL, f"max {worst:.2e} ({wf})")
check("0.1 identical NaN pattern", not nanbad, nanbad[:5])

# ---------------------------------------------------------------- 0.2 pre-injection, benign users
calc_pre = compute_features(base, FEATS)
bmask = ref["user"].isin(benign).to_numpy()
worst2, wf2, nanbad2 = compare(calc_pre, ref, bmask, "preinjection_benign")
check("0.2 pre-injection counts -> 100 features, 595 benign users: max |diff| <= 1e-9", worst2 <= TOL,
      f"max {worst2:.2e} ({wf2})")
check("0.2 identical NaN pattern (benign)", not nanbad2, nanbad2[:5])

# ---------------------------------------------------------------- 0.3 injection mapping
gt = pd.read_parquet(ROOT / "New_Attack_Simulation" / "reports" / "11_Ground_Truth_and_Quality_Validation" / "ground_truth_channel_days_r62.parquet",
                     columns=["user", "day", "channel", "count_before", "count_added", "count_after"])
gt["user"] = gt["user"].astype(str)
gt["day"] = pd.to_datetime(gt["day"]).dt.normalize()
diff = inj[RAW_DAILY].to_numpy(dtype="float64") - base[RAW_DAILY].to_numpy(dtype="float64")
D = pd.DataFrame(diff, columns=RAW_DAILY)
D["user"], D["day"] = base["user"], base["day"]
expected = pd.DataFrame(0.0, index=D.index, columns=RAW_DAILY)
key = pd.Series(D.index, index=pd.MultiIndex.from_frame(D[["user", "day"]]))
missing = 0
for r in gt.itertuples(index=False):
    i = key.get((r.user, r.day))
    if i is None:
        missing += 1
        continue
    for f in INJECTION_FIELDS[r.channel]:
        expected.loc[i, f] += r.count_added
check("0.3 every ground-truth channel-day is an eligible row", missing == 0, missing)
mism = (np.abs(D[RAW_DAILY].to_numpy() - expected.to_numpy()) > 0).sum(axis=0)
check("0.3 injected - pre-injection = count_added on the NB10 fields, 0 elsewhere", mism.sum() == 0,
      dict(zip(RAW_DAILY, mism.tolist())) if mism.sum() else "all 18 raw fields match")
chan_field = {"EMAIL": "external_email_count", "USB": "usb_copy_count", "HTTP": "www_upload_count"}
cb = np.array([base.loc[key[(r.user, r.day)], chan_field[r.channel]] for r in gt.itertuples(index=False)])
check("0.3 ground-truth count_before equals the pre-injection channel count", np.array_equal(cb, gt["count_before"].to_numpy()))

passed = all(c["passed"] for c in CHECKS)
(RES / "gate0.json").write_text(json.dumps(dict(passed=passed, checks=CHECKS, tolerance=TOL,
                                                seconds=round(time.time() - t0)), indent=1), encoding="utf-8")
print("GATE 0:", "PASS" if passed else "FAIL -> study stops (PREREGISTRATION §3)", f"({time.time() - t0:.0f}s)")
