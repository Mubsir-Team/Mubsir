"""
Gates G1-G4 of the switching-baseline study (PREREGISTRATION.md §7) + feature computation for P1 (dev rows) and P2
(existing H146 cases of eligible hosts). No model is trained here. Writes data/features_p1.parquet,
data/features_p2.parquet and results/gates.json.
G1 or G3 or G4 failure -> stop (outcome 4). G2 failure on an H feature -> outcome 4, analysis still run descriptively.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent  # studies/<name>/ -> Model/ (moved 2026-11-26)
ROOT = MODEL_DIR.parent
sys.path.insert(0, str(MODEL_DIR / "pipeline"))
sys.path.insert(0, str(HERE))
from common import LABELS, load_dataset  # noqa: E402
from hab_features import switching_features, habit_rate, R_NAMES, H_NAMES, M0  # noqa: E402

PRE = (HERE / "PREREGISTRATION.sha256").read_text().strip()
assert hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest() == PRE, "pre-registration changed"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
assert sha(HERE / "results" / "habit_2010_dev_users.csv").startswith("adc39b4b1d94b5ab"), "habit file changed"
assert sha(HERE / "rot_features.py").startswith("e15d7740c96bf764"), "rot_features changed"
(HERE / "data").mkdir(exist_ok=True)
CHECKS = []
Ecols = ["E_email_today", "E_usb_today", "E_http_today"]


def check(gate, name, ok, detail=""):
    CHECKS.append(dict(gate=gate, check=name, passed=bool(ok), detail=str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {gate} {name}  {detail}", flush=True)


habit = pd.read_csv(HERE / "results" / "habit_2010_dev_users.csv").set_index("user")
P1 = sorted(habit.index[habit.eligible].astype(str))
r0 = habit["rate"]
check("setup", "P1 = 714 eligible users (347 attackers)", len(P1) == 714 and int(habit.loc[P1, "is_attacker"].sum()) == 347)

# ------------------------------------------------------------------ G1 unit check
E_U_H = np.eye(3)
rot = np.array([E_U_H[i % 3] for i in range(6)])
blk = np.array([E_U_H[i // 2] for i in range(6)])
f_rot, f_blk, f_q = (switching_features(x, 0.4) for x in (rot, blk, np.zeros((6, 3))))
check("G1", "ROT toy: pairs 5, changes 5, rate_dev_hab 0.6, excess_hab 3",
      f_rot["_pairs_10"][-1] == 5 and f_rot["_changes_10"][-1] == 5 and np.isclose(f_rot["rate_dev_hab_10"][-1], 0.6)
      and np.isclose(f_rot["excess_hab_10"][-1], 3.0))
check("G1", "BLK toy: pairs 5, changes 2, rate_dev_hab 0, excess_hab 0",
      f_blk["_pairs_10"][-1] == 5 and f_blk["_changes_10"][-1] == 2 and np.isclose(f_blk["rate_dev_hab_10"][-1], 0.0)
      and np.isclose(f_blk["excess_hab_10"][-1], 0.0))
check("G1", "quiet window: all features 0", all(np.allclose(f_q[k], 0) for k in R_NAMES + H_NAMES))
check("G1", "causality: features at row t unchanged when later rows change",
      all(np.allclose(switching_features(np.r_[rot, blk], 0.4)[k][:6], f_rot[k]) for k in R_NAMES + H_NAMES))

# r0 reproduction from 2010
CNT = ["external_email_count", "usb_copy_count", "www_upload_count"]
BASE = ["email_count", "usb_copy", "http_upload"]
bl = pd.read_parquet(ROOT / "Data_engineering" / "data" / "features" / "user_baseline_r62_v1.parquet")
bl = bl[bl.user.astype(str).isin(P1)].set_index("user")
ud = pd.read_parquet(ROOT / "Data_engineering" / "data" / "processed" / "user_day_r62_v1.parquet",
                     columns=["user", "day", "is_weekend", "logon_count"] + CNT)
ud = ud[ud.user.astype(str).isin(P1)]
ud["day"] = pd.to_datetime(ud["day"])
b10 = ud[(ud.day >= "2010-01-02") & (ud.day <= "2010-12-31") & (~ud.is_weekend.astype(bool)) & (ud.logon_count > 0)]
worst = 0.0
for u, g in b10.sort_values(["user", "day"]).groupby("user"):
    p50 = bl.loc[u, [f"{b}_p50" for b in BASE]].to_numpy(dtype=float)
    p95 = bl.loc[u, [f"{b}_p95" for b in BASE]].to_numpy(dtype=float)
    x = g[CNT].to_numpy(dtype=float)
    E = np.zeros_like(x)
    for c in range(3):
        if np.isfinite(p50[c]) and np.isfinite(p95[c]):
            E[:, c] = np.maximum(0.0, x[:, c] - p50[c]) / max(1.0, p95[c] - p50[c])
    worst = max(worst, abs(habit_rate(E)[0] - r0[u]))
check("G1", "r0 recomputed = habit_2010_dev_users.csv for 714 users (|diff| <= 1e-12)", worst <= 1e-12, f"max {worst:.1e}")
if not all(c["passed"] for c in CHECKS):
    raise SystemExit("G1 failed -> study stops (outcome 4)")

# ------------------------------------------------------------------ features: P1 dev rows
df = load_dataset()
lab = pd.read_parquet(LABELS)
keep = df["user"].astype(str).isin(set(P1)).to_numpy()
d = pd.DataFrame({"user": df.loc[keep, "user"].astype(str).to_numpy(), "day": pd.to_datetime(df.loc[keep, "day"]).to_numpy(),
                  "y": lab.loc[keep, "y"].to_numpy(), "y_10": lab.loc[keep, "y_10"].to_numpy()})
d[Ecols] = df.loc[keep, Ecols].to_numpy(dtype="float64")
d["row_in_dataset"] = np.flatnonzero(keep)
check("setup", "P1 rows sorted by user then day",
      bool((d.groupby("user")["day"].apply(lambda s: s.is_monotonic_increasing)).all()))
parts = []
for u, g in d.groupby("user", sort=False):
    f = switching_features(g[Ecols].to_numpy(), r0[u])
    parts.append(pd.DataFrame({k: f[k] for k in R_NAMES + H_NAMES}, index=g.index))
d = d.join(pd.concat(parts))
d.to_parquet(HERE / "data" / "features_p1.parquet", index=False)

# ------------------------------------------------------------------ features: P2 existing H146 cases (read-only source)
cases = pd.read_parquet(MODEL_DIR / "studies" / "controlled_rotation" / "data" / "counterfactual_cases.parquet")
cases = cases[(cases.cohort == "H146") & cases.host.isin(set(P1))].sort_values(["case", "row_index"]).reset_index(drop=True)
parts = []
for cs, g in cases.groupby("case", sort=False):
    f = switching_features(g[Ecols].to_numpy(), r0[g["host"].iloc[0]])
    parts.append(pd.DataFrame({k: f[k] for k in R_NAMES + H_NAMES}, index=g.index))
cases = cases.join(pd.concat(parts))
check("setup", "P2 = 123 hosts, 246 ROT and 246 BLK cases",
      cases.host.nunique() == 123 and cases[cases.variant == "ROT"].case.nunique() == 246
      and cases[cases.variant == "BLK"].case.nunique() == 246)
cases.to_parquet(HERE / "data" / "features_p2.parquet", index=False)

# ------------------------------------------------------------------ G2 shortcut gate
d["pos"] = d.groupby("user").cumcount()
first = d[d.y == 1].groupby("user")["pos"].min()
pre = d[d["pos"] < d["user"].map(first).fillna(np.inf)]
g2 = {}
for k in R_NAMES + H_NAMES:
    m = pre.groupby("user")[k].mean()
    a = habit.loc[m.index, "is_attacker"].to_numpy()
    auc = roc_auc_score(a, m.to_numpy())
    g2[k] = max(auc, 1 - auc)
n_g2 = int(pre.user.nunique())
for k, v in g2.items():
    check("G2", f"shortcut AUC {k} < 0.60", v < 0.60, f"{v:.3f}")

# ------------------------------------------------------------------ G3 fold integrity
folds = pd.read_csv(MODEL_DIR / "config" / "cv_folds.csv")
folds = folds[folds.user.astype(str).isin(set(P1))]
um = pd.read_csv(MODEL_DIR / "config" / "user_meta.csv").set_index("user")
g3 = []
for (r, k), f in folds.groupby(["repeat", "fold"]):
    v = f.loc[f.role == "valid", "user"]
    att = int(um.loc[v, "is_attacker"].sum())
    ben = len(v) - att
    grp = {ro: set(um.loc[f.loc[f.role == ro, "user"], "group"]) for ro in ("fit", "es", "valid")}
    overlap = (grp["fit"] & grp["es"]) | (grp["fit"] & grp["valid"]) | (grp["es"] & grp["valid"])
    g3.append(dict(repeat=r, fold=k, valid_benign=ben, valid_attackers=att, fit_users=int((f.role == "fit").sum()),
                   es_users=int((f.role == "es").sum()), group_overlap=len(overlap)))
g3 = pd.DataFrame(g3)
g3.to_csv(HERE / "results" / "gate3_folds.csv", index=False)
check("G3", "every restricted validation fold has >= 20 benign and >= 20 attackers",
      (g3.valid_benign >= 20).all() and (g3.valid_attackers >= 20).all(),
      f"min benign {g3.valid_benign.min()}, min attackers {g3.valid_attackers.min()}")
check("G3", "no fit/es/valid group overlap", (g3.group_overlap == 0).all())
check("G3", "every P1 user in exactly one validation fold per repeat",
      (folds[folds.role == "valid"].groupby("repeat").user.nunique() == 714).all())

# ------------------------------------------------------------------ G4 pipeline unchanged
before = json.loads((HERE / "hashes_before.json").read_text(encoding="utf-8"))
for p in ("pipeline/common.py", "pipeline/models.py", "pipeline/metrics.py", "pipeline/05_tune.py", "config/cv_folds.csv",
          "config/selection_rules.json", "data/model_dataset_995.parquet", "work/labels.parquet"):
    check("G4", f"{p} identical to hashes_before.json", sha(MODEL_DIR / p) == before[p])

hard = [c for c in CHECKS if c["gate"] in ("setup", "G1", "G3", "G4")]
h_fail = [k for k in H_NAMES if g2[k] >= 0.60]
r_fail = [k for k in R_NAMES if g2[k] >= 0.60]
out = dict(preregistration_sha256=PRE, hard_gates_passed=all(c["passed"] for c in hard),
           g2_auc=g2, g2_users=n_g2, g2_h_failed=h_fail, g2_r_failed=r_fail,
           consequence=("stop (outcome 4)" if not all(c["passed"] for c in hard) else
                        "outcome 4 forced by G2 (analysis descriptive)" if h_fail else
                        "H2 inconclusive (R feature failed G2)" if r_fail else "proceed"),
           p1_rows=int(len(d)), p2_cases=int(cases.case.nunique()), m0=M0, checks=CHECKS)
(HERE / "results" / "gates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("GATES:", out["consequence"], f"| G2 users {n_g2}")
