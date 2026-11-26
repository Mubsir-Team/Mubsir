"""
Shortcut gate (§5.2) + training of the registered arms (§5, §6) of the controlled rotation study.

Regime 1: arms B, C, C+, A-cv, D on the original dev simulation, Stage A folds/seeds/early stopping. Each fold model
scores its validation users' original rows (benign reference for thresholds) and the counterfactual cases of the hosts
in that validation fold (hosts never seen in training).
Regime 2 (S4): repeat 0; arms C+ and D trained on original fit rows + H146 cases of hosts outside the validation fold;
early stopping on original es users only; scores validation users and validation-fold hosts' H146 cases.
Outputs: scores/*.parquet, results/shortcut_gate.json, results/training_log.csv. No estimand is computed here.
"""

import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "pipeline"))
sys.path.insert(0, str(HERE))
import models  # noqa: E402
from common import SEED, WORK, LABELS, load_dataset  # noqa: E402
from rot_features import rotation_features, NAMES as ROT, NORM_NAMES as NORM  # noqa: E402

assert hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest() == \
    (HERE / "PREREGISTRATION.sha256").read_text().strip(), "pre-registration changed"
assert json.loads((HERE / "results" / "data_generation.json").read_text())["passed"]
T = importlib.import_module("05_tune")
OUTS = HERE / "scores"
OUTS.mkdir(exist_ok=True)
RES = HERE / "results"
t0 = time.time()

D = T.load()                                   # dev rows (float32, manifest order), y_10, folds, rules-hash check
FEATS = list(D["X"].columns)
users = D["users"]
folds = D["folds"]
R = 3

# ------------------------------------------------------------------ feature groups (§5)
fg = pd.read_csv(HERE.parent / "rotation" / "results" / "feature_groups.csv")
B = fg.loc[fg.group.isin(["email", "usb_file", "http", "context"]), "feature"].tolist()
pre = lambda ps: [f for f in FEATS if f.startswith(ps)]
AGG = pre(("cumulative_deviation", "combined_channel_intensity", "consecutive_elevated_days", "nr_stealth_acc"))
DIST = pre(("nr_m_", "nr_entropy", "nr_excl_days", "nr_overlap_days", "cross_channel_deviation_days",
            "channel_distribution_shift", "active_channel_count", "multi_channel_days"))
TRANS = pre(("nr_alt", "nr_gap_alt_2", "nr_handoff", "dominant_channel_switches"))
assert (len(B), len(AGG), len(DIST), len(TRANS)) == (50, 15, 22, 13), (len(B), len(AGG), len(DIST), len(TRANS))
assert sorted(B + AGG + DIST + TRANS) == sorted(FEATS)


def ordered(cols):
    return [f for f in FEATS if f in set(cols)]


# ------------------------------------------------------------------ new features: original dev rows and cases
df = load_dataset()
dev_mask = df["user"].astype(str).isin(set(users)).to_numpy()
dfd = df.loc[dev_mask].reset_index(drop=True)
Ecols = ["E_email_today", "E_usb_today", "E_http_today"]


def new_feats(frame, key):
    parts = []
    for _, g in frame.groupby(key, sort=False):
        f = rotation_features(g[Ecols].to_numpy())
        parts.append(pd.DataFrame({k: f[k] for k in ROT + NORM + ["_base_rate"]}, index=g.index))
    return pd.concat(parts).loc[frame.index]


NF = new_feats(dfd, "user")
cases = pd.read_parquet(HERE / "data" / "counterfactual_cases.parquet")
cases = cases.sort_values(["case", "row_index"]).reset_index(drop=True)
NFc = new_feats(cases, "case")

# ------------------------------------------------------------------ shortcut gate (§5.2)
lab = pd.read_parquet(LABELS).loc[dev_mask].reset_index(drop=True)
g = pd.DataFrame({"user": dfd["user"].astype(str), "y": lab["y"].to_numpy(), "base": NF["_base_rate"].to_numpy()})
g["pos"] = g.groupby("user").cumcount()
first = g[g.y == 1].groupby("user")["pos"].min()
g["before_first"] = g["pos"] < g["user"].map(first).fillna(np.inf)
per_user = g[g.before_first].groupby("user")["base"].mean()
is_att = D["umeta"].loc[per_user.index, "is_attacker"].to_numpy()
auc = roc_auc_score(is_att, per_user.to_numpy())
auc_f = max(auc, 1 - auc)
gate_pass = auc_f < 0.60
json.dump(dict(user_level_auc=auc, folded_auc=auc_f, threshold=0.60, personal_norm_kept=bool(gate_pass),
               users=int(len(per_user)), attackers=int(is_att.sum())), open(RES / "shortcut_gate.json", "w"), indent=1)
print(f"shortcut gate: folded AUC of base_rate = {auc_f:.3f} -> personal-norm features "
      f"{'KEPT' if gate_pass else 'DROPPED'}", flush=True)

ARMS = {"B": ordered(B), "C": ordered(B + AGG), "C+": ordered(B + AGG + DIST), "A-cv": FEATS,
        "D": FEATS + ROT + (NORM if gate_pass else [])}
X0 = pd.concat([D["X"], NF[ROT + NORM].astype("float32")], axis=1)
Xc = pd.concat([cases[FEATS].astype("float32"), NFc[ROT + NORM].astype("float32")], axis=1)
y0 = D["y"]
yc = cases["y_10"].to_numpy()
case_host = cases["host"].to_numpy()

# ------------------------------------------------------------------ Regime 1
log = []
orig = {a: np.full((R, len(X0)), np.nan) for a in ARMS}
csc = {a: np.full((R, len(Xc)), np.nan) for a in ARMS}
for (r, k), f in folds.groupby(["repeat", "fold"]):
    fit_u, es_u, val_u = (set(f.loc[f.role == ro, "user"]) for ro in ("fit", "es", "valid"))
    fr, er, vr = T.rows(D, fit_u), T.rows(D, es_u), T.rows(D, val_u)
    cr = np.flatnonzero(np.isin(case_host, list(val_u)))
    for a, cols in ARMS.items():
        Xv = pd.concat([X0.iloc[vr][cols], Xc.iloc[cr][cols]], ignore_index=True)
        s, n = models.fit_predict("lightgbm", X0.iloc[fr][cols], y0[fr], X0.iloc[er][cols], y0[er], Xv, seed=SEED + r)
        orig[a][r, vr] = s[:len(vr)]
        csc[a][r, cr] = s[len(vr):]
        log.append(dict(regime=1, repeat=r, fold=k, arm=a, features=len(cols), trees=n, valid_cases=len(cr)))
    print(f"regime 1 r{r} k{k} done {time.time() - t0:.0f}s", flush=True)
for a in ARMS:
    assert not np.isnan(orig[a]).any() and not np.isnan(csc[a]).any(), a
stageA = pd.read_parquet(WORK / "oof" / "lightgbm_h10.parquet").to_numpy().T
acv_identical = bool(np.array_equal(orig["A-cv"], stageA))
print("A-cv reproduces Stage A OOF exactly:", acv_identical, flush=True)
assert acv_identical, "A-cv does not reproduce Stage A"

# ------------------------------------------------------------------ Regime 2 (S4), repeat 0
reg2_orig = {a: np.full(len(X0), np.nan) for a in ("C+", "D")}
reg2_case = {a: np.full(len(Xc), np.nan) for a in ("C+", "D")}
h146 = cases["cohort"].to_numpy() == "H146"
for k, f in folds[folds.repeat == 0].groupby("fold"):
    fit_u, es_u, val_u = (set(f.loc[f.role == ro, "user"]) for ro in ("fit", "es", "valid"))
    fr, er, vr = T.rows(D, fit_u), T.rows(D, es_u), T.rows(D, val_u)
    train_c = np.flatnonzero(h146 & ~np.isin(case_host, list(val_u)))
    valid_c = np.flatnonzero(h146 & np.isin(case_host, list(val_u)))
    for a in ("C+", "D"):
        cols = ARMS[a]
        Xf = pd.concat([X0.iloc[fr][cols], Xc.iloc[train_c][cols]], ignore_index=True)
        yf = np.r_[y0[fr], yc[train_c]]
        Xv = pd.concat([X0.iloc[vr][cols], Xc.iloc[valid_c][cols]], ignore_index=True)
        s, n = models.fit_predict("lightgbm", Xf, yf, X0.iloc[er][cols], y0[er], Xv, seed=SEED)
        reg2_orig[a][vr] = s[:len(vr)]
        reg2_case[a][valid_c] = s[len(vr):]
        log.append(dict(regime=2, repeat=0, fold=k, arm=a, features=len(cols), trees=n, valid_cases=len(valid_c),
                        train_cases_rows=len(train_c)))
    print(f"regime 2 k{k} done {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ save
keys = cases[["case", "host", "cohort", "variant", "L", "row_index", "first_attack_index", "in_span", "y", "y_10"]]
for a in ARMS:
    pd.DataFrame({f"rep{r}": orig[a][r] for r in range(R)}).to_parquet(OUTS / f"orig_regime1_{a}.parquet", index=False)
    pd.concat([keys, pd.DataFrame({f"rep{r}": csc[a][r] for r in range(R)})], axis=1).to_parquet(
        OUTS / f"cases_regime1_{a}.parquet", index=False)
for a in ("C+", "D"):
    pd.DataFrame({"rep0": reg2_orig[a]}).to_parquet(OUTS / f"orig_regime2_{a}.parquet", index=False)
    pd.concat([keys, pd.DataFrame({"rep0": reg2_case[a]})], axis=1).to_parquet(OUTS / f"cases_regime2_{a}.parquet", index=False)
pd.DataFrame({"user": users}).to_parquet(OUTS / "orig_users.parquet", index=False)
pd.DataFrame(log).to_csv(RES / "training_log.csv", index=False)
json.dump(dict(arms={a: len(c) for a, c in ARMS.items()}, personal_norm_kept=bool(gate_pass), acv_reproduces_stageA=acv_identical,
               seconds=round(time.time() - t0)), open(RES / "training_summary.json", "w"), indent=1)
print("done", round(time.time() - t0), "s")
