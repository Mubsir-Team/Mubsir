"""
Training of arms A, R, H (PREREGISTRATION.md §6, §8). Requires results/gates.json with hard gates passed.
Existing 5x3 folds restricted to the 714 P1 users; LightGBM defaults via models.fit_predict; seed 42 + repeat.
Each fold model scores its validation users' P1 rows (OOF) and the existing H146 cases of hosts in that validation fold.
Outputs scores/p1_{arm}.parquet, scores/p2_{arm}.parquet, results/training_log.csv. No estimand is computed here.
"""

import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "pipeline"))
sys.path.insert(0, str(HERE))
import models  # noqa: E402
from common import SEED  # noqa: E402
from hab_features import R_NAMES, H_NAMES  # noqa: E402

assert hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest() == \
    (HERE / "PREREGISTRATION.sha256").read_text().strip(), "pre-registration changed"
gates = json.loads((HERE / "results" / "gates.json").read_text())
assert gates["hard_gates_passed"], "hard gate failed -> study stopped"
T = importlib.import_module("05_tune")
OUT = HERE / "scores"
OUT.mkdir(exist_ok=True)
t0 = time.time()

D = T.load()                                   # frozen loader: dev rows, manifest order, rules-hash check
FEATS = list(D["X"].columns)
p1 = pd.read_parquet(HERE / "data" / "features_p1.parquet")
P1 = set(p1.user)
mask = np.isin(D["users"], list(P1))
X = D["X"].loc[mask].reset_index(drop=True)
assert (D["users"][mask] == p1["user"].to_numpy()).all(), "P1 row order mismatch"
assert np.allclose(X[["E_email_today", "E_usb_today", "E_http_today"]].to_numpy(),
                   p1[["E_email_today", "E_usb_today", "E_http_today"]].to_numpy()), "E mismatch"
X = pd.concat([X, p1[R_NAMES + H_NAMES].astype("float32")], axis=1)
y = D["y"][mask]
assert (y == p1["y_10"].to_numpy()).all()
users = p1["user"].to_numpy()
row_of = pd.Series(np.arange(len(users))).groupby(users).apply(np.array).to_dict()
rows = lambda us: np.concatenate([row_of[u] for u in sorted(us)])  # noqa: E731

p2 = pd.read_parquet(HERE / "data" / "features_p2.parquet")
Xc = p2[FEATS].astype("float32").join(p2[R_NAMES + H_NAMES].astype("float32"))
host = p2["host"].to_numpy()

ARMS = {"A": FEATS, "R": FEATS + R_NAMES, "H": FEATS + H_NAMES}
folds = D["folds"][D["folds"].user.isin(P1)]
R = 3
s1 = {a: np.full((R, len(X)), np.nan) for a in ARMS}
s2 = {a: np.full((R, len(Xc)), np.nan) for a in ARMS}
log = []
for (r, k), f in folds.groupby(["repeat", "fold"]):
    fit_u, es_u, val_u = (set(f.loc[f.role == ro, "user"]) for ro in ("fit", "es", "valid"))
    fr, er, vr = rows(fit_u), rows(es_u), rows(val_u)
    cr = np.flatnonzero(np.isin(host, list(val_u)))
    for a, cols in ARMS.items():
        Xv = pd.concat([X.iloc[vr][cols], Xc.iloc[cr][cols]], ignore_index=True)
        s, n = models.fit_predict("lightgbm", X.iloc[fr][cols], y[fr], X.iloc[er][cols], y[er], Xv, seed=SEED + r)
        s1[a][r, vr], s2[a][r, cr] = s[:len(vr)], s[len(vr):]
        log.append(dict(repeat=r, fold=k, arm=a, features=len(cols), trees=n, fit_users=len(fit_u), es_users=len(es_u),
                        valid_users=len(val_u), valid_cases=int(len(np.unique(p2.loc[cr, "case"])))))
    print(f"r{r} k{k} done {time.time() - t0:.0f}s", flush=True)
for a in ARMS:
    assert not np.isnan(s1[a]).any() and not np.isnan(s2[a]).any(), a

keys1 = p1[["user", "day", "row_in_dataset", "y", "y_10"]]
keys2 = p2[["case", "host", "cohort", "variant", "L", "row_index", "first_attack_index", "in_span", "y", "y_10"]]
for a in ARMS:
    pd.concat([keys1, pd.DataFrame({f"rep{r}": s1[a][r] for r in range(R)})], axis=1).to_parquet(OUT / f"p1_{a}.parquet", index=False)
    pd.concat([keys2, pd.DataFrame({f"rep{r}": s2[a][r] for r in range(R)})], axis=1).to_parquet(OUT / f"p2_{a}.parquet", index=False)
pd.DataFrame(log).to_csv(HERE / "results" / "training_log.csv", index=False)
json.dump(dict(arms={a: len(c) for a, c in ARMS.items()}, p1_users=len(P1), p1_rows=len(X), p2_cases=int(p2.case.nunique()),
               seconds=round(time.time() - t0)), open(HERE / "results" / "training_summary.json", "w"), indent=1)
print("done", round(time.time() - t0), "s")
