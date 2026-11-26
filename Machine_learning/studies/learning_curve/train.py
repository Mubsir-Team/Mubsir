"""
Gates G1-G3 and training of the six learning-curve arms (PREREGISTRATION.md §3-4).

G3 (integrity) is checked first, then G2 (subsampling) before any model is fitted, then G1 (F100 == Stage A OOF)
after training. Every arm uses the exact Stage A fit replica (same as studies/switching_weight/train.py); only the
fold's `fit` users are subsampled, the `es` and `valid` users are unchanged.
Outputs: scores/{arm}_s{seed}.parquet (OOF on dev rows, one column per repeat), results/gates.json,
results/subsample_sizes.csv, results/training_log.csv. No estimand is computed here.
"""

import hashlib
import importlib
import json
import math
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent
sys.path.insert(0, str(MODEL_DIR / "pipeline"))
import models  # noqa: E402
from common import SEED, WORK  # noqa: E402

PRE = (HERE / "PREREGISTRATION.sha256").read_text().strip()
assert hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest() == PRE, "pre-registration changed"
OUT, RES = HERE / "scores", HERE / "results"
OUT.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
DRY = "--dry-run" in sys.argv          # gates G3 + G2 only, no model fitted
t0 = time.time()
CHECKS = []


def check(gate, name, ok, detail=""):
    CHECKS.append(dict(gate=gate, check=name, passed=bool(ok), detail=str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {gate} {name}  {detail}", flush=True)


def stop(msg):
    (RES / "gates.json").write_text(json.dumps(dict(preregistration_sha256=PRE, consequence=msg, checks=CHECKS), indent=1),
                                    encoding="utf-8")
    raise SystemExit(msg)


# ------------------------------------------------------------------ G3 integrity (first)
before = json.loads((HERE / "hashes_before.json").read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256((MODEL_DIR / p).read_bytes()).hexdigest()  # noqa: E731
G3_FILES = sorted(p.relative_to(MODEL_DIR).as_posix() for p in (MODEL_DIR / "pipeline").glob("*.py")) + [
    "config/cv_folds.csv", "config/user_meta.csv", "config/selection_rules.json", "data/model_dataset_995.parquet",
    "work/labels.parquet"]
for p in G3_FILES:
    check("G3", f"{p} identical to hashes_before.json", sha(p) == before.get(p))
if not all(c["passed"] for c in CHECKS):
    stop("G3 failed -> study stops (outcome 4)")

# ------------------------------------------------------------------ data
T = importlib.import_module("05_tune")
D = T.load()
X, y, folds, um = D["X"], D["y"], D["folds"], D["umeta"]
ARMS = {"F30": 0.30, "F45": 0.45, "F60": 0.60, "F75": 0.75, "F86": 859 / 995, "F100": 1.0}
SEEDS = (0, 1)
R = 3


def subsample(fit_users, f, r, k, s):
    """PREREGISTRATION §3: whole schedule groups, per stratum, in one random order; nested across f."""
    u = um.loc[sorted(fit_users), ["stratum", "group"]]
    rng = np.random.default_rng(2026 + 1000 * s + 10 * r + k)
    take = []
    for _, g in u.groupby("stratum", sort=True):
        groups = sorted(g["group"].unique())
        order = rng.permutation(len(groups))
        target = max(1, round(f * len(g)))
        got = []
        for i in order:
            if len(got) >= target:
                break
            got += list(g.index[g["group"] == groups[i]])
        take += got
    return set(take)


# ------------------------------------------------------------------ G2 subsampling (before any fit)
plan, sizes = {}, []
for (r, k), fo in folds.groupby(["repeat", "fold"]):
    fit_u = set(fo.loc[fo.role == "fit", "user"])
    strata = set(um.loc[sorted(fit_u), "stratum"])
    for s in SEEDS:
        prev = None
        for a, f in sorted(ARMS.items(), key=lambda kv: kv[1]):
            sub = fit_u if a == "F100" else subsample(fit_u, f, r, k, s)
            plan[(r, k, s, a)] = sub
            dev = len(sub) / (f * len(fit_u)) - 1
            sizes.append(dict(repeat=r, fold=k, seed=s, arm=a, target=f * len(fit_u), realised=len(sub), rel_dev=dev,
                              strata_kept=len(set(um.loc[sorted(sub), "stratum"])), strata=len(strata),
                              nested=True if prev is None else prev <= sub, inside_fit=sub <= fit_u))
            prev = sub
sz = pd.DataFrame(sizes)
sz.to_csv(RES / "subsample_sizes.csv", index=False)
check("G2", "realised size within ±6% of f x n_fit (all arms)", (sz.rel_dev.abs() <= 0.06).all(),
      f"range {sz.rel_dev.min():+.3f} .. {sz.rel_dev.max():+.3f}")
f86 = sz[sz.arm == "F86"]
check("G2", "realised size within ±2% for F86", (f86.rel_dev.abs() <= 0.02).all(),
      f"range {f86.rel_dev.min():+.3f} .. {f86.rel_dev.max():+.3f}")
check("G2", "smaller arms are subsets of larger ones", sz.nested.all())
check("G2", "every stratum keeps at least one user", (sz.strata_kept == sz.strata).all())
check("G2", "no user outside the fold's fit users", sz.inside_fit.all())
if not all(c["passed"] for c in CHECKS if c["gate"] == "G2"):
    stop("G2 failed -> study stops (outcome 4)")
if DRY:
    print(f"dry run: G3 + G2 checked, no model fitted ({time.time() - t0:.0f}s)")
    raise SystemExit(0)


# ------------------------------------------------------------------ training
def fit(fr, er, seed):
    """Replica of models.fit_predict('lightgbm', ...) (Stage A configuration)."""
    m = models._lgbm(seed, models.MAX_TREES, None)
    m.fit(X.iloc[fr], y[fr], eval_set=[(X.iloc[er], y[er])],
          callbacks=[lgb.early_stopping(models.ES_ROUNDS, first_metric_only=True, verbose=False)])
    return m, int(m.best_iteration_) or models.MAX_TREES


runs = [(a, s) for a in ARMS for s in (SEEDS if a != "F100" else (0,))]
oof = {(a, s): np.full((R, len(X)), np.nan) for a, s in runs}
log = []
for (r, k), fo in folds.groupby(["repeat", "fold"]):
    es_u, val_u = (set(fo.loc[fo.role == ro, "user"]) for ro in ("es", "valid"))
    er, vr = T.rows(D, es_u), T.rows(D, val_u)
    for a, s in runs:
        sub = plan[(r, k, s, a)]
        m, n = fit(T.rows(D, sub), er, SEED + r)
        oof[(a, s)][r, vr] = m.predict_proba(X.iloc[vr], num_iteration=n)[:, 1]
        log.append(dict(repeat=r, fold=k, arm=a, seed=s, fit_users=len(sub), trees=n))
    print(f"r{r} k{k} done {time.time() - t0:.0f}s", flush=True)
for key, v in oof.items():
    assert not np.isnan(v).any(), key

# ------------------------------------------------------------------ G1 reproduction
stageA = pd.read_parquet(WORK / "oof" / "lightgbm_h10.parquet").to_numpy().T
check("G1", "F100 OOF == Stage A OOF exactly (3 repeats)", np.array_equal(oof[("F100", 0)], stageA),
      f"max |diff| {np.abs(oof[('F100', 0)] - stageA).max():.2e}")

# ------------------------------------------------------------------ save
keys = pd.DataFrame({"user": D["users"], "y_10": y, "in_window": D["in_window"]})
for (a, s), v in oof.items():
    pd.concat([keys, pd.DataFrame({f"rep{r}": v[r] for r in range(R)})], axis=1).to_parquet(OUT / f"{a}_s{s}.parquet", index=False)
pd.DataFrame(log).to_csv(RES / "training_log.csv", index=False)
g1 = all(c["passed"] for c in CHECKS if c["gate"] == "G1")
out = dict(preregistration_sha256=PRE, g1_passed=g1, g2_passed=True, g3_passed=True,
           consequence="proceed" if g1 else "stop (outcome 4)", checks=CHECKS, seconds=round(time.time() - t0))
(RES / "gates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("GATES:", out["consequence"], "|", round(time.time() - t0), "s")
