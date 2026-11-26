"""
Gates G1-G3 + training of the six arms of the switching-weight study (PREREGISTRATION.md §4, §6).

G3 (integrity) is checked first. Every arm is trained on the Stage A folds with a local replica of the frozen
models.fit_predict lightgbm branch that also returns the fitted model (needed for SHAP). G1 then proves the replica is
exact (A0 == Stage A OOF), and G2 is the manipulation check (switching share of mean |TreeSHAP| on repeat-0 OOF rows).
Outputs: scores/p1_{arm}.parquet, scores/p2_{arm}.parquet, results/gates.json, results/training_log.csv,
results/shap_shares.csv. No estimand is computed here.
"""

import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent  # studies/<name>/ -> Model/ (moved 2026-11-26)
sys.path.insert(0, str(MODEL_DIR / "pipeline"))
import models  # noqa: E402
from common import SEED, WORK  # noqa: E402

PRE = (HERE / "PREREGISTRATION.sha256").read_text().strip()
assert hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest() == PRE, "pre-registration changed"
OUT = HERE / "scores"
OUT.mkdir(exist_ok=True)
RES = HERE / "results"
t0 = time.time()
CHECKS = []


def check(gate, name, ok, detail=""):
    CHECKS.append(dict(gate=gate, check=name, passed=bool(ok), detail=str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {gate} {name}  {detail}", flush=True)


# ------------------------------------------------------------------ G3 integrity (first)
before = json.loads((HERE / "hashes_before.json").read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256((MODEL_DIR / p).read_bytes()).hexdigest()  # noqa: E731
for p in ("pipeline/common.py", "pipeline/models.py", "pipeline/metrics.py", "pipeline/05_tune.py", "config/cv_folds.csv",
          "config/selection_rules.json", "data/model_dataset_995.parquet", "work/labels.parquet",
          "studies/controlled_rotation/data/counterfactual_cases.parquet"):
    check("G3", f"{p} identical to hashes_before.json",
          sha(p) == before.get(p, before.get(p.replace("studies/controlled_rotation/", "study_controlled_rotation/"))))  # pre-move key
if not all(c["passed"] for c in CHECKS):
    raise SystemExit("G3 failed -> study stops")

# ------------------------------------------------------------------ data
T = importlib.import_module("05_tune")
D = T.load()
X, y, folds = D["X"], D["y"], D["folds"]
FEATS = list(X.columns)
SW = [f for f in FEATS if f.startswith(("nr_alt", "nr_gap_alt_2", "nr_handoff", "dominant_channel_switches"))]
assert len(SW) == 13, SW
sw_idx = [FEATS.index(f) for f in SW]
cases = pd.read_parquet(MODEL_DIR / "studies" / "controlled_rotation" / "data" / "counterfactual_cases.parquet")
cases = cases[cases.cohort == "H146"].sort_values(["case", "row_index"]).reset_index(drop=True)
Xc = cases[FEATS].astype("float32")
host = cases["host"].to_numpy()


def contri(w):
    return [float(w) if f in SW else 1.0 for f in FEATS]


ARMS = {"A0": (FEATS, None), "W2": (FEATS, {"feature_contri": contri(2)}), "W5": (FEATS, {"feature_contri": contri(5)}),
        "W10": (FEATS, {"feature_contri": contri(10)}), "W0": (FEATS, {"feature_contri": contri(0)}), "SW-only": (SW, None)}


def fit(cols, params, fr, er, seed):
    """Replica of models.fit_predict('lightgbm', ...) returning the model and the early-stopping iteration."""
    m = models._lgbm(seed, models.MAX_TREES, params)
    m.fit(X.iloc[fr][cols], y[fr], eval_set=[(X.iloc[er][cols], y[er])],
          callbacks=[lgb.early_stopping(models.ES_ROUNDS, first_metric_only=True, verbose=False)])
    return m, int(m.best_iteration_) or models.MAX_TREES


R = 3
s1 = {a: np.full((R, len(X)), np.nan) for a in ARMS}
s2 = {a: np.full((R, len(Xc)), np.nan) for a in ARMS}
shap_sum = {a: np.zeros(len(FEATS)) for a in ARMS if a != "SW-only"}
log = []
for (r, k), f in folds.groupby(["repeat", "fold"]):
    fit_u, es_u, val_u = (set(f.loc[f.role == ro, "user"]) for ro in ("fit", "es", "valid"))
    fr, er, vr = T.rows(D, fit_u), T.rows(D, es_u), T.rows(D, val_u)
    cr = np.flatnonzero(np.isin(host, list(val_u)))
    for a, (cols, params) in ARMS.items():
        m, n = fit(cols, params, fr, er, SEED + r)
        s1[a][r, vr] = m.predict_proba(X.iloc[vr][cols], num_iteration=n)[:, 1]
        s2[a][r, cr] = m.predict_proba(Xc.iloc[cr][cols], num_iteration=n)[:, 1]
        if r == 0 and a in shap_sum:
            c = m.predict(X.iloc[vr][cols], pred_contrib=True, num_iteration=n)[:, :-1]
            shap_sum[a] += np.abs(c).sum(axis=0)
        used = m.booster_.feature_importance("split", iteration=n)
        log.append(dict(repeat=r, fold=k, arm=a, features=len(cols), trees=n,
                        switching_splits=int(sum(used[cols.index(s)] for s in SW)), total_splits=int(used.sum())))
    print(f"r{r} k{k} done {time.time() - t0:.0f}s", flush=True)
for a in ARMS:
    assert not np.isnan(s1[a]).any() and not np.isnan(s2[a]).any(), a

# ------------------------------------------------------------------ G1 reproduction
stageA = pd.read_parquet(WORK / "oof" / "lightgbm_h10.parquet").to_numpy().T
check("G1", "A0 OOF == Stage A OOF exactly (3 repeats)", np.array_equal(s1["A0"], stageA),
      f"max |diff| {np.abs(s1['A0'] - stageA).max():.2e}")

# ------------------------------------------------------------------ G2 manipulation check
share = {a: float(v[sw_idx].sum() / v.sum()) for a, v in shap_sum.items()}
lg = pd.DataFrame(log)
split_share = (lg.groupby("arm").switching_splits.sum() / lg.groupby("arm").total_splits.sum()).to_dict()
pd.DataFrame([dict(arm=a, shap_share_switching=share.get(a), split_share_switching=split_share.get(a)) for a in ARMS]) \
    .to_csv(RES / "shap_shares.csv", index=False)
mono = share["A0"] < share["W2"] < share["W5"] < share["W10"]
check("G2", "switching SHAP share A0 < W2 < W5 < W10", mono,
      " | ".join(f"{a} {share[a]:.4f}" for a in ("W0", "A0", "W2", "W5", "W10")))
check("G2", "share(W10) >= 2 x share(A0)", share["W10"] >= 2 * share["A0"], f"ratio {share['W10'] / share['A0']:.2f}")

# ------------------------------------------------------------------ save
keys1 = pd.DataFrame({"user": D["users"], "y_10": y})
keys2 = cases[["case", "host", "variant", "L", "row_index", "first_attack_index", "in_span", "y", "y_10"]]
for a in ARMS:
    pd.concat([keys1, pd.DataFrame({f"rep{r}": s1[a][r] for r in range(R)})], axis=1).to_parquet(OUT / f"p1_{a}.parquet", index=False)
    pd.concat([keys2, pd.DataFrame({f"rep{r}": s2[a][r] for r in range(R)})], axis=1).to_parquet(OUT / f"p2_{a}.parquet", index=False)
lg.to_csv(RES / "training_log.csv", index=False)
g1 = all(c["passed"] for c in CHECKS if c["gate"] == "G1")
g2 = all(c["passed"] for c in CHECKS if c["gate"] == "G2")
out = dict(preregistration_sha256=PRE, g1_passed=g1, g2_passed=g2, g3_passed=True, switching_shap_share=share,
           switching_split_share=split_share, consequence="stop (outcome 4)" if not g1 else
           "outcome 4 forced by G2 (estimands descriptive)" if not g2 else "proceed", checks=CHECKS,
           seconds=round(time.time() - t0))
(RES / "gates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("GATES:", out["consequence"], "|", round(time.time() - t0), "s")
