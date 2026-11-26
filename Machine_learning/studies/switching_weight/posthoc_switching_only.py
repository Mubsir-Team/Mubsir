"""
POST-HOC, DESCRIPTIVE ONLY (written after the registered analysis; no test; no effect on the outcome).
Why does the switching-only model (SW-only) detect rotating cases (ROT) less often than block cases (BLK)?

1. Refits the repeat-0 SW-only fold models exactly as train.py (deterministic; the case scores must equal
   scores/p2_SW-only.parquet rep0) and computes TreeSHAP on dev validation rows and on the existing cases.
2. What the model learned: per feature, the direction of its SHAP vs its value on dev rows (Spearman), and the mean
   feature value on benign rows vs attackers' attack rows (y_10 = 1).
3. What ROT and BLK look like: mean feature values and mean SHAP per feature on in-span case rows.
Writes results/posthoc_switching_only_*.csv.
"""

import importlib
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent  # studies/<name>/ -> Model/ (moved 2026-11-26)
sys.path.insert(0, str(MODEL_DIR / "pipeline"))
import models  # noqa: E402
from common import SEED  # noqa: E402

T = importlib.import_module("05_tune")
D = T.load()
X, y, folds = D["X"], D["y"], D["folds"]
SW = [f for f in X.columns if f.startswith(("nr_alt", "nr_gap_alt_2", "nr_handoff", "dominant_channel_switches"))]
um = D["umeta"]
cases = pd.read_parquet(MODEL_DIR / "studies" / "controlled_rotation" / "data" / "counterfactual_cases.parquet")
cases = cases[cases.cohort == "H146"].sort_values(["case", "row_index"]).reset_index(drop=True)
stored = pd.read_parquet(HERE / "scores" / "p2_SW-only.parquet")["rep0"].to_numpy()

dev_val, dev_shap, case_shap, case_score = [], [], np.full((len(cases), len(SW)), np.nan), np.full(len(cases), np.nan)
for k, f in folds[folds.repeat == 0].groupby("fold"):
    fit_u, es_u, val_u = (set(f.loc[f.role == ro, "user"]) for ro in ("fit", "es", "valid"))
    fr, er, vr = T.rows(D, fit_u), T.rows(D, es_u), T.rows(D, val_u)
    m = models._lgbm(SEED, models.MAX_TREES, None)
    m.fit(X.iloc[fr][SW], y[fr], eval_set=[(X.iloc[er][SW], y[er])],
          callbacks=[lgb.early_stopping(models.ES_ROUNDS, first_metric_only=True, verbose=False)])
    n = int(m.best_iteration_) or models.MAX_TREES
    dev_val.append(vr)
    dev_shap.append(m.predict(X.iloc[vr][SW], pred_contrib=True, num_iteration=n)[:, :-1])
    cr = np.flatnonzero(cases["host"].isin(val_u).to_numpy())
    case_shap[cr] = m.predict(cases.iloc[cr][SW], pred_contrib=True, num_iteration=n)[:, :-1]
    case_score[cr] = m.predict_proba(cases.iloc[cr][SW], num_iteration=n)[:, 1]
assert np.allclose(case_score, stored), "refit does not reproduce stored SW-only scores"
vr = np.concatenate(dev_val)
S = np.vstack(dev_shap)
Xv = X.iloc[vr][SW].reset_index(drop=True)
att = pd.Series(D["users"][vr]).map(um["is_attacker"]).to_numpy() == 1
yv = y[vr]

learned = []
for j, f in enumerate(SW):
    rho = spearmanr(Xv[f], S[:, j]).statistic
    learned.append(dict(feature=f, spearman_value_vs_shap=rho, mean_abs_shap=float(np.abs(S[:, j]).mean()),
                        mean_benign_rows=float(Xv.loc[~att, f].mean()), mean_attack_rows=float(Xv.loc[att & (yv == 1), f].mean())))
learned = pd.DataFrame(learned).sort_values("mean_abs_shap", ascending=False)
learned.to_csv(HERE / "results" / "posthoc_switching_only_learned.csv", index=False)

span = cases["in_span"].to_numpy()
rows = []
for v in ("ROT", "BLK", "FIX-EMAIL", "SIM"):
    msk = span & (cases["variant"] == v).to_numpy()
    for j, f in enumerate(SW):
        rows.append(dict(variant=v, feature=f, mean_value_in_span=float(cases.loc[msk, f].mean()),
                         mean_shap_in_span=float(case_shap[msk, j].mean())))
    rows.append(dict(variant=v, feature="(total log-odds from switching features)", mean_value_in_span=np.nan,
                     mean_shap_in_span=float(case_shap[msk].sum(axis=1).mean())))
cs = pd.DataFrame(rows)
cs.to_csv(HERE / "results" / "posthoc_switching_only_cases.csv", index=False)

pd.set_option("display.width", 220)
print("SW-only refit reproduces stored case scores: True")
print("\nWhat SW-only learned on dev (validation rows, repeat 0):")
print(learned.round(3).to_string(index=False))
print("\nIn-span case rows, mean value / mean SHAP:")
print(cs.pivot_table(index="feature", columns="variant", values=["mean_value_in_span", "mean_shap_in_span"]).round(3).to_string())
