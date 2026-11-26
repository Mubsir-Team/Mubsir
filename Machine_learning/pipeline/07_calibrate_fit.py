"""
07 — Platt calibration, final fit and dev-frozen thresholds (config/selection_rules.json stage_B).

Adopted configuration (checked from 05/06 outputs): LightGBM defaults, label y_10, 100 predictors,
no reweighting, no ablation.

  1. Platt scaling  risk = 100 * sigmoid(a * logit(p) + b), fitted on the Stage A out-of-fold scores of
     all 3 outer repeats stacked (each dev row 3 times), row level, target y_10. a, b are also fitted
     per repeat (stability), and calibration quality is checked with leave-fold-out Platt fits
     (each outer fold calibrated by a fit on the other 4 folds of its repeat).
  2. Final model: LightGBM defaults on all 859 dev users, tree count = mean best iteration of the 15
     Stage A folds (rounded), seed 42.
  3. Thresholds frozen from dev, on the risk scale: the fixed project bands 60 / 85 (not fitted), and the
     dev 1% benign-user FPR point (calibrated OOF peaks of the 511 benign dev users, mean of 3 repeats).
     Their dev operating characteristics are reported from OOF scores.

Outputs: model/lgbm_y10.txt, model/manifest.json, results/07_*.csv|json, reports/07_calibrate_fit.md
Test users are never loaded into any matrix here.
"""

import hashlib
import importlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

import models
from common import (SEED, WORK, RESULTS, REPORTS, MODEL_DIR, DATASET, SELECTION_RULES, sha256, write_json, md_table,
                    feature_list)
from metrics import threshold_at_fpr

T = importlib.import_module("05_tune")
OUT_MODEL = MODEL_DIR / "model"
OUT_MODEL.mkdir(exist_ok=True)
BANDS = {"medium": 60.0, "high": 85.0}
EPS = 1e-7


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def platt_fit(p, y, iters=100):
    """Unregularised 2-parameter logistic regression of y on logit(p) (Newton-Raphson)."""
    x = logit(p)
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = a * x + b
        q = 1 / (1 + np.exp(-z))
        w = q * (1 - q)
        g = np.array([np.sum((q - y) * x), np.sum(q - y)])
        H = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def platt_apply(p, a, b):
    return 1 / (1 + np.exp(-(a * logit(p) + b)))


def ece(y, q, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(q, edges) - 1, 0, bins - 1)
    tot = 0.0
    rows = []
    for i in range(bins):
        m = idx == i
        if m.any():
            tot += m.mean() * abs(q[m].mean() - y[m].mean())
            rows.append(dict(bin=f"{edges[i]:.1f}-{edges[i + 1]:.1f}", rows=int(m.sum()), mean_pred=q[m].mean(),
                             observed=y[m].mean()))
    return tot, pd.DataFrame(rows)


# ------------------------------------------------------------------ adopted configuration
adopt5 = json.loads((RESULTS / "05_adoption.json").read_text())
adopt6 = json.loads((RESULTS / "06_ablations.json").read_text())
assert adopt5["adopted"] is False, "05 adopted a tuned configuration; this script expects defaults"
assert all(v["decision"] in ("not adopted", "documentation only") for k, v in adopt6.items() if k != "default")

D = T.load()                      # also checks the selection_rules.json hash
y, users = D["y"], D["users"]
oof = pd.read_parquet(WORK / "oof" / "lightgbm_h10.parquet")
stageA = pd.read_csv(WORK / "oof" / "lightgbm_h10_folds.csv")
folds = D["folds"]
benign_users = sorted({u for u in set(users) if D["umeta"].loc[u, "is_attacker"] == 0})
attackers = sorted(set(users) - set(benign_users))

# ------------------------------------------------------------------ 1. Platt
P = np.concatenate([oof[c].to_numpy() for c in oof.columns])
Y = np.tile(y, len(oof.columns))
a, b = platt_fit(P, Y)
per_rep = [dict(repeat=r, a=v[0], b=v[1]) for r, c in enumerate(oof.columns) for v in [platt_fit(oof[c].to_numpy(), y)]]

# quality: leave-fold-out calibration within each repeat
qual = []
rel_tables = []
for r, c in enumerate(oof.columns):
    p = oof[c].to_numpy()
    cal = np.full(len(p), np.nan)
    vf = folds[(folds.repeat == r) & (folds.role == "valid")]
    for k in range(5):
        rk = T.rows(D, set(vf.loc[vf.fold == k, "user"]))
        other = np.setdiff1d(np.arange(len(p)), rk)
        ak, bk = platt_fit(p[other], y[other])
        cal[rk] = platt_apply(p[rk], ak, bk)
    e_raw, _ = ece(y, p)
    e_cal, tab = ece(y, cal)
    qual.append(dict(repeat=r, brier_raw=brier_score_loss(y, p), brier_calibrated_lfo=brier_score_loss(y, cal),
                     ece_raw=e_raw, ece_calibrated_lfo=e_cal, mean_pred_raw=p.mean(), mean_pred_cal=cal.mean(),
                     observed_rate=y.mean(), pr_auc_raw=average_precision_score(y, p),
                     pr_auc_cal=average_precision_score(y, cal)))
    rel_tables.append(tab.assign(repeat=r))
qual = pd.DataFrame(qual)
rel = pd.concat(rel_tables).groupby("bin")[["rows", "mean_pred", "observed"]].mean()

# ------------------------------------------------------------------ 2. dev operating points on the risk scale (OOF)
ops, thr_1pct = [], []
for r, c in enumerate(oof.columns):
    risk = 100 * platt_apply(oof[c].to_numpy(), a, b)
    f = pd.DataFrame({"user": users, "risk": risk, "w": D["in_window"]})
    peak = f.groupby("user")["risk"].max()
    t1 = threshold_at_fpr(peak.loc[benign_users].to_numpy(), 0.01)
    thr_1pct.append(t1)
    for name, t in (("band_medium_60", 60.0), ("band_high_85", 85.0), ("dev_1pct_user_fpr", t1)):
        hit = set(f.loc[f.w & (f.risk > t), "user"])
        ops.append(dict(repeat=r, threshold=name, value=t, user_recall=len(hit & set(attackers)) / len(attackers),
                        fp_benign_users=int((peak.loc[benign_users] > t).sum()), benign_users=len(benign_users),
                        row_alarm_rate_benign=float((risk[D["benign_row"]] > t).mean())))
ops = pd.DataFrame(ops)
ops_mean = ops.groupby("threshold")[["value", "user_recall", "fp_benign_users", "row_alarm_rate_benign"]].mean()
dev_thr = float(np.mean(thr_1pct))
band_share = pd.DataFrame({
    "band": ["Low (<=60)", "Medium (60-85]", "High (>85)"],
    "share_of_rows": [np.mean([(100 * platt_apply(oof[c].to_numpy(), a, b) <= 60).mean() for c in oof.columns]),
                      np.mean([((r_ := 100 * platt_apply(oof[c].to_numpy(), a, b)) > 60).mean() - (r_ > 85).mean()
                               for c in oof.columns]),
                      np.mean([(100 * platt_apply(oof[c].to_numpy(), a, b) > 85).mean() for c in oof.columns])]})

# ------------------------------------------------------------------ 3. final model
n_trees = int(round(stageA["n_trees"].mean()))
final = models.fit_final("lightgbm", D["X"], y, n_trees, seed=SEED)
model_path = OUT_MODEL / "lgbm_y10.txt"
final.booster_.save_model(str(model_path))
# reload check: the saved text model reproduces the in-memory predictions exactly
import lightgbm as lgb  # noqa: E402
re = lgb.Booster(model_file=str(model_path))
p_mem = final.predict_proba(D["X"].iloc[:5000])[:, 1]
p_re = re.predict(D["X"].iloc[:5000].to_numpy())
reload_diff = float(np.abs(p_mem - p_re).max())
assert reload_diff < 1e-12, reload_diff
assert list(re.feature_name()) == feature_list()

manifest = {
    "model": "mubsir-lgbm-y10",
    "version": "2.0.0-dev",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "algorithm": "LightGBM (binary), default configuration",
    "params": dict(models.LGBM_DEFAULTS, objective="binary", n_estimators=n_trees, random_state=SEED),
    "label": "y_10 = 1 if at least one injected attack day in the user's last 10 workdays (current day included)",
    "features": feature_list(),
    "n_features": len(feature_list()),
    "missing_values": "NaN allowed only in channel_distribution_shift_{5,10,20,30}wd (window without activity); "
                      "never replace with 0",
    "calibration": {"method": "platt", "a": a, "b": b,
                    "formula": "risk = 100 / (1 + exp(-(a * logit(p) + b))), p clipped to [1e-7, 1-1e-7]",
                    "fit_data": "Stage A out-of-fold scores, 3 repeats stacked, 859 dev users, y_10",
                    "meaning": "calibrated daily probability (x100) of at least one injected attack day in the last "
                               "10 workdays, under the dev population mix; not a per-employee probability"},
    "thresholds": {"bands": {"low": "risk <= 60", "medium": "60 < risk <= 85", "high": "risk > 85"},
                   "alert": "risk > 85", "dev_1pct_user_fpr_risk": dev_thr,
                   "note": "60/85 are fixed project decisions; dev_1pct_user_fpr_risk is frozen from dev OOF "
                           "(mean of 3 repeats) for reporting on test, not an alert rule"},
    "scope": "the 995 eligible multi-channel users (USB or HTTP baseline in 2010)",
    "training": {"dev_users": len(set(users)), "dev_rows": int(len(y)), "positives_y10": int(y.sum()),
                 "test_users_untouched": 136},
    "provenance": {"dataset_sha256": sha256(DATASET), "selection_rules_sha256": T.RULES_SHA,
                   "model_file_sha256": sha256(model_path)},
}
write_json(OUT_MODEL / "manifest.json", manifest)

# ------------------------------------------------------------------ outputs
write_json(RESULTS / "07_calibration.json", {"a": a, "b": b, "per_repeat": per_rep, "n_trees": n_trees,
                                              "dev_1pct_user_fpr_risk": dev_thr, "thr_per_repeat": thr_1pct,
                                              "reload_max_abs_diff": reload_diff})
qual.to_csv(RESULTS / "07_calibration_quality.csv", index=False)
rel.to_csv(RESULTS / "07_reliability.csv")
ops.to_csv(RESULTS / "07_dev_operating_points.csv", index=False)

text = f"""# 07 — Calibration, final fit, dev-frozen thresholds

Adopted configuration: LightGBM defaults, y_10, 100 predictors (05: tuning not adopted; 06: no ablation adopted).
Dev users only (859); test users never loaded.

## Platt calibration
Stacked fit (3 repeats, {len(P):,} rows): **a = {a:.4f}, b = {b:.4f}**.

{md_table(pd.DataFrame(per_rep), index=False)}

Quality, with each outer fold calibrated by a fit on the other 4 folds of its repeat:

{md_table(qual, index=False)}

Reliability (calibrated, leave-fold-out, mean of 3 repeats):

{md_table(rel)}

## Final model
- LightGBM defaults, **{n_trees} trees** (mean best iteration of the 15 Stage A folds), seed {SEED}, all {len(set(users))} dev users, {len(y):,} rows.
- Saved as `model/lgbm_y10.txt` (native text). Reloaded predictions match in-memory predictions (max diff {reload_diff:.1e}).
- `model/manifest.json` holds the feature order, calibration, thresholds, and provenance hashes.
- The OOF-fitted calibration does not see the final model's slightly larger training set (documented limitation).

## Dev operating points on the risk scale (OOF, calibrated, mean of 3 repeats)
{md_table(ops_mean)}

- `user_recall`: attackers with an in-window day above the threshold.
- `fp_benign_users`: of {len(benign_users)} benign dev users, those whose peak risk exceeds it.
- Frozen for test: bands 60 / 85, and dev 1% user-FPR point = **risk {dev_thr:.2f}**.

Share of dev rows per band (OOF, mean of 3 repeats):

{md_table(band_share, index=False)}
"""
(REPORTS / "07_calibrate_fit.md").write_text(text, encoding="utf-8")
print(text)
