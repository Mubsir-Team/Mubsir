"""
Pre-test audit (dev only; the test part is not loaded into any matrix and no test score exists).

  P1  Calibration transfer: simulate "fold model -> final model" on dev. For each of the 15 outer folds a
      model is refit like the final model (fit + es users, fixed 237 trees, no early stopping) and scores
      the fold's validation users; the frozen Platt mapping is applied unchanged. Compared with the
      early-stopped fold models (the OOF the calibration was fitted on): calibration, score scale,
      benign-user peaks and FP counts at the frozen thresholds.
  P2  Risk definition: manifest formula == implementation; risk_peak = max daily risk per user.
  P3  Threshold unit: FP counts in 07 are user-level (peak over the period); row rates reported separately.
  P4  Operators and freezing: >, >= semantics checked on boundary values; thresholds read only from manifest.
  P5  Artifact integrity: model file, manifest, feature order, calibration, split, rules, dataset hashes;
      final model reproducible (refit -> identical predictions).
Writes results/pretest/* and, if every check passes, results/pretest/freeze.json (hashes frozen before test).
"""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import lightgbm as lgb  # noqa: E402
import models  # noqa: E402
from common import (SEED, WORK, RESULTS, MODEL_DIR, DATASET, SELECTION_RULES, FINAL_SPLIT, CV_FOLDS, sha256,
                    feature_list, write_json)  # noqa: E402
from metrics import threshold_at_fpr  # noqa: E402

T = importlib.import_module("05_tune")
C7 = None  # 07 is not imported (it would refit); its outputs are read from disk
OUT = RESULTS / "pretest"
OUT.mkdir(exist_ok=True)
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(dict(check=name, status="PASS" if ok else "FAIL", detail=str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


man = json.loads((MODEL_DIR / "model" / "manifest.json").read_text(encoding="utf-8"))
cal7 = json.loads((RESULTS / "07_calibration.json").read_text())
A, B = man["calibration"]["a"], man["calibration"]["b"]
EPS = 1e-7


def risk_of(p):
    """Exactly the manifest formula."""
    p = np.clip(p, EPS, 1 - EPS)
    return 100.0 / (1.0 + np.exp(-(A * np.log(p / (1 - p)) + B)))


D = T.load()
y, users = D["y"], D["users"]
benign = sorted({u for u in set(users) if D["umeta"].loc[u, "is_attacker"] == 0})
attackers = sorted(set(users) - set(benign))

# ============================================================ P5 integrity (first: everything else relies on it)
model_path = MODEL_DIR / "model" / "lgbm_y10.txt"
bst = lgb.Booster(model_file=str(model_path))
check("P5 model file SHA-256 = manifest", sha256(model_path) == man["provenance"]["model_file_sha256"])
check("P5 dataset SHA-256 = manifest", sha256(DATASET) == man["provenance"]["dataset_sha256"])
check("P5 selection_rules SHA-256 = manifest = registered",
      sha256(SELECTION_RULES) == man["provenance"]["selection_rules_sha256"] == T.RULES_SHA)
check("P5 feature order: manifest = pipeline feature_list = booster", man["features"] == feature_list() == bst.feature_name())
check("P5 booster has 237 trees = manifest n_estimators = 07", bst.num_trees() == man["params"]["n_estimators"] == cal7["n_trees"] == 237,
      bst.num_trees())
check("P5 calibration a, b: manifest = 07 results", A == cal7["a"] and B == cal7["b"], (A, B))
check("P5 dev 1% threshold: manifest = 07 results", man["thresholds"]["dev_1pct_user_fpr_risk"] == cal7["dev_1pct_user_fpr_risk"])
split = pd.read_csv(FINAL_SPLIT)
folds = pd.read_csv(CV_FOLDS)
test_users = set(split.loc[split.part == "test", "user"])
check("P5 split: 859 dev / 136 test (52 attackers, 84 benign), disjoint",
      len(set(users)) == 859 and len(test_users) == 136 and not (test_users & set(users))
      and split[split.part == "test"].is_attacker.sum() == 52)
check("P5 no test user in CV folds or in the training matrix", not (set(folds.user) & test_users))
check("P5 final model trained on dev rows only (manifest row count = dev rows)", man["training"]["dev_rows"] == len(y) == 89471)
refit = models.fit_final("lightgbm", D["X"], y, 237, seed=SEED)
d_refit = float(np.abs(refit.predict_proba(D["X"])[:, 1] - bst.predict(D["X"].to_numpy())).max())
check("P5 final model reproducible (refit vs saved file, all dev rows)", d_refit == 0.0, f"{d_refit:.1e}")

# ============================================================ P2 risk definition
p = bst.predict(D["X"].to_numpy()[:2000])
p_raw_margin = bst.predict(D["X"].to_numpy()[:2000], raw_score=True)
check("P2 booster.predict returns probabilities = sigmoid(raw margin)",
      np.allclose(p, 1 / (1 + np.exp(-p_raw_margin)), atol=1e-12))
manual = 100 / (1 + np.exp(-(A * np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))) + B)))
check("P2 risk = 100 * sigmoid(a * logit(p) + b) (manifest formula)", np.allclose(risk_of(p), manual, atol=1e-12))
check("P2 risk monotone in p and bounded in (0, 100)", np.all(np.diff(risk_of(np.sort(p))) >= 0) and
      risk_of(np.array([0.0, 1.0])).tolist() == [risk_of(np.array([EPS]))[0], risk_of(np.array([1 - EPS]))[0]])
check("P2 manifest states the formula used here", "a * logit(p) + b" in man["calibration"]["formula"])

# ============================================================ P3 threshold unit (07 definitions re-derived)
oof = pd.read_parquet(WORK / "oof" / "lightgbm_h10.parquet")
ops7 = pd.read_csv(RESULTS / "07_dev_operating_points.csv")
ok3 = True
frozen_fp = []
for r, c in enumerate(oof.columns):
    risk = risk_of(oof[c].to_numpy())
    peak = pd.Series(risk).groupby(users).max()          # risk_peak = max daily risk over the user's period
    for name in ("band_high_85", "band_medium_60", "dev_1pct_user_fpr"):
        got = ops7[(ops7.repeat == r) & (ops7.threshold == name)].iloc[0]
        t = got.value       # 07 used 85 / 60, and for the 1% point the repeat's OWN threshold
        fp_users = int((peak.loc[benign] > t).sum())
        row_rate = float((risk[D["benign_row"]] > t).mean())
        ok3 &= fp_users == got.fp_benign_users and abs(row_rate - got.row_alarm_rate_benign) < 1e-12
    frozen_fp.append(int((peak.loc[benign] > man["thresholds"]["dev_1pct_user_fpr_risk"]).sum()))
check("P3 07 FP counts are user-level (benign users whose PEAK daily risk exceeds the threshold) "
      "and row rates are reported separately; recomputed values match", ok3)
check("P3 note: 07's 1% row used each repeat's own threshold; at the FROZEN mean threshold the dev FP users are",
      True, f"{frozen_fp} of 511 per repeat (mean {np.mean(frozen_fp):.2f} = {np.mean(frozen_fp) / 511:.2%})")
nd = pd.Series(users).value_counts()
check("P3 observation period per benign dev user (workdays)", True,
      f"median {int(nd.loc[benign].median())}, min {int(nd.loc[benign].min())}, max {int(nd.loc[benign].max())}")

# ============================================================ P4 operators
t1 = man["thresholds"]["dev_1pct_user_fpr_risk"]
check("P4 manifest operators: high = risk > 85, medium = 60 < risk <= 85, low = risk <= 60, alert = risk > 85",
      man["thresholds"]["bands"] == {"low": "risk <= 60", "medium": "60 < risk <= 85", "high": "risk > 85"}
      and man["thresholds"]["alert"] == "risk > 85")


def band(r):
    return np.where(r > 85, "High", np.where(r > 60, "Medium", "Low"))


check("P4 boundary values: 60 -> Low, 60.0001 -> Medium, 85 -> Medium, 85.0001 -> High",
      band(np.array([60.0, 60.0001, 85.0, 85.0001])).tolist() == ["Low", "Medium", "Medium", "High"])
peaks0 = pd.Series(risk_of(oof["rep0"].to_numpy())).groupby(users).max().loc[benign].to_numpy()
check("P4 1% point semantics: alarm = peak > t (strict); t from threshold_at_fpr (share of peaks > t <= 1%)",
      np.mean(peaks0 > threshold_at_fpr(peaks0, 0.01)) <= 0.01)

# ============================================================ P1 calibration transfer (dev simulation)
rows = []
cmp_scores = {"fold_model": [], "final_like": [], "y": [], "benign_row": [], "users": []}
for (r, k), f in folds.groupby(["repeat", "fold"]):
    fit_u, es_u, val_u = (set(f.loc[f.role == ro, "user"]) for ro in ("fit", "es", "valid"))
    tr = T.rows(D, fit_u | es_u)
    vr = T.rows(D, val_u)
    m = models.fit_final("lightgbm", D["X"].iloc[tr], y[tr], 237, seed=SEED + r)
    s_final = m.predict_proba(D["X"].iloc[vr])[:, 1]
    s_fold = oof[f"rep{r}"].to_numpy()[vr]
    for name, s in (("fold_model", s_fold), ("final_like", s_final)):
        rk = risk_of(s) / 100
        peak = pd.Series(100 * rk).groupby(users[vr]).max()
        bu = [u for u in val_u if D["umeta"].loc[u, "is_attacker"] == 0]
        rows.append(dict(repeat=r, fold=k, model=name, mean_cal_pred=rk.mean(), observed=y[vr].mean(),
                         mean_raw=s.mean(), share_raw_above_0_9=(s > 0.9).mean(),
                         benign_peak_p95=float(np.quantile(peak.loc[bu], 0.95)),
                         fp_users_85=int((peak.loc[bu] > 85).sum()), fp_users_60=int((peak.loc[bu] > 60).sum()),
                         fp_users_t1=int((peak.loc[bu] > t1).sum()), benign_users=len(bu)))
    if r == 0:
        cmp_scores["fold_model"].append(s_fold); cmp_scores["final_like"].append(s_final)
        cmp_scores["y"].append(y[vr])
tr_df = pd.DataFrame(rows)
tr_df.to_csv(OUT / "P1_calibration_transfer_per_fold.csv", index=False)
summ = tr_df.groupby("model")[["mean_cal_pred", "observed", "mean_raw", "share_raw_above_0_9", "benign_peak_p95"]].mean()
summ = summ.join(tr_df.groupby("model")[["fp_users_85", "fp_users_60", "fp_users_t1", "benign_users"]].sum() / 3)
summ.to_csv(OUT / "P1_calibration_transfer_summary.csv")
yy = np.concatenate(cmp_scores["y"])
ece = {}
for name in ("fold_model", "final_like"):
    q = risk_of(np.concatenate(cmp_scores[name])) / 100
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(q, edges) - 1, 0, 9)
    ece[name] = float(sum((idx == i).mean() * abs(q[idx == i].mean() - yy[idx == i].mean()) for i in range(10) if (idx == i).any()))
mismatch = summ.loc["final_like", "mean_cal_pred"] - summ.loc["fold_model", "mean_cal_pred"]
check("P1 transfer: mapping applied unchanged to a final-like model (fixed 237 trees, fit+es users)", True,
      f"mean calibrated prediction {summ.loc['final_like', 'mean_cal_pred']:.4f} vs fold models "
      f"{summ.loc['fold_model', 'mean_cal_pred']:.4f} (observed {summ.loc['final_like', 'observed']:.4f}); "
      f"ECE {ece['final_like']:.4f} vs {ece['fold_model']:.4f} (repeat 0)")
check("P1 expected score-scale mismatch is small (|mean calibrated prediction diff| < 0.005)", abs(mismatch) < 0.005,
      f"{mismatch:+.4f}")
fp85 = (summ.loc["final_like", "fp_users_85"], summ.loc["fold_model", "fp_users_85"])
check("P1 FP users at 85 per repeat (511 benign): final-like vs fold models", True, f"{fp85[0]:.1f} vs {fp85[1]:.1f}")

# ============================================================ freeze
checks = pd.DataFrame(CHECKS)
checks.to_csv(OUT / "checks.csv", index=False)
all_pass = (checks.status == "PASS").all()
if all_pass:
    freeze = {"model_file": sha256(model_path), "manifest": sha256(MODEL_DIR / "model" / "manifest.json"),
              "final_split": sha256(FINAL_SPLIT), "selection_rules": sha256(SELECTION_RULES),
              "dataset": sha256(DATASET), "calibration": {"a": A, "b": B},
              "thresholds": {"medium": 60.0, "high": 85.0, "dev_1pct_user_fpr": t1, "operator": ">"}}
    write_json(OUT / "freeze.json", freeze)
print(checks.status.value_counts().to_dict(), "freeze written" if all_pass else "NOT FROZEN")
