"""
12 — Release: finish the model bundle and verify the whole pipeline end to end.

Bundle files written here: model/example_input.parquet (60 dev rows, 2 users), model/requirements.txt,
model/.gitattributes (* -text, keeps hashes stable on Windows), model/SHA256SUMS.txt (last).
The model, manifest and calibration are NOT modified (their hashes are frozen in results/pretest/freeze.json).

Checks (stop on first failure): bundle integrity; scorer reproduces the frozen test scores exactly;
explain() additivity; input validation; manifest <-> freeze <-> files; recorded decisions (04-08);
split and leakage invariants; every report present.
Outputs: results/12_release_checks.csv, reports/12_release_verification.md
"""

import importlib.metadata as md
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from common import (WORK, RESULTS, REPORTS, MODEL_DIR, DATASET, SELECTION_RULES, FINAL_SPLIT, CV_FOLDS, USER_META,
                    EVAL_META, FORBIDDEN, CHECKS, check, sha256, feature_list, load_dataset, md_table)

B = MODEL_DIR / "model"
sys.path.insert(0, str(B))
fz = json.loads((RESULTS / "pretest" / "freeze.json").read_text())

# ------------------------------------------------------------------ bundle files
df = load_dataset()
um = pd.read_csv(USER_META).set_index("user")
split = pd.read_csv(FINAL_SPLIT)
meta = pd.read_parquet(EVAL_META)
dev_u = split.loc[split.part == "dev"]
att = sorted(dev_u.loc[dev_u.is_attacker == 1, "user"])
att = next(u for u in att if um.loc[u, "window_workdays"] == 10)
ben = sorted(dev_u.loc[dev_u.is_attacker == 0, "user"])[0]
ex = []
for u in (att, ben):
    rows = df[df.user.astype(str) == u]
    if u == att:
        start = meta.loc[rows.index, "in_window"].to_numpy().argmax()
        rows = rows.iloc[max(0, start - 15): max(0, start - 15) + 30]
    else:
        rows = rows.iloc[:30]
    ex.append(rows[["user", "day"] + feature_list()])
example = pd.concat(ex).reset_index(drop=True)
example["user"] = example["user"].astype(str)
example.to_parquet(B / "example_input.parquet", index=False)
reqs = [f"{p}=={md.version(p)}" for p in ("lightgbm", "numpy", "pandas", "pyarrow")]
(B / "requirements.txt").write_text("\n".join(reqs) + "\n", encoding="utf-8")
(B / ".gitattributes").write_bytes(b"* -text\n")
files = ["lgbm_y10.txt", "manifest.json", "mubsir_infer.py", "verify_bundle.py", "example_input.parquet",
         "requirements.txt", "README.md", ".gitattributes"]
(B / "SHA256SUMS.txt").write_text("".join(f"{sha256(B / f)}  {f}\n" for f in files), encoding="utf-8")

# ------------------------------------------------------------------ 1. bundle integrity
r = subprocess.run([sys.executable, str(B / "verify_bundle.py"), str(B)], capture_output=True, text=True)
check("verify_bundle.py (stdlib) passes", r.returncode == 0, r.stdout.strip())
check("model file unchanged since freeze", sha256(B / "lgbm_y10.txt") == fz["model_file"])
check("manifest unchanged since freeze", sha256(B / "manifest.json") == fz["manifest"])
check("split / rules / dataset unchanged since freeze",
      sha256(FINAL_SPLIT) == fz["final_split"] and sha256(SELECTION_RULES) == fz["selection_rules"]
      and sha256(DATASET) == fz["dataset"])

from mubsir_infer import MubsirScorer  # noqa: E402
s = MubsirScorer(B)
man = s.manifest
check("scorer feature order = pipeline feature list (100)", s.features == feature_list() and len(s.features) == 100)

# ------------------------------------------------------------------ 2. scorer reproduces the frozen test scores
ts = pd.read_parquet(WORK / "test" / "test_scores.parquet")
test_rows = df[df.user.astype(str).isin(set(split.loc[split.part == "test", "user"]))].copy()
test_rows["user"] = test_rows["user"].astype(str)
out = s.score(test_rows)
check("bundle scorer reproduces the frozen test risk exactly", np.array_equal(out["risk"].to_numpy(), ts["risk"].to_numpy()),
      f"max diff {np.abs(out['risk'].to_numpy() - ts['risk'].to_numpy()).max():.1e}")
check("levels follow the frozen operators", (out["level"] == np.where(out.risk > 85, "High",
      np.where(out.risk > 60, "Medium", "Low"))).all() and (out["alert"] == (out.risk > 85)).all())
us = s.score_users(test_rows)
peak = ts.groupby("user")["risk"].max()
check("score_users: risk_peak = max daily risk per user", np.allclose(us.set_index("user")["risk_peak"], peak.loc[us.user]))

# ------------------------------------------------------------------ 3. explain additivity
ex_out = s.explain(example, k=100)
X = example[s.features].astype("float32").to_numpy()
contrib = s.booster.predict(X, pred_contrib=True)
logit_cal = s.a * contrib.sum(axis=1) + s.b
risk_from_logit = 100 / (1 + np.exp(-logit_cal))
check("explain: base + all contributions = calibrated logit (risk reproduced)",
      np.allclose(risk_from_logit, s.score(example)["risk"], atol=1e-6))
check("explain: one dict per row with top features", len(ex_out) == len(example) and "top_features" in ex_out[0])

# ------------------------------------------------------------------ 4. input validation
bad = example.drop(columns=[s.features[0]])
try:
    s.score(bad); ok = False
except ValueError:
    ok = True
check("missing feature column raises ValueError", ok)
bad = example.copy(); bad.loc[0, "cumulative_deviation_10wd"] = np.nan
try:
    s.score(bad); ok = False
except ValueError:
    ok = True
check("NaN outside channel_distribution_shift_* raises ValueError", ok)
okc = example.copy(); okc.loc[0, "channel_distribution_shift_5wd"] = np.nan
check("NaN in channel_distribution_shift_* accepted", len(s.score(okc)) == len(okc))
bad = example.copy(); bad.loc[0, "E_usb_today"] = np.inf
try:
    s.score(bad); ok = False
except ValueError:
    ok = True
check("infinite value raises ValueError", ok)

# ------------------------------------------------------------------ 5. recorded decisions
sel = json.loads((RESULTS / "04_selection.json").read_text())
check("Stage A decision: lightgbm, y_10", sel["algorithm"] == "lightgbm" and sel["horizon"] == 10)
check("Stage B tuning not adopted", json.loads((RESULTS / "05_adoption.json").read_text())["adopted"] is False)
abl = json.loads((RESULTS / "06_ablations.json").read_text())
check("no ablation adopted", all(v["decision"] in ("not adopted", "documentation only") for k, v in abl.items() if k != "default"))
pre = pd.read_csv(RESULTS / "pretest" / "checks.csv")
check("pre-test audit all PASS before test", (pre.status == "PASS").all(), f"{len(pre)} checks")
check("manifest params = defaults, 237 trees", man["params"]["n_estimators"] == 237 and man["params"]["num_leaves"] == 31)

# ------------------------------------------------------------------ 6. split / leakage invariants
folds = pd.read_csv(CV_FOLDS)
test_u = set(split.loc[split.part == "test", "user"])
check("no test user in CV folds", not (set(folds.user) & test_u))
check("dev/test schedule groups disjoint", not (set(split.loc[split.part == "dev", "group"]) & set(split.loc[split.part == "test", "group"])))
check("no forbidden column among model features", not (set(s.features) & FORBIDDEN))
check("final model trained on dev rows only", man["training"]["dev_rows"] == int((~df.user.astype(str).isin(test_u)).sum()))

# ------------------------------------------------------------------ 7. reports present
need = ["pre_training_verification.md", "01_labels.md", "02_split.md", "03_verify_inputs.md", "04_cv_compare.md",
        "audit_stageA.md", "audit_stageA2.md", "stageB_preregistration.md", "stageB_preflight.md", "05_tune.md",
        "06_ablations.md", "07_calibrate_fit.md", "08_test_evaluation.md", "09_breakdown.md", "10_shap.md",
        "11_latency.md"]
missing = [f for f in need if not (REPORTS / f).exists()]
check("all stage reports present", not missing, missing)

pd.DataFrame(CHECKS, columns=["check", "ok", "detail"]).to_csv(RESULTS / "12_release_checks.csv", index=False)
sums = (B / "SHA256SUMS.txt").read_text(encoding="utf-8")
text = f"""# 12 — Release verification

**{len(CHECKS)}/{len(CHECKS)} checks PASS.**

## Bundle (`model/`)
```
{sums}```
Built with: {', '.join(reqs)}.

## Checks
""" + "\n".join(f"- [PASS] {n} {d}" for n, ok, d in CHECKS) + "\n"
(REPORTS / "12_release_verification.md").write_text(text, encoding="utf-8")
print(text)
