"""
03 — Independent verification of model inputs, labels and split before any training.

Re-derives what 01 and 02 produced with independent code (plain loops) and checks
the forbidden-column, identity and leakage rules of the handover (§5.2) and the
audit (C1/C2/C3). Stops on the first failure.

Output: reports/03_verify_inputs.md
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import (KEYS, HORIZONS, ALLOWLIST, DATASET, LABELS, EVAL_META, FINAL_SPLIT, CV_FOLDS, REPORTS,
                    FORBIDDEN, CHECKS, check, sha256, feature_list, load_dataset, md_table)

# ------------------------------------------------------------------ dataset integrity
prep_report = (REPORTS / "pre_training_verification.md").read_text(encoding="utf-8")
m = re.search(r"SHA-256 `([0-9a-f]{64})`", prep_report)
digest = sha256(DATASET)
check("Model dataset SHA-256 matches prepare_dataset.py report", m is not None and m.group(1) == digest, digest[:16])

df = load_dataset()
feats = feature_list()
allow = pd.read_csv(ALLOWLIST)
excluded = set(allow.loc[allow["decision"] != "keep", "feature_name"])
check("100 predictors", len(feats) == 100, len(feats))
check("Dataset columns = keys + y + predictors", set(df.columns) == set(KEYS) | {"y"} | set(feats),
      set(df.columns) ^ (set(KEYS) | {"y"} | set(feats)))
check("No forbidden column among predictors", not (set(feats) & FORBIDDEN))
check("No excluded allowlist column (C1/C2/ids) among predictors", not (set(feats) & excluded), set(feats) & excluded)
check("No raw baseline column (p50/p90/p95/mad/active_rate) among predictors",
      not [c for c in feats if re.fullmatch(r"(email_count|usb_copy|http_upload|email_bytes)_(p50|p90|p95|mad)"
                                         r"|(email|usb|http)_active_(rate|days)|baseline_workdays", c)])
check("All predictors numeric", all(pd.api.types.is_numeric_dtype(df[c]) for c in feats))
check("No inf in predictors", np.isfinite(df[feats].fillna(0).to_numpy()).all())
nan_cols = [c for c in feats if df[c].isna().any()]
check("NaN only in channel_distribution_shift_*", all(c.startswith("channel_distribution_shift_") for c in nan_cols),
      nan_cols)
const = [c for c in feats if df[c].nunique(dropna=False) <= 1]
check("No constant predictor", not const, const)
per_user_const = [c for c in feats if df.groupby("user")[c].nunique(dropna=False).max() <= 1]
check("No predictor constant within every user (identity column)", not per_user_const, per_user_const)

# ------------------------------------------------------------------ labels (independent loop)
lab = pd.read_parquet(LABELS)
def same_keys(a, b):
    return (len(a) == len(b) and (a["user"].astype(str).to_numpy() == b["user"].astype(str).to_numpy()).all()
            and (a["day"].to_numpy() == b["day"].to_numpy()).all())


check("Labels aligned with dataset keys", same_keys(lab, df) and (lab["y"].to_numpy() == df["y"].to_numpy()).all())
mism = 0
for _, g in df.groupby("user", sort=False):
    y = g["y"].to_numpy()
    idx = g.index.to_numpy()
    for h in HORIZONS:
        ref = np.array([y[max(0, i - h + 1): i + 1].max() for i in range(len(y))])
        mism += int((ref != lab.loc[idx, f"y_{h}"].to_numpy()).sum())
check("y_h equals an independent trailing-window loop", mism == 0, mism)
for h in HORIZONS:
    eq = [c for c in feats if np.array_equal(df[c].fillna(-1).to_numpy(), lab[f"y_{h}"].to_numpy())]
    check(f"No predictor equals y_{h}", not eq, eq)

meta = pd.read_parquet(EVAL_META)
check("Evaluation metadata aligned with dataset keys", same_keys(meta, df))
check("Evaluation metadata holds no predictor column", not (set(meta.columns) & set(feats)))

# ------------------------------------------------------------------ split
split = pd.read_csv(FINAL_SPLIT)
folds = pd.read_csv(CV_FOLDS)
part = split.set_index("user")["part"]
check("Split covers the 995 dataset users", set(split["user"]) == set(df["user"]) and len(split) == 995)
dev_u, test_u = set(part[part == "dev"].index), set(part[part == "test"].index)
check("Dev and test users disjoint", not (dev_u & test_u))
grp = split.set_index("user")["group"]
check("Dev and test schedule groups disjoint", not (set(grp[list(dev_u)]) & set(grp[list(test_u)])))
d_days = set(map(tuple, df.loc[df.user.isin(dev_u), KEYS].astype(str).to_numpy()))
t_days = set(map(tuple, df.loc[df.user.isin(test_u), KEYS].astype(str).to_numpy()))
check("No (user, day) shared by dev and test", not (d_days & t_days))
check("CV folds use dev users only", set(folds["user"]) == dev_u)
check("Legacy-val users all in dev", split.loc[split.old_split == "val", "part"].eq("dev").all())

# ------------------------------------------------------------------ leakage scan (dev rows only)
dev_rows = df["user"].isin(dev_u).to_numpy()
scan = []
for c in feats:
    x = df.loc[dev_rows, c].fillna(-1).to_numpy()
    a = roc_auc_score(lab.loc[dev_rows, "y_10"], x)
    scan.append((c, max(a, 1 - a)))
scan = pd.DataFrame(scan, columns=["feature", "auc_y10_dev"]).sort_values("auc_y10_dev", ascending=False)
check("No single predictor separates y_10 almost perfectly (AUC < 0.99, dev only)",
      scan["auc_y10_dev"].max() < 0.99, scan.iloc[0].to_dict())
scan.to_csv(REPORTS.parent / "results" / "03_single_feature_auc_dev.csv", index=False)

report = f"""# 03 — Input, label and split verification

**{len(CHECKS)}/{len(CHECKS)} checks PASS.** Model dataset SHA-256 `{digest}`.

## Strongest single predictors for y_10 (dev rows only, AUC folded to >= 0.5)
{md_table(scan.head(10), index=False)}

Full list: `results/03_single_feature_auc_dev.csv`.

## Checks
""" + "\n".join(f"- [{'PASS' if ok else 'FAIL'}] {n} {d}" for n, ok, d in CHECKS) + "\n"
(REPORTS / "03_verify_inputs.md").write_text(report, encoding="utf-8")
print(scan.head(10).to_string(index=False))
