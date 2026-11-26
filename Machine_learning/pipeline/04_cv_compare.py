"""
04 — Stage A: compare the 5 algorithms x 3 label horizons on dev cross-validation,
then apply the pre-registered rules (config/selection_rules.json) to pick ONE
algorithm and ONE label horizon for the single production model.

Dev users only (config/cv_folds.csv, 15 folds). Test users are never loaded into X.

Outputs:
  work/oof/{algo}_h{h}.parquet   out-of-fold scores (repeat, row, score) for every dev row
  results/04_cv_folds.csv        per-fold metrics
  results/04_cv_summary.csv      per (algo, h) summary
  results/04_selection.json      the rule outcomes
  reports/04_cv_compare.md

Resumable: an (algo, h) whose OOF file exists is not retrained (delete work/oof to rerun).
"""

import json
import sys
import time

import numpy as np
import pandas as pd

from common import (SEED, HORIZONS, DEFAULT_HORIZON, LABELS, EVAL_META, FINAL_SPLIT, CV_FOLDS, SELECTION_RULES,
                    WORK, RESULTS, REPORTS, feature_list, load_dataset, write_json, md_table)
from metrics import row_metrics, user_metrics
from models import ALGOS, fit_predict

OOF_DIR = WORK / "oof"
OOF_DIR.mkdir(exist_ok=True)
rules = json.loads(SELECTION_RULES.read_text(encoding="utf-8"))

feats = feature_list()
df = load_dataset()
lab = pd.read_parquet(LABELS)
meta = pd.read_parquet(EVAL_META)
split = pd.read_csv(FINAL_SPLIT)
folds = pd.read_csv(CV_FOLDS)

dev_users = set(split.loc[split["part"] == "dev", "user"])
dev = (df["user"].astype(str).isin(dev_users)).to_numpy()
X = df.loc[dev, feats].astype("float32").reset_index(drop=True)
users = df.loc[dev, "user"].astype(str).reset_index(drop=True)
Y = {h: lab.loc[dev, f"y_{h}"].to_numpy() for h in HORIZONS}
y_day = lab.loc[dev, "y"].to_numpy()
M = meta.loc[dev].reset_index(drop=True)
benign_rows = (M["is_attacker"] == 0).to_numpy()
assert not set(users) - dev_users

user_rows = pd.Series(np.arange(len(users))).groupby(users.to_numpy()).apply(np.array).to_dict()


def rows_of(user_list):
    return np.concatenate([user_rows[u] for u in user_list])


fold_index = {}
for (rep, fold), f in folds.groupby(["repeat", "fold"]):
    fold_index[(rep, fold)] = {r: rows_of(f.loc[f.role == r, "user"].tolist()) for r in ("fit", "es", "valid")}

algos = sys.argv[1:] or list(ALGOS)
fold_rows = []
for algo in algos:
    for h in HORIZONS:
        path = OOF_DIR / f"{algo}_h{h}.parquet"
        if path.exists():
            print(f"skip {algo} h={h} (cached)")
            continue
        y = Y[h]
        oof = np.full((len(folds["repeat"].unique()), len(X)), np.nan)
        recs = []
        for (rep, fold), ix in fold_index.items():
            t0 = time.time()
            s, n_trees = fit_predict(algo, X.iloc[ix["fit"]], y[ix["fit"]], X.iloc[ix["es"]], y[ix["es"]],
                                     X.iloc[ix["valid"]], seed=SEED + rep)
            oof[rep, ix["valid"]] = s
            v = ix["valid"]
            r = row_metrics(y[v], s, benign_rows[v])
            r.update({f"day_{k}": val for k, val in row_metrics(y_day[v], s, benign_rows[v]).items()
                      if k in ("pr_auc", "roc_auc")})
            r.update(algo=algo, h=h, repeat=rep, fold=fold, n_trees=n_trees, seconds=round(time.time() - t0, 1))
            recs.append(r)
            print(f"{algo:14s} h={h:2d} rep={rep} fold={fold} PR-AUC={r['pr_auc']:.4f} "
                  f"R@1%={r['recall@1%fpr']:.4f} trees={n_trees} {r['seconds']}s", flush=True)
        assert not np.isnan(oof).any()
        pd.DataFrame(recs).to_csv(OOF_DIR / f"{algo}_h{h}_folds.csv", index=False)
        pd.DataFrame({f"rep{r}": oof[r] for r in range(oof.shape[0])}).to_parquet(path, index=False)

# ------------------------------------------------------------------ summarise every cached (algo, h)
fold_tab, summary = [], []
for algo in ALGOS:
    for h in HORIZONS:
        path = OOF_DIR / f"{algo}_h{h}.parquet"
        if not path.exists():
            continue
        f = pd.read_csv(OOF_DIR / f"{algo}_h{h}_folds.csv")
        fold_tab.append(f)
        oof = pd.read_parquet(path)
        pooled = []
        for col in oof.columns:
            frame = pd.DataFrame({"user": users, "is_attacker": M["is_attacker"], "in_window": M["in_window"],
                                  "wd_from_first_attack": M["wd_from_first_attack"], "score": oof[col]})
            pooled.append(user_metrics(frame))
        pooled = pd.DataFrame(pooled)
        row = {"algo": algo, "h": h}
        for k in ("pr_auc", "roc_auc", "recall@1%fpr", "recall@5%fpr", "day_pr_auc", "day_roc_auc"):
            row[k] = f[k].mean()
            row[k + "_sd"] = f[k].std()
        for k in pooled.columns:
            if k.startswith(("user_", "latency_", "benign_users")):
                row[k] = pooled[k].mean()
        row["mean_trees"] = f["n_trees"].mean() if f["n_trees"].notna().any() else np.nan
        row["seconds_per_fold"] = f["seconds"].mean()
        summary.append(row)
fold_tab = pd.concat(fold_tab)
summary = pd.DataFrame(summary)
fold_tab.to_csv(RESULTS / "04_cv_folds.csv", index=False)
summary.to_csv(RESULTS / "04_cv_summary.csv", index=False)

if set(summary["algo"]) != set(ALGOS) or len(summary) != len(ALGOS) * len(HORIZONS):
    print("Partial run: summary written, selection skipped.")
    sys.exit(0)

# ------------------------------------------------------------------ rule A: algorithm
A = rules["rule_A_algorithm"]
per_algo = summary.groupby("algo")[["recall@1%fpr", "pr_auc"]].mean()
inc = per_algo.loc[A["incumbent"]]
qualify = {a: per_algo.loc[a, "recall@1%fpr"] - inc["recall@1%fpr"] for a in A["challengers"]
           if per_algo.loc[a, "recall@1%fpr"] >= inc["recall@1%fpr"] + 0.005 and per_algo.loc[a, "pr_auc"] >= inc["pr_auc"]}
algo_sel = max(qualify, key=qualify.get) if qualify else A["incumbent"]

# ------------------------------------------------------------------ rule B: horizon
s = summary[summary.algo == algo_sel].set_index("h")
d = s.loc[DEFAULT_HORIZON]
qual_h = {}
for h in rules["rule_B_horizon"]["candidates"]:
    c = s.loc[h]
    ok = (c["user_recall@1%ufpr"] >= d["user_recall@1%ufpr"] + 0.005 and c["user_auc"] >= d["user_auc"] - 0.002
          and c["latency_median@1%ufpr"] <= d["latency_median@1%ufpr"] + 1)
    if ok:
        qual_h[h] = c["user_recall@1%ufpr"] - d["user_recall@1%ufpr"]
h_sel = max(qual_h, key=qual_h.get) if qual_h else DEFAULT_HORIZON

selection = {"algorithm": algo_sel, "horizon": int(h_sel), "rule_A_per_algo_mean_over_h": per_algo.round(5).to_dict(),
             "rule_A_qualifying": qualify, "rule_B_qualifying": {int(k): v for k, v in qual_h.items()},
             "mean_trees_selected": float(s.loc[h_sel, "mean_trees"])}
write_json(RESULTS / "04_selection.json", selection)

show = ["algo", "h", "pr_auc", "pr_auc_sd", "recall@1%fpr", "recall@1%fpr_sd", "recall@5%fpr", "roc_auc", "day_pr_auc",
        "user_auc", "user_recall@1%ufpr", "latency_median@1%ufpr", "user_recall@5%ufpr", "latency_median@5%ufpr",
        "mean_trees", "seconds_per_fold"]
report = f"""# 04 — Stage A: algorithms x label horizons (dev CV, 15 folds)

Rules: `config/selection_rules.json` (written before training). Test users not used.

**Selected algorithm: `{algo_sel}` · selected label horizon: y_{h_sel}**

## Summary (fold means; user metrics pooled per repeat, mean of 3)
{md_table(summary[show].sort_values(['h', 'pr_auc'], ascending=[True, False]), index=False)}

- `pr_auc`, `recall@*%fpr`, `roc_auc`: row level on the model's own label y_h (not comparable across h).
- `day_pr_auc`: row level on the attack-day label y (same target for every h).
- `user_*`, `latency_*`: comparable across h. Threshold = share of benign dev users that ever alarm (1% / 5%).
  Latency in workdays from the first injected day to the first in-window alarm.

## Rule A — algorithm (mean over h)
{md_table(per_algo)}

Qualifying challengers: {qualify or 'none'} -> **{algo_sel}**.

## Rule B — horizon ({algo_sel})
{md_table(s[['user_auc', 'user_recall@1%ufpr', 'latency_median@1%ufpr', 'user_recall@5%ufpr', 'day_pr_auc']])}

Qualifying candidates: {qual_h or 'none'} -> **y_{h_sel}**.
"""
(REPORTS / "04_cv_compare.md").write_text(report, encoding="utf-8")
print(report)
