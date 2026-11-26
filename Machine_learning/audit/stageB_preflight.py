"""
Stage B pre-flight verification (read-only with respect to the pipeline).

No Optuna, no calibration, no test scores. Writes only results/preflight_stageB/*.

  V1  Default baseline OOF: outer-validation users never in training
      (role disjointness + in-memory refit of all 15 folds must reproduce the cached
      OOF exactly + label-poison test: randomising validation labels changes nothing)
  V2  Early stopping: eval set = training-side es users only (booster eval keys,
      disjointness, best iteration reproduced)
  V3  User-recall thresholds: which users AND which models the frozen threshold depends on
  V4  Duration-macro PR-AUC: two independent implementations agree; every duration group
      present in every outer-train / outer-valid / planned inner fold
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import models  # noqa: E402
from common import (SEED, LABELS, EVAL_META, FINAL_SPLIT, CV_FOLDS, USER_META, WORK, RESULTS,
                    feature_list, load_dataset)  # noqa: E402

OUT = RESULTS / "preflight_stageB"
OUT.mkdir(exist_ok=True)
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(dict(check=name, status="PASS" if ok else "MISMATCH", detail=str(detail)))
    print(f"[{'PASS' if ok else 'MISMATCH'}] {name}  {detail}", flush=True)


feats = feature_list()
df = load_dataset()
lab = pd.read_parquet(LABELS)
meta = pd.read_parquet(EVAL_META)
split = pd.read_csv(FINAL_SPLIT)
umeta = pd.read_csv(USER_META).set_index("user")
folds = pd.read_csv(CV_FOLDS)
dev_users = set(split.loc[split.part == "dev", "user"])
dev = df["user"].astype(str).isin(dev_users).to_numpy()
X = df.loc[dev, feats].astype("float32").reset_index(drop=True)
users = df.loc[dev, "user"].astype(str).reset_index(drop=True).to_numpy()
y = lab.loc[dev, "y_10"].to_numpy()
grp = umeta["group"]
row_of = pd.Series(np.arange(len(users))).groupby(users).apply(np.array).to_dict()


def rows(us):
    return np.concatenate([row_of[u] for u in us])


# ================================================================ V1 + V2
role_problems, group_problems, cover_problems = 0, 0, 0
for (r, k), f in folds.groupby(["repeat", "fold"]):
    s = {ro: set(f.loc[f.role == ro, "user"]) for ro in ("fit", "es", "valid")}
    role_problems += len(s["fit"] & s["valid"]) + len(s["es"] & s["valid"]) + len(s["fit"] & s["es"])
    g = {ro: set(grp[list(v)]) for ro, v in s.items()}
    group_problems += len(g["fit"] & g["valid"]) + len(g["es"] & g["valid"]) + len(g["fit"] & g["es"])
    cover_problems += int((s["fit"] | s["es"] | s["valid"]) != dev_users)
check("V1 fit / es / valid users disjoint in all 15 outer folds", role_problems == 0, role_problems)
check("V1 fit / es / valid schedule groups disjoint in all 15 outer folds", group_problems == 0, group_problems)
check("V1 fit + es + valid = all dev users in every outer fold", cover_problems == 0, cover_problems)

src = inspect.getsource(models.fit_predict)
check("V2 lightgbm eval_set is the es data only (source)", "eval_set=[(X_es, y_es)]" in src)

cached = pd.read_parquet(WORK / "oof" / "lightgbm_h10.parquet")
stageA_folds = pd.read_csv(WORK / "oof" / "lightgbm_h10_folds.csv")
max_diff, tree_mismatch, poison_diff, eval_keys_bad, scored_by_own_fold = 0.0, 0, 0.0, 0, 0
rng = np.random.default_rng(0)
import lightgbm as lgb  # noqa: E402
for (r, k), f in folds.groupby(["repeat", "fold"]):
    ix = {ro: rows(f.loc[f.role == ro, "user"].tolist()) for ro in ("fit", "es", "valid")}
    # refit exactly as Stage A
    s, n = models.fit_predict("lightgbm", X.iloc[ix["fit"]], y[ix["fit"]], X.iloc[ix["es"]], y[ix["es"]],
                              X.iloc[ix["valid"]], seed=SEED + r)
    max_diff = max(max_diff, float(np.abs(s - cached[f"rep{r}"].to_numpy()[ix["valid"]]).max()))
    n_a = stageA_folds.loc[(stageA_folds.repeat == r) & (stageA_folds.fold == k), "n_trees"].item()
    tree_mismatch += int(n != n_a)
    # poison test: random labels for validation rows; they are never passed, so nothing may change
    y_p = y.copy()
    y_p[ix["valid"]] = rng.integers(0, 2, len(ix["valid"]))
    s_p, _ = models.fit_predict("lightgbm", X.iloc[ix["fit"]], y_p[ix["fit"]], X.iloc[ix["es"]], y_p[ix["es"]],
                                X.iloc[ix["valid"]], seed=SEED + r)
    poison_diff = max(poison_diff, float(np.abs(s_p - s).max()))
    # booster evaluation sets: must be the single es set
    m = models._lgbm(SEED + r, models.MAX_TREES)
    m.fit(X.iloc[ix["fit"]], y[ix["fit"]], eval_set=[(X.iloc[ix["es"]], y[ix["es"]])],
          callbacks=[lgb.early_stopping(models.ES_ROUNDS, first_metric_only=True, verbose=False)])
    eval_keys_bad += int(list(m.evals_result_) != ["valid_0"] or m.best_iteration_ != n)
    scored_by_own_fold += len(ix["valid"])
check("V1 in-memory refit reproduces the cached default OOF (15 folds)", max_diff == 0.0, f"max abs diff {max_diff:.2e}")
check("V1/V2 refit tree counts equal Stage A best iterations", tree_mismatch == 0, tree_mismatch)
check("V1 label-poison test: random validation labels change no prediction", poison_diff == 0.0, f"{poison_diff:.2e}")
check("V2 booster evaluated only on the es set; best iteration reproduced", eval_keys_bad == 0, eval_keys_bad)
check("V1 every dev row scored exactly once per repeat by its own outer fold",
      scored_by_own_fold == 3 * len(users), f"{scored_by_own_fold} = 3 x {len(users)}")

# ================================================================ V3 thresholds
# The Stage A audit's frozen leave-fold-out (LFO) threshold for outer fold k uses OOF peaks of benign
# users in the other folds j != k. Those scores come from model j, whose fit/es users include fold k.
dep = []
for r in range(3):
    fr = folds[folds.repeat == r]
    for k in range(5):
        valid_k = set(fr.loc[(fr.fold == k) & (fr.role == "valid"), "user"])
        src_users = set(fr.loc[(fr.fold != k) & (fr.role == "valid"), "user"]) - set(umeta.index[umeta.is_attacker == 1])
        models_used = range(5)
        trained_on_k = [j for j in models_used if j != k and
                        valid_k <= set(fr.loc[(fr.fold == j) & fr.role.isin(["fit", "es"]), "user"])]
        dep.append(dict(repeat=r, fold=k, threshold_source_benign_users=len(src_users),
                        source_users_overlap_valid_k=len(src_users & valid_k),
                        sibling_models_used=4, sibling_models_trained_on_fold_k_users=len(trained_on_k)))
dep = pd.DataFrame(dep)
dep.to_csv(OUT / "V3_lfo_threshold_dependencies.csv", index=False)
check("V3 LFO threshold source users exclude the outer-validation fold",
      (dep.source_users_overlap_valid_k == 0).all())
check("V3 LFO threshold source SCORES come from models not trained on the outer-validation fold",
      (dep.sibling_models_trained_on_fold_k_users == 0).all(),
      f"in every fold, all {dep.sibling_models_trained_on_fold_k_users.iloc[0]} sibling models were trained on fold-k users")

# planned replacement: threshold from inner OOF of outer-train users (only outer-train data). Feasibility:
inner_rows = []
for (r, k), f in folds.groupby(["repeat", "fold"]):
    tr = umeta.loc[f.loc[f.role != "valid", "user"]].reset_index()
    sg = StratifiedGroupKFold(5, shuffle=True, random_state=142 + 10 * r + k)
    for i, (a, b) in enumerate(sg.split(tr, tr["stratum"], tr["group"])):
        sub = tr.iloc[a].reset_index(drop=True)
        _, es_i = next(StratifiedGroupKFold(7, shuffle=True, random_state=1142 + 100 * r + 10 * k + i)
                       .split(sub, sub["stratum"], sub["group"]))
        val_g, tr_g, es_g = set(tr.iloc[b]["group"]), set(sub.drop(index=es_i)["group"]), set(sub.iloc[es_i]["group"])
        inner_rows.append(dict(repeat=r, fold=k, inner=i, inner_valid_users=len(b),
                               inner_valid_benign=int((tr.iloc[b].is_attacker == 0).sum()),
                               es_share=len(es_i) / len(sub),
                               group_overlap=len(val_g & tr_g) + len(val_g & es_g) + len(tr_g & es_g),
                               durations_in_inner_valid=tr.iloc[b].loc[tr.iloc[b].is_attacker == 1, "window_workdays"].nunique()))
    inner_rows[-1]["outer_train_benign"] = int((tr.is_attacker == 0).sum())
inner = pd.DataFrame(inner_rows)
inner.to_csv(OUT / "V4_planned_inner_splits.csv", index=False)
check("Planned inner splits: no schedule group shared between inner fit / es / valid", (inner.group_overlap == 0).all())
check("Planned inner splits: early-stopping share 10-20%", inner.es_share.between(0.10, 0.20).all(),
      f"{inner.es_share.min():.3f}-{inner.es_share.max():.3f}")
check("Planned inner splits: all 6 durations in every inner validation fold", (inner.durations_in_inner_valid == 6).all(),
      inner.durations_in_inner_valid.min())
otb = inner.outer_train_benign.dropna()
check("Inner-OOF threshold resolution: benign outer-train users per outer fold", otb.min() >= 300,
      f"{int(otb.min())}-{int(otb.max())} benign users -> k = {int(np.floor(0.01 * otb.min()))}-{int(np.floor(0.01 * otb.max()))} FP at 1%")

# ================================================================ V4 duration-macro implementations
dur = umeta["window_workdays"]
row_dur = pd.Series(users).map(dur).to_numpy()
benign = pd.Series(users).map(umeta["is_attacker"]).to_numpy() == 0


def macro_a(score, mask):
    vals = []
    for d in (5, 10, 20, 30, 40, 50):
        m = mask & (benign | (row_dur == d))
        vals.append(average_precision_score(y[m], score[m]))
    return float(np.mean(vals)), vals


def macro_b(score, mask):
    """Independent implementation: explicit precision/recall steps over sorted unique thresholds."""
    vals = []
    for d in (5, 10, 20, 30, 40, 50):
        m = mask & (benign | (row_dur == d))
        s, t = score[m], y[m]
        order = np.argsort(-s, kind="mergesort")
        s, t = s[order], t[order]
        distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
        tp = np.cumsum(t)[distinct]
        fp = (distinct + 1) - tp
        prec, rec = tp / (tp + fp), tp / t.sum()
        vals.append(float(np.sum(np.diff(np.r_[0.0, rec]) * prec)))
    return float(np.mean(vals)), vals


a2 = pd.read_csv(RESULTS / "audit_stageA2" / "2_objectives_all.csv")
a2 = a2[(a2.algo == "lightgbm") & (a2.level == "pooled_repeat")].set_index("repeat")
allrows = np.ones(len(users), bool)
d_ab, d_audit = 0.0, 0.0
for r in range(3):
    s = cached[f"rep{r}"].to_numpy()
    va, _ = macro_a(s, allrows)
    vb, _ = macro_b(s, allrows)
    d_ab = max(d_ab, abs(va - vb))
    d_audit = max(d_audit, abs(va - a2.loc[r, "duration_macro_row_pr_auc"]))
check("V4 two independent duration-macro implementations agree (pooled, 3 repeats)", d_ab < 1e-12, f"{d_ab:.1e}")
check("V4 implementation equals the audit-2 values used for the objective decision", d_audit < 1e-12, f"{d_audit:.1e}")
check("V4 duration groups come from user_meta.window_workdays (evaluation only, not a predictor)",
      "window_workdays" not in feats and set(dur.dropna().unique()) == {5, 10, 20, 30, 40, 50})
per_fold_dur = folds[folds.role == "valid"].assign(d=lambda t: t.user.map(dur)).dropna().groupby(["repeat", "fold"])["d"].nunique()
check("V4 all 6 durations in every outer-validation fold", (per_fold_dur == 6).all(), per_fold_dur.min())

pd.DataFrame(CHECKS).to_csv(OUT / "checks.csv", index=False)
print(pd.DataFrame(CHECKS).status.value_counts().to_dict())
