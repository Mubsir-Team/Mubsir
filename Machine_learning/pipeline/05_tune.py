"""
05 — Stage B: nested-CV tuning of LightGBM (y_10) and the pre-registered adoption decision.

Implements config/selection_rules.json -> stage_B (hash checked at start):
  * 15 outer folds (config/cv_folds.csv). For each, Optuna (TPE seed 42, 60 trials, defaults as
    trial 0) runs on the OUTER-TRAIN users only, with 5 schedule-grouped inner folds; objective =
    duration-macro row PR-AUC on the pooled inner OOF.
  * Tuned arm: best configuration trained on outer fit users (early stopping on outer es users),
    scoring the outer-validation users. Default arm: LightGBM defaults on the same folds; must
    reproduce the Stage A OOF exactly.
  * User-recall threshold per arm and outer fold: from that arm's inner OOF peaks of the benign
    outer-train users (frozen_inner_user_recall_1pct), then applied to the outer-validation fold.

Usage:
  python 05_tune.py repeat R [--trials N] [--out DIR]   run the 5 outer folds of repeat R (resumable)
  python 05_tune.py adopt [--out DIR]                    aggregate the 15 folds and apply the adoption rule
"""

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

import models
from common import (SEED, LABELS, EVAL_META, FINAL_SPLIT, CV_FOLDS, USER_META, SELECTION_RULES, WORK, RESULTS,
                    REPORTS, feature_list, load_dataset, write_json, md_table)
from metrics import (duration_macro_row_pr_auc, frozen_threshold, detect_with_threshold, row_metrics,
                     DURATIONS)

RULES_SHA = "41de4354e861fb08f2018d76ee1d35fe45cbd3b3b4af3a37a96fba70fdf5deb0"
N_TRIALS = 60
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_TRIAL0 = dict(learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=50, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=1e-3, min_split_gain=0.0)
# reg_alpha: LightGBM default is 0, outside the registered log range [1e-3, 10]; trial 0 uses the lower
# bound 1e-3 (negligible). The DEFAULT ARM uses the exact defaults (models.LGBM_DEFAULTS, reg_alpha 0).


def suggest(trial):
    md = trial.suggest_int("max_depth", -1, 12)
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        max_depth=-1 if md in (0, 1, 2) else md,
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
    )


# ------------------------------------------------------------------ data (dev rows only)
def load():
    sha = hashlib.sha256(SELECTION_RULES.read_bytes()).hexdigest()
    assert sha == RULES_SHA, f"selection_rules.json changed since registration: {sha}"
    feats = feature_list()
    df = load_dataset()
    lab = pd.read_parquet(LABELS)
    meta = pd.read_parquet(EVAL_META)
    split = pd.read_csv(FINAL_SPLIT)
    umeta = pd.read_csv(USER_META).set_index("user")
    dev_users = set(split.loc[split.part == "dev", "user"])
    dev = df["user"].astype(str).isin(dev_users).to_numpy()
    D = dict(
        X=df.loc[dev, feats].astype("float32").reset_index(drop=True),
        y=lab.loc[dev, "y_10"].to_numpy(),
        users=df.loc[dev, "user"].astype(str).to_numpy(),
        in_window=meta.loc[dev, "in_window"].to_numpy(),
        umeta=umeta, folds=pd.read_csv(CV_FOLDS),
    )
    D["row_dur"] = pd.Series(D["users"]).map(umeta["window_workdays"]).to_numpy()
    D["benign_row"] = pd.Series(D["users"]).map(umeta["is_attacker"]).to_numpy() == 0
    D["row_of"] = pd.Series(np.arange(len(D["users"]))).groupby(D["users"]).apply(np.array).to_dict()
    return D


def rows(D, us):
    return np.concatenate([D["row_of"][u] for u in sorted(us)])


def inner_splits(D, outer_train, r, k):
    """5 schedule-grouped inner folds on the outer-train users + inner early-stopping users."""
    tr = D["umeta"].loc[sorted(outer_train), ["stratum", "group"]].reset_index()
    out = []
    sg = StratifiedGroupKFold(5, shuffle=True, random_state=142 + 10 * r + k)
    for i, (a, b) in enumerate(sg.split(tr, tr["stratum"], tr["group"])):
        sub = tr.iloc[a].reset_index(drop=True)
        _, es_i = next(StratifiedGroupKFold(7, shuffle=True, random_state=1142 + 100 * r + 10 * k + i)
                       .split(sub, sub["stratum"], sub["group"]))
        es = set(sub.loc[es_i, "user"])
        fit = set(sub["user"]) - es
        valid = set(tr.iloc[b]["user"])
        g = D["umeta"]["group"]
        assert not (set(g[list(fit)]) & set(g[list(es)])) and not (set(g[list(fit)]) & set(g[list(valid)])) \
            and not (set(g[list(es)]) & set(g[list(valid)])), "inner group overlap"
        out.append((fit, es, valid))
    return out


def inner_oof(D, splits, train_rows, params, seed):
    """Pooled inner OOF over the outer-train rows (aligned with train_rows)."""
    pos = pd.Series(np.arange(len(train_rows)), index=train_rows)
    oof = np.full(len(train_rows), np.nan)
    trees = []
    for fit, es, valid in splits:
        f, e, v = rows(D, fit), rows(D, es), rows(D, valid)
        s, n = models.fit_predict("lightgbm", D["X"].iloc[f], D["y"][f], D["X"].iloc[e], D["y"][e],
                                  D["X"].iloc[v], seed=seed, params=params)
        oof[pos.loc[v].to_numpy()] = s
        trees.append(n)
    assert not np.isnan(oof).any()
    return oof, trees


def objective_value(D, train_rows, oof):
    v, _ = duration_macro_row_pr_auc(D["y"][train_rows], oof, D["row_dur"][train_rows], D["benign_row"][train_rows])
    return v


# ------------------------------------------------------------------ one outer fold
def run_outer(D, r, k, n_trials, out_dir, stageA_oof):
    tag = f"r{r}_k{k}"
    if (out_dir / f"outer_{tag}.json").exists():
        print(f"skip {tag} (done)", flush=True)
        return
    t0 = time.time()
    f = D["folds"][(D["folds"].repeat == r) & (D["folds"].fold == k)]
    fit_u, es_u, val_u = (set(f.loc[f.role == ro, "user"]) for ro in ("fit", "es", "valid"))
    outer_train = fit_u | es_u
    assert not (outer_train & val_u)
    seed = SEED + r
    splits = inner_splits(D, outer_train, r, k)
    train_rows = rows(D, outer_train)
    benign_train = {u for u in outer_train if D["umeta"].loc[u, "is_attacker"] == 0}

    best = {"value": -np.inf}

    def objective(trial):
        params = suggest(trial)
        oof, trees = inner_oof(D, splits, train_rows, params, seed)
        v = objective_value(D, train_rows, oof)
        trial.set_user_attr("inner_trees", trees)
        if v > best["value"]:
            best.update(value=v, oof=oof, params=params, number=trial.number)
        return v

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial(DEFAULT_TRIAL0)
    study.optimize(objective, n_trials=n_trials)
    study.trials_dataframe().to_csv(out_dir / f"trials_{tag}.csv", index=False)
    assert best["number"] == study.best_trial.number

    # default arm: exact defaults; inner OOF only for its frozen threshold
    def_oof, _ = inner_oof(D, splits, train_rows, None, seed)
    fr, er, vr = rows(D, fit_u), rows(D, es_u), rows(D, val_u)
    X, y = D["X"], D["y"]
    s_def, n_def = models.fit_predict("lightgbm", X.iloc[fr], y[fr], X.iloc[er], y[er], X.iloc[vr], seed=seed)
    stageA = stageA_oof[f"rep{r}"].to_numpy()[vr]
    assert np.array_equal(s_def, stageA), "default arm does not reproduce the Stage A OOF"
    s_tun, n_tun = models.fit_predict("lightgbm", X.iloc[fr], y[fr], X.iloc[er], y[er], X.iloc[vr], seed=seed,
                                      params=best["params"])

    att_val = sorted(u for u in val_u if D["umeta"].loc[u, "is_attacker"] == 1)
    ben_val = sorted(set(val_u) - set(att_val))
    res = dict(repeat=r, fold=k, n_trials=n_trials, seconds=round(time.time() - t0, 1),
               best_trial=best["number"], best_params=best["params"], inner_best=best["value"],
               inner_trial0=study.trials[0].value, inner_default_exact=objective_value(D, train_rows, def_oof),
               trees_default=n_def, trees_tuned=n_tun, n_attackers_valid=len(att_val), n_benign_valid=len(ben_val))
    for arm, inner, outer in (("default", def_oof, s_def), ("tuned", best["oof"], s_tun)):
        t, nb = frozen_threshold(D["users"][train_rows], inner, benign_train)
        det, fp = detect_with_threshold(D["users"][vr], outer, D["in_window"][vr], att_val, ben_val, t)
        res[f"{arm}_threshold"] = float(t)
        res[f"{arm}_threshold_benign_users"] = nb
        res[f"{arm}_detected"] = det
        res[f"{arm}_fp_benign_valid"] = fp
        res[f"{arm}_row_pr_auc"] = float(average_precision_score(y[vr], outer))
    pd.DataFrame({"row": vr, "default": s_def, "tuned": s_tun}).to_parquet(out_dir / f"outer_{tag}.parquet", index=False)
    write_json(out_dir / f"outer_{tag}.json", res)
    print(f"{tag} done {res['seconds']}s inner best {res['inner_best']:.4f} (trial {res['best_trial']}) "
          f"trial0 {res['inner_trial0']:.4f} | row AP def {res['default_row_pr_auc']:.4f} tun {res['tuned_row_pr_auc']:.4f} "
          f"| det def {len(res['default_detected'])} tun {len(res['tuned_detected'])} / {len(att_val)} "
          f"fp def {res['default_fp_benign_valid']} tun {res['tuned_fp_benign_valid']}", flush=True)


# ------------------------------------------------------------------ adoption
def adopt(D, out_dir):
    rules = json.loads(SELECTION_RULES.read_text(encoding="utf-8"))["stage_B"]
    res = [json.loads((out_dir / f"outer_r{r}_k{k}.json").read_text()) for r in range(3) for k in range(5)]
    n_att = int((D["umeta"].loc[sorted(set(D["users"])), "is_attacker"] == 1).sum())
    per_rep, per_dur = [], []
    for r in range(3):
        sc = {a: np.full(len(D["y"]), np.nan) for a in ("default", "tuned")}
        for k in range(5):
            p = pd.read_parquet(out_dir / f"outer_r{r}_k{k}.parquet")
            for a in sc:
                sc[a][p["row"].to_numpy()] = p[a].to_numpy()
        row = {"repeat": r}
        for a in sc:
            assert not np.isnan(sc[a]).any()
            m, vals = duration_macro_row_pr_auc(D["y"], sc[a], D["row_dur"], D["benign_row"])
            row[f"{a}_duration_macro"] = m
            det = sum(len(x[f"{a}_detected"]) for x in res if x["repeat"] == r)
            row[f"{a}_user_recall"] = det / n_att
            row[f"{a}_fp_users"] = sum(x[f"{a}_fp_benign_valid"] for x in res if x["repeat"] == r)
            rm = row_metrics(D["y"], sc[a], D["benign_row"])
            row[f"{a}_pooled_row_pr_auc"] = rm["pr_auc"]
            row[f"{a}_recall@1%fpr_row"] = rm["recall@1%fpr"]
            for d, v in zip(DURATIONS, vals):
                per_dur.append(dict(repeat=r, arm=a, duration=d, row_pr_auc=v))
        per_rep.append(row)
    per_rep = pd.DataFrame(per_rep)
    folds_df = pd.DataFrame([{k_: v for k_, v in x.items() if not isinstance(v, (list, dict))} for x in res])
    S = {a: dict(primary=per_rep[f"{a}_duration_macro"].mean(), row_pr_auc=folds_df[f"{a}_row_pr_auc"].mean(),
                 user_recall=per_rep[f"{a}_user_recall"].mean(), fp_users_per_repeat=per_rep[f"{a}_fp_users"].mean())
         for a in ("default", "tuned")}
    cond = [S["tuned"]["primary"] >= S["default"]["primary"] + 0.005,
            S["tuned"]["row_pr_auc"] >= S["default"]["row_pr_auc"] - 0.002,
            S["tuned"]["user_recall"] >= S["default"]["user_recall"]]
    adopted = all(cond)
    decision = dict(adopted=bool(adopted), conditions=dict(zip(rules["tuning_adoption"]["adopt_if"], map(bool, cond))),
                    default=S["default"], tuned=S["tuned"], rules_sha=RULES_SHA)
    write_json(RESULTS / "05_adoption.json", decision)
    per_rep.to_csv(RESULTS / "05_nested_per_repeat.csv", index=False)
    folds_df.to_csv(RESULTS / "05_nested_per_fold.csv", index=False)
    pd.DataFrame([dict(repeat=x["repeat"], fold=x["fold"], **x["best_params"]) for x in res]).to_csv(
        RESULTS / "05_best_params_per_fold.csv", index=False)
    pdur = pd.DataFrame(per_dur).groupby(["duration", "arm"])["row_pr_auc"].mean().unstack()
    pdur.to_csv(RESULTS / "05_per_duration_ap.csv")
    return decision, per_rep, folds_df, pdur


def report(decision, per_rep, folds_df, pdur):
    d, t = decision["default"], decision["tuned"]
    tab = pd.DataFrame({"default": d, "tuned": t}).T
    tab["primary_delta_vs_default"] = tab["primary"] - d["primary"]
    params = pd.read_csv(RESULTS / "05_best_params_per_fold.csv")
    fcols = ["repeat", "fold", "best_trial", "inner_trial0", "inner_default_exact", "inner_best", "trees_default",
             "trees_tuned", "default_fp_benign_valid", "tuned_fp_benign_valid", "n_benign_valid", "seconds"]
    text = f"""# 05 — Stage B nested tuning (LightGBM, y_10)

Rules: `config/selection_rules.json` stage_B (SHA-256 `{RULES_SHA[:16]}…`, verified at start). Dev users only; test untouched.

## Decision: **{'ADOPT tuned procedure' if decision['adopted'] else 'KEEP DEFAULT configuration'}**
| Condition | Holds |
|---|---|
""" + "\n".join(f"| {c} | {'yes' if ok else 'no'} |" for c, ok in decision["conditions"].items()) + f"""

## Nested estimates
{md_table(tab)}

- `primary`: duration-macro row PR-AUC on pooled outer OOF per repeat, mean of 3.
- `row_pr_auc`: 15-fold mean.
- `user_recall`: frozen inner-OOF threshold (1% of benign outer-train users), pooled per repeat, mean of 3.
- `fp_users_per_repeat`: realised false-positive benign users out of 511 per repeat.

## Per repeat
{md_table(per_rep, index=False)}

## Per duration (row PR-AUC, mean of 3 repeats)
{md_table(pdur)}

## Per outer fold
{md_table(folds_df[fcols], index=False)}

`inner_trial0` = trial 0 (defaults with reg_alpha = 1e-3); `inner_default_exact` = exact defaults on the same inner folds.

## Chosen parameters per outer fold (stability)
{md_table(params.describe().T[['mean', 'std', 'min', 'max']])}
"""
    (REPORTS / "05_tune.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["repeat", "adopt"])
    ap.add_argument("repeat", nargs="?", type=int)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--out", type=str, default=str(WORK / "stageB"))
    a = ap.parse_args()
    from pathlib import Path
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = load()
    if a.mode == "repeat":
        stageA = pd.read_parquet(WORK / "oof" / "lightgbm_h10.parquet")
        for k in map(int, a.folds.split(",")):
            run_outer(D, a.repeat, k, a.trials, out, stageA)
    else:
        report(*adopt(D, out))
