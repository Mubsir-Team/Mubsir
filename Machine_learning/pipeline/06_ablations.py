"""
06 — Stage B ablations (LightGBM defaults, y_10), judged by the pre-registered rule
(config/selection_rules.json -> stage_B.ablation_adoption = the three tuning_adoption conditions,
comparing each variant with the adopted Stage B configuration = LightGBM defaults, whose nested
results are reused from 05).

Variants (fixed before running):
  no_nr     drop the 24 rotation features nr_*                         (76 predictors)
  spw       scale_pos_weight = negatives / positives of each training set (100 predictors)
  c1_doc    add the 11 non-static C1 companion columns                  (111 predictors)
            DOCUMENTATION ONLY: measures the injection artefact; never adopted. The 4 static
            email_bytes_p50/p90/p95/mad baseline columns are left out (they are per-user
            identity columns, a C2 issue, and would mix two effects).

Each variant runs on the same 15 outer folds, seeds, fit/es users and inner folds as 05, so the
frozen inner-OOF user-recall threshold is derived exactly as for the default arm.

Usage:  python 06_ablations.py run VARIANT     (resumable per outer fold)
        python 06_ablations.py report
"""

import argparse
import importlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import models
from common import NAS, RESULTS, REPORTS, WORK, SEED, ALLOWLIST, write_json, md_table, load_dataset
from metrics import duration_macro_row_pr_auc, frozen_threshold, detect_with_threshold, row_metrics, DURATIONS

T = importlib.import_module("05_tune")   # same data loading, rules-hash check, inner splits
OUT = WORK / "stageB" / "ablations"
VARIANTS = ("no_nr", "spw", "c1_doc")
STATIC_C1 = {"email_bytes_p50", "email_bytes_p90", "email_bytes_p95", "email_bytes_mad"}


def variant_X(D, variant):
    X = D["X"]
    if variant == "no_nr":
        return X[[c for c in X.columns if not c.startswith("nr_")]]
    if variant == "c1_doc":
        allow = pd.read_csv(ALLOWLIST)
        c1 = [c for c in allow.loc[allow.decision == "exclude_C1", "feature_name"] if c not in STATIC_C1]
        src = pd.read_parquet(NAS / "12_Feature_Engineering_and_Validation" / "behavior_features_r62_ml_final_new.parquet",
                              columns=["user", "day"] + c1)
        src["user"] = src["user"].astype(str)
        keys = load_dataset()[["user", "day"]]
        keys["user"] = keys["user"].astype(str)
        keys = keys[keys["user"].isin(set(D["users"]))].reset_index(drop=True)
        m = keys.merge(src, on=["user", "day"], how="left", validate="one_to_one")
        assert (m["user"].to_numpy() == D["users"]).all() and m[c1].notna().all().all()
        return pd.concat([X, m[c1].astype("float32")], axis=1)
    return X


def fit_predict(variant, Xf, yf, Xe, ye, Xv, seed):
    params = None
    if variant == "spw":
        params = {"scale_pos_weight": float((yf == 0).sum() / max(1, (yf == 1).sum()))}
    return models.fit_predict("lightgbm", Xf, yf, Xe, ye, Xv, seed=seed, params=params)


def run_outer(D, X, variant, r, k):
    path = OUT / f"{variant}_r{r}_k{k}.json"
    if path.exists():
        return
    t0 = time.time()
    f = D["folds"][(D["folds"].repeat == r) & (D["folds"].fold == k)]
    fit_u, es_u, val_u = (set(f.loc[f.role == ro, "user"]) for ro in ("fit", "es", "valid"))
    outer_train = fit_u | es_u
    seed = SEED + r
    y = D["y"]
    # inner OOF on outer-train users only -> frozen threshold
    train_rows = T.rows(D, outer_train)
    pos = pd.Series(np.arange(len(train_rows)), index=train_rows)
    inner = np.full(len(train_rows), np.nan)
    for fi, ei, vi in T.inner_splits(D, outer_train, r, k):
        a, b, c = T.rows(D, fi), T.rows(D, ei), T.rows(D, vi)
        s, _ = fit_predict(variant, X.iloc[a], y[a], X.iloc[b], y[b], X.iloc[c], seed)
        inner[pos.loc[c].to_numpy()] = s
    assert not np.isnan(inner).any()
    benign_train = {u for u in outer_train if D["umeta"].loc[u, "is_attacker"] == 0}
    t, nb = frozen_threshold(D["users"][train_rows], inner, benign_train)
    # outer model
    fr, er, vr = T.rows(D, fit_u), T.rows(D, es_u), T.rows(D, val_u)
    s, n = fit_predict(variant, X.iloc[fr], y[fr], X.iloc[er], y[er], X.iloc[vr], seed)
    att = sorted(u for u in val_u if D["umeta"].loc[u, "is_attacker"] == 1)
    ben = sorted(set(val_u) - set(att))
    det, fp = detect_with_threshold(D["users"][vr], s, D["in_window"][vr], att, ben, t)
    pd.DataFrame({"row": vr, "score": s}).to_parquet(OUT / f"{variant}_r{r}_k{k}.parquet", index=False)
    write_json(path, dict(variant=variant, repeat=r, fold=k, n_features=X.shape[1], trees=n, threshold=float(t),
                          threshold_benign_users=nb, detected=det, fp_benign_valid=fp,
                          row_pr_auc=float(average_precision_score(y[vr], s)), seconds=round(time.time() - t0, 1)))
    print(f"{variant} r{r}k{k} AP {average_precision_score(y[vr], s):.4f} det {len(det)}/{len(att)} fp {fp} "
          f"trees {n} {time.time() - t0:.0f}s", flush=True)


def summarise(D, variant):
    """Same aggregation as 05.adopt: pooled per repeat (primary, user recall), 15-fold mean (row PR-AUC)."""
    n_att = int((D["umeta"].loc[sorted(set(D["users"])), "is_attacker"] == 1).sum())
    reps, durs, folds = [], [], []
    for r in range(3):
        sc = np.full(len(D["y"]), np.nan)
        det = fp = 0
        for k in range(5):
            if variant == "default":
                p = pd.read_parquet(WORK / "stageB" / f"outer_r{r}_k{k}.parquet").rename(columns={"default": "score"})
                j = json.loads((WORK / "stageB" / f"outer_r{r}_k{k}.json").read_text())
                det += len(j["default_detected"]); fp += j["default_fp_benign_valid"]
                folds.append(j["default_row_pr_auc"])
            else:
                p = pd.read_parquet(OUT / f"{variant}_r{r}_k{k}.parquet")
                j = json.loads((OUT / f"{variant}_r{r}_k{k}.json").read_text())
                det += len(j["detected"]); fp += j["fp_benign_valid"]
                folds.append(j["row_pr_auc"])
            sc[p["row"].to_numpy()] = p["score"].to_numpy()
        assert not np.isnan(sc).any()
        m, vals = duration_macro_row_pr_auc(D["y"], sc, D["row_dur"], D["benign_row"])
        rm = row_metrics(D["y"], sc, D["benign_row"])
        reps.append(dict(primary=m, user_recall=det / n_att, fp_users=fp, recall_1pct_row=rm["recall@1%fpr"]))
        durs.append(vals)
    reps = pd.DataFrame(reps)
    out = dict(variant=variant, primary=reps.primary.mean(), row_pr_auc=float(np.mean(folds)),
               user_recall=reps.user_recall.mean(), fp_users_per_repeat=reps.fp_users.mean(),
               row_recall_1pct=reps.recall_1pct_row.mean())
    for d, v in zip(DURATIONS, np.mean(durs, axis=0)):
        out[f"ap_d{d}"] = v
    return out


def report(D):
    base = summarise(D, "default")
    rows = [base]
    for v in VARIANTS:
        s = summarise(D, v)
        cond = [s["primary"] >= base["primary"] + 0.005, s["row_pr_auc"] >= base["row_pr_auc"] - 0.002,
                s["user_recall"] >= base["user_recall"]]
        s["conditions_met"] = f"{sum(cond)}/3 ({'/'.join('y' if c else 'n' for c in cond)})"
        s["decision"] = "documentation only" if v == "c1_doc" else ("ADOPT" if all(cond) else "not adopted")
        rows.append(s)
    tab = pd.DataFrame(rows).set_index("variant")
    for c in ("primary", "row_pr_auc", "user_recall"):
        tab[f"d_{c}"] = tab[c] - base[c]
    tab.to_csv(RESULTS / "06_ablations.csv")
    write_json(RESULTS / "06_ablations.json", {k: {kk: (vv if not isinstance(vv, float) else round(vv, 6))
                                                    for kk, vv in r.items()} for k, r in tab.to_dict("index").items()})
    main = ["primary", "d_primary", "row_pr_auc", "d_row_pr_auc", "user_recall", "d_user_recall",
            "fp_users_per_repeat", "row_recall_1pct", "conditions_met", "decision"]
    text = f"""# 06 — Stage B ablations (LightGBM defaults, y_10, nested 15 outer folds)

Rule: `stage_B.ablation_adoption` — each variant vs the adopted configuration (LightGBM defaults, nested results from 05):
primary (duration-macro row PR-AUC) >= +0.005, row PR-AUC >= −0.002, frozen inner-OOF user recall >= default.
Dev users only; test untouched.

{md_table(tab[main].fillna(''))}

Conditions order: primary / row PR-AUC / user recall.

## Row PR-AUC per attack duration (mean of 3 repeats)
{md_table(tab[[f'ap_d{d}' for d in DURATIONS]])}

- `no_nr`: drops the 24 rotation features (`nr_*`).
- `spw`: scale_pos_weight = negatives/positives of each training set (no other change).
- `c1_doc`: adds 11 non-static C1 companion columns that the injection did not update; documents the size of the
  simulation artefact, never adopted.
"""
    (REPORTS / "06_ablations.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("variant", nargs="?", choices=VARIANTS)
    ap.add_argument("--folds", default=None, help="r:k list, e.g. 0:0,0:1 (default: all 15)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    D = T.load()
    if a.mode == "run":
        X = variant_X(D, a.variant)
        todo = [tuple(map(int, x.split(":"))) for x in a.folds.split(",")] if a.folds else \
            [(r, k) for r in range(3) for k in range(5)]
        for r, k in todo:
            run_outer(D, X, a.variant, r, k)
    else:
        report(D)
