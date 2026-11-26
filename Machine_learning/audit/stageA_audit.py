"""
Stage A methodological audit (read-only with respect to the pipeline).

Does not modify selection rules, split, labels, features or training code; it only
reads their outputs (work/oof, work/labels, work/eval_meta, config/*) and the
source files, and writes results/audit_stageA/*.csv. Dev users only; no test
scores exist yet and none are computed here.

Sections
  1  CatBoost vs LightGBM: aggregation facts, paired user-level bootstrap CIs
  2  cumulative_deviation_10wd: independent recomputation, truncation (future-invariance)
     test, pre-injection counterfactual, signal decomposition
  3  Label causality: y_h vs earlier/later attack days, off-day gaps, latency distributions
  4  Recall@1%FPR resolution: FP counts per repeat, per-fold vs pooled thresholds
  5  Row weight of long campaigns (input to the Stage B objective discussion)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from common import (HORIZONS, LABELS, EVAL_META, FINAL_SPLIT, CV_FOLDS, USER_META, WORK, RESULTS, ROOT, NAS,
                    feature_list, load_dataset)  # noqa: E402
from metrics import threshold_at_fpr  # noqa: E402

OUT = RESULTS / "audit_stageA"
OUT.mkdir(exist_ok=True)
B = 2000
RNG = np.random.default_rng(42)

df = load_dataset()
lab = pd.read_parquet(LABELS)
meta = pd.read_parquet(EVAL_META)
split = pd.read_csv(FINAL_SPLIT)
umeta = pd.read_csv(USER_META)
folds = pd.read_csv(CV_FOLDS)
dev_users = set(split.loc[split.part == "dev", "user"])
dev = df["user"].astype(str).isin(dev_users).to_numpy()
M = meta.loc[dev].reset_index(drop=True)
M["user"] = M["user"].astype(str)
L = lab.loc[dev].reset_index(drop=True)


# ============================================================ 1. CatBoost vs LightGBM
def user_table(scores):
    """Per user: peak over the whole period and max inside the attack window."""
    f = pd.DataFrame({"user": M["user"], "score": scores, "in_window": M["in_window"]})
    peak = f.groupby("user")["score"].max()
    inwin = f[f["in_window"]].groupby("user")["score"].max()
    t = pd.DataFrame({"peak": peak}).join(inwin.rename("inwin"))
    return t


users = umeta.set_index("user").loc[sorted(dev_users), ["is_attacker", "group"]]
att_idx = np.where(users["is_attacker"].to_numpy() == 1)[0]
ben_idx = np.where(users["is_attacker"].to_numpy() == 0)[0]


def auc_fast(pos, neg):
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank(method="average").to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def user_stats(peak, inwin, a_idx, b_idx, fpr=0.01):
    t = threshold_at_fpr(peak[b_idx], fpr)
    return auc_fast(peak[a_idx], peak[b_idx]), float(np.mean(inwin[a_idx] > t)), int(np.sum(peak[b_idx] > t))


boot_rows, point_rows, fp_rows = [], [], []
group_codes = users["group"].to_numpy()
att_groups = pd.Series(att_idx).groupby(group_codes[att_idx]).apply(list).tolist()
for h in HORIZONS:
    tabs = {}
    for algo in ("lightgbm", "catboost"):
        oof = pd.read_parquet(WORK / "oof" / f"{algo}_h{h}.parquet")
        tabs[algo] = [user_table(oof[c].to_numpy()).loc[users.index] for c in oof.columns]
    P = {a: np.stack([t["peak"].to_numpy() for t in tabs[a]]) for a in tabs}
    W = {a: np.stack([t["inwin"].fillna(-np.inf).to_numpy() for t in tabs[a]]) for a in tabs}
    for rep in range(P["lightgbm"].shape[0]):
        for a in tabs:
            for f in (0.01, 0.05):
                auc, rec, fp = user_stats(P[a][rep], W[a][rep], att_idx, ben_idx, f)
                point_rows.append(dict(h=h, algo=a, repeat=rep, fpr_target=f, user_auc=auc, user_recall=rec,
                                       fp_users=fp, n_benign=len(ben_idx), achieved_fpr=fp / len(ben_idx)))
    for scheme in ("user", "attacker_group"):
        diffs = np.empty((B, 2))
        for b in range(B):
            if scheme == "user":
                a_s = RNG.choice(att_idx, len(att_idx), replace=True)
            else:
                pick = RNG.integers(0, len(att_groups), len(att_groups))
                a_s = np.concatenate([att_groups[i] for i in pick])
            b_s = RNG.choice(ben_idx, len(ben_idx), replace=True)
            d_auc, d_rec = [], []
            for rep in range(P["lightgbm"].shape[0]):
                s_l = user_stats(P["lightgbm"][rep], W["lightgbm"][rep], a_s, b_s)
                s_c = user_stats(P["catboost"][rep], W["catboost"][rep], a_s, b_s)
                d_auc.append(s_c[0] - s_l[0])
                d_rec.append(s_c[1] - s_l[1])
            diffs[b] = [np.mean(d_auc), np.mean(d_rec)]
        pt = pd.DataFrame(point_rows)
        pt = pt[(pt.h == h) & (pt.fpr_target == 0.01)].groupby("algo")[["user_auc", "user_recall"]].mean()
        for k, name in enumerate(("user_auc", "user_recall@1%ufpr")):
            col = "user_auc" if k == 0 else "user_recall"
            boot_rows.append(dict(h=h, resample=scheme, metric=name,
                                  lightgbm=pt.loc["lightgbm", col], catboost=pt.loc["catboost", col],
                                  diff_cat_minus_lgbm=pt.loc["catboost", col] - pt.loc["lightgbm", col],
                                  ci_low=np.percentile(diffs[:, k], 2.5), ci_high=np.percentile(diffs[:, k], 97.5),
                                  share_boot_diff_gt0=float(np.mean(diffs[:, k] > 0))))
    print(f"bootstrap h={h} done", flush=True)
pd.DataFrame(boot_rows).to_csv(OUT / "1_bootstrap_cat_vs_lgbm.csv", index=False)
pd.DataFrame(point_rows).to_csv(OUT / "1_user_points_fp_counts.csv", index=False)

# fold-level paired row metrics
fr = pd.read_csv(RESULTS / "04_cv_folds.csv")
pair = fr[fr.algo.isin(["lightgbm", "catboost"])].pivot_table(
    index=["h", "repeat", "fold"], columns="algo", values=["recall@1%fpr", "pr_auc"])
wins = []
for h in HORIZONS:
    p = pair.loc[h]
    for m in ("recall@1%fpr", "pr_auc"):
        d = p[(m, "catboost")] - p[(m, "lightgbm")]
        wins.append(dict(h=h, metric=m, mean_diff=d.mean(), sd_diff=d.std(), catboost_wins=int((d > 0).sum()),
                         folds=len(d)))
pd.DataFrame(wins).to_csv(OUT / "1_fold_paired_row_metrics.csv", index=False)

# ============================================================ 2. cumulative_deviation_10wd
CH = [("external_email_count", "email_count_p50", "email_count_p95"),
      ("usb_copy_count", "usb_copy_p50", "usb_copy_p95"),
      ("www_upload_count", "http_upload_p50", "http_upload_p95")]
SRC_FEAT = NAS / "12_Feature_Engineering_and_Validation" / "behavior_features_r62_ml_final_new.parquet"
PRE_FEAT = ROOT / "Data_engineering" / "data" / "features" / "behavior_features_r62_v1.parquet"
cols = ["user", "day"] + [c for t in CH for c in t]


def daily_E(frame):
    """NB12 rule: channel valid iff P50, P95 finite, P50 > 0, P95 > 0, P95 >= P50; else E = 0."""
    out = np.zeros(len(frame))
    for x, p50, p95 in CH:
        v = frame[p50].notna() & frame[p95].notna() & (frame[p50] > 0) & (frame[p95] > 0) & (frame[p95] >= frame[p50])
        e = np.maximum(0.0, frame[x] - frame[p50]) / np.maximum(1.0, frame[p95] - frame[p50])
        out += np.where(v, e.fillna(0.0), 0.0)
    return out


def trailing_loop(frame, values, w=10):
    """Independent explicit loop: sum of values over rows [i-w+1, i] of the same user."""
    res = np.empty(len(frame))
    for _, idx in frame.groupby("user", sort=False).indices.items():
        v = values[idx]
        res[idx] = [v[max(0, i - w + 1): i + 1].sum() for i in range(len(v))]
    return res


inj = pd.read_parquet(SRC_FEAT, columns=cols + ["cumulative_deviation_10wd", "is_weekend", "logon_count"])
inj = inj[inj["user"].isin(set(df["user"].astype(str)))].sort_values(["user", "day"]).reset_index(drop=True)
pre = pd.read_parquet(PRE_FEAT, columns=cols)
pre = pre[pre["user"].isin(set(df["user"].astype(str)))].sort_values(["user", "day"]).reset_index(drop=True)
assert (inj["user"].astype(str).to_numpy() == df["user"].astype(str).to_numpy()).all()
assert (inj["day"].to_numpy() == df["day"].to_numpy()).all()
pre_keys_ok = (pre["user"].astype(str).to_numpy() == inj["user"].astype(str).to_numpy()).all() and \
              (pre["day"].astype("datetime64[ms]").to_numpy() == inj["day"].to_numpy()).all()

E_inj = daily_E(inj)
cd_loop = trailing_loop(inj, E_inj)
recompute_diff = float(np.abs(cd_loop - inj["cumulative_deviation_10wd"].to_numpy()).max())
model_col_diff = float(np.abs(df["cumulative_deviation_10wd"].to_numpy() - inj["cumulative_deviation_10wd"].to_numpy()).max())
baseline_constant = all(inj.groupby("user")[c].nunique(dropna=False).max() <= 1 for _, a, b in CH for c in (a, b))
baseline_same_pre = all(np.allclose(inj[c].fillna(-1), pre[c].fillna(-1)) for _, a, b in CH for c in (a, b))

# truncation test: recompute on prefixes of 60 random users cut at a random day; values must not change
trunc_mismatch = 0
for u in RNG.choice(inj["user"].unique(), 60, replace=False):
    g = inj[inj["user"] == u]
    cut = RNG.integers(5, len(g))
    gg = g.iloc[:cut].reset_index(drop=True)
    v = trailing_loop(gg, daily_E(gg))
    trunc_mismatch += int((np.abs(v - g["cumulative_deviation_10wd"].to_numpy()[:cut]) > 1e-9).sum())

# counterfactual: same rows without the injected events
E_pre = daily_E(pre)
cd_pre = trailing_loop(pre, E_pre)
delta = inj["cumulative_deviation_10wd"].to_numpy() - cd_pre
inj_day = (lab["y"].to_numpy() == 1)
inj_in_last10 = trailing_loop(inj, inj_day.astype(float)) > 0
changed = np.abs(delta) > 1e-9
dE = E_inj - E_pre
first_attack_row = pd.Series(np.where(inj_day, np.arange(len(inj)), np.nan)).groupby(inj["user"].to_numpy()).transform("min")
before_first = np.arange(len(inj)) < first_attack_row.fillna(np.inf).to_numpy()

y10 = lab["y_10"].to_numpy()
day_type = meta["day_type"].to_numpy()
is_att_user = meta["is_attacker"].to_numpy() == 1
d = dev
cd_rows = [
    ("Recomputed (explicit loop) vs source column, max abs diff", recompute_diff),
    ("Model dataset column vs source column, max abs diff", model_col_diff),
    ("Baseline P50/P95 constant within every user", baseline_constant),
    ("Baseline P50/P95 identical before and after injection", baseline_same_pre),
    ("Pre-injection file aligned with injected file (user, day)", pre_keys_ok),
    ("Truncation test: values changed when future rows removed (60 users)", trunc_mismatch),
    ("Rows where injection changed the feature", int(changed.sum())),
    ("  ...of which have NO injected day in the trailing 10 rows", int((changed & ~inj_in_last10).sum())),
    ("Rows before the attacker's first injected day with any change", int((changed & before_first).sum())),
    ("Benign-user rows with any change", int((changed & ~is_att_user).sum())),
    ("Max |delta - trailing sum of daily dE| (delta explained by injected days only)",
     float(np.abs(delta - trailing_loop(inj, dE)).max())),
    ("Days where daily E changed but no injection (y=0)", int(((np.abs(dE) > 1e-9) & ~inj_day).sum())),
    ("AUC vs y_10 on dev rows, injected feature", roc_auc_score(y10[d], inj["cumulative_deviation_10wd"].to_numpy()[d])),
    ("AUC vs y_10 on dev rows, pre-injection counterfactual (same rows/labels)", roc_auc_score(y10[d], cd_pre[d])),
    ("AUC vs y_10, attackers' rows only, injected", roc_auc_score(y10[d & is_att_user], inj["cumulative_deviation_10wd"].to_numpy()[d & is_att_user])),
    ("AUC vs y_10, attackers' rows only, counterfactual", roc_auc_score(y10[d & is_att_user], cd_pre[d & is_att_user])),
]
pd.DataFrame(cd_rows, columns=["check", "value"]).to_csv(OUT / "2_cumulative_deviation_checks.csv", index=False)

# share of the feature value on positive rows that comes from injected events, by stealth
st = umeta.set_index("user")["stealth_level"]
pos = d & (y10 == 1)
share = pd.DataFrame({"stealth": inj.loc[pos, "user"].map(st).to_numpy(), "inj": inj.loc[pos, "cumulative_deviation_10wd"].to_numpy(),
                      "pre": cd_pre[pos]})
by_st = share.groupby("stealth").agg(rows=("inj", "size"), median_injected=("inj", "median"), median_counterfactual=("pre", "median"),
                                     mean_share_from_injection=("inj", lambda s: np.nan))
by_st["mean_share_from_injection"] = share.groupby("stealth").apply(
    lambda g: float(((g["inj"] - g["pre"]).sum()) / g["inj"].sum()))
auc_st = []
for s in ("High", "Medium", "Low"):
    m = d & ((~is_att_user) | (meta["user"].astype(str).map(st).to_numpy() == s))
    auc_st.append((s, roc_auc_score(y10[m], inj["cumulative_deviation_10wd"].to_numpy()[m]), roc_auc_score(y10[m], cd_pre[m])))
by_st = by_st.join(pd.DataFrame(auc_st, columns=["stealth", "auc_injected", "auc_counterfactual"]).set_index("stealth"))
by_st.to_csv(OUT / "2_cumulative_deviation_by_stealth.csv")

# injected E per attack day (simulator-feature coupling)
attack_dE = pd.DataFrame({"stealth": meta.loc[inj_day, "user"].astype(str).map(st).to_numpy(), "dE": dE[inj_day]})
attack_dE.groupby("stealth")["dE"].describe(percentiles=[.1, .25, .5, .75, .9]).to_csv(OUT / "2_injected_dE_per_attack_day.csv")

# ============================================================ 3. Labels and latency
y = lab["y"].to_numpy()
viol = 0
for h in HORIZONS:
    yh = lab[f"y_{h}"].to_numpy()
    nxt = trailing_loop(inj, y.astype(float), w=h) > 0          # attack day in [t-h+1, t]
    viol += int(((yh == 1) & ~nxt).sum()) + int(((yh == 0) & nxt).sum())
# would a FORWARD-looking label differ? (shows the label is not forward-looking)
fwd = np.zeros(len(y), dtype=bool)
for _, idx in inj.groupby("user", sort=False).indices.items():
    v = y[idx]
    fwd[idx] = [v[i: i + 10].max() > 0 for i in range(len(v))]
pre_pos = int(((lab["y_10"].to_numpy() == 1) & before_first).sum())

off = meta["day_type"].to_numpy() == "offday"
pos_idx = np.arange(len(y))
last_attack = pd.Series(np.where(y == 1, pos_idx, np.nan)).groupby(inj["user"].to_numpy()).ffill().to_numpy()
gap = pos_idx - last_attack                                    # workdays since last attack day (NaN = none yet)
off_gap = pd.Series(gap[off])
lab_rows = [
    ("y_h(t)=1 without an attack day in [t-h+1, t], or vice versa (all h)", viol),
    ("Rows with y_10=1 before the attacker's first injected day", pre_pos),
    ("Rows where a forward-looking 10-day label would differ from y_10", int((fwd != (lab['y_10'].to_numpy() == 1)).sum())),
    ("Off-day rows", int(off.sum())),
    ("Off-days with an earlier attack day in the same window", int(off_gap.notna().sum())),
    ("Off-days before the first attack day (leading off-days)", int(off_gap.isna().sum())),
    ("Max workdays from last attack day to an off-day", float(off_gap.max())),
    ("Off-days positive under y_5 / y_10 / y_20",
     f"{int(lab.loc[off, 'y_5'].sum())} / {int(lab.loc[off, 'y_10'].sum())} / {int(lab.loc[off, 'y_20'].sum())}"),
    ("Rows with wd_from_first_attack == 0 that are injected days", f"{int((meta['wd_from_first_attack'] == 0).sum())} rows, all y=1: "
                                                                     f"{bool((meta.loc[meta['wd_from_first_attack'] == 0, 'y'] == 1).all())}"),
]
pd.DataFrame(lab_rows, columns=["check", "value"]).to_csv(OUT / "3_label_checks.csv", index=False)
off_gap.value_counts(dropna=False).sort_index().rename("offdays").to_csv(OUT / "3_offday_gap_distribution.csv")

lat_rows = []
for h in HORIZONS:
    for algo in ("lightgbm", "catboost"):
        oof = pd.read_parquet(WORK / "oof" / f"{algo}_h{h}.parquet")
        for f in (0.01, 0.05):
            firsts, det = [], []
            for c in oof.columns:
                fr_ = pd.DataFrame({"user": M["user"], "score": oof[c].to_numpy(), "in_window": M["in_window"],
                                    "wd": M["wd_from_first_attack"], "is_att": M["is_attacker"]})
                bp = fr_[fr_.is_att == 0].groupby("user")["score"].max().to_numpy()
                t = threshold_at_fpr(bp, f)
                fa = fr_[fr_.in_window & (fr_.score > t)].groupby("user")["wd"].min().astype(float)
                firsts.append(fa)
                det.append(len(fa))
            allf = pd.concat(firsts)
            lat_rows.append(dict(h=h, algo=algo, fpr_target=f, detected_mean=np.mean(det), n_attackers=len(att_idx),
                                 p25=allf.quantile(.25), median=allf.median(), p75=allf.quantile(.75), p90=allf.quantile(.90),
                                 max=allf.max(), share_day0=float((allf == 0).mean()), share_le2=float((allf <= 2).mean()),
                                 alarms_on_leading_offday=int((allf < 0).sum())))
pd.DataFrame(lat_rows).to_csv(OUT / "3_latency_distribution.csv", index=False)

# latency by duration for the selected model (lightgbm, h=10, 1% user FPR, pooled repeats)
oof = pd.read_parquet(WORK / "oof" / "lightgbm_h10.parquet")
dur = umeta.set_index("user")["window_workdays"]
rows = []
for c in oof.columns:
    fr_ = pd.DataFrame({"user": M["user"], "score": oof[c].to_numpy(), "in_window": M["in_window"],
                        "wd": M["wd_from_first_attack"], "is_att": M["is_attacker"]})
    t = threshold_at_fpr(fr_[fr_.is_att == 0].groupby("user")["score"].max().to_numpy(), 0.01)
    fa = fr_[fr_.in_window & (fr_.score > t)].groupby("user")["wd"].min().astype(float)
    for u in users.index[att_idx]:
        rows.append(dict(repeat=c, user=u, duration=dur[u], detected=u in fa.index, latency=fa.get(u, np.nan)))
lr = pd.DataFrame(rows)
lr.groupby("duration").agg(detected_share=("detected", "mean"), p25=("latency", lambda s: s.quantile(.25)),
                           median=("latency", "median"), p75=("latency", lambda s: s.quantile(.75)),
                           p90=("latency", lambda s: s.quantile(.9))).to_csv(OUT / "3_latency_by_duration_lgbm_h10.csv")

# ============================================================ 4. Recall@1%FPR resolution
res_rows = []
for (rep, fold), f in folds[folds.role == "valid"].groupby(["repeat", "fold"]):
    nb = int((users.loc[f["user"], "is_attacker"] == 0).sum())
    rows_b = int(((M["is_attacker"] == 0) & M["user"].isin(set(f["user"]))).sum())
    res_rows.append(dict(repeat=rep, fold=fold, benign_users=nb, k_user_fp_at_1pct=int(np.floor(0.01 * nb)),
                         benign_rows=rows_b, k_row_fp_at_1pct=int(np.floor(0.01 * rows_b))))
pd.DataFrame(res_rows).to_csv(OUT / "4_fold_resolution.csv", index=False)

# per-fold user thresholds (alternative) vs pooled, lightgbm/catboost h=10
alt = []
for algo in ("lightgbm", "catboost"):
    oof = pd.read_parquet(WORK / "oof" / f"{algo}_h10.parquet")
    for rep, c in enumerate(oof.columns):
        ut = user_table(oof[c].to_numpy()).loc[users.index]
        det, fp = 0, 0
        for fold, f in folds[(folds.role == "valid") & (folds.repeat == rep)].groupby("fold"):
            sub = ut.loc[f["user"]]
            ia = users.loc[f["user"], "is_attacker"].to_numpy() == 1
            t = threshold_at_fpr(sub["peak"].to_numpy()[~ia], 0.01)
            det += int((sub["inwin"].to_numpy()[ia] > t).sum())
            fp += int((sub["peak"].to_numpy()[~ia] > t).sum())
        alt.append(dict(algo=algo, repeat=rep, threshold="per_fold", user_recall=det / len(att_idx), fp_users=fp))
pd.DataFrame(alt).to_csv(OUT / "4_per_fold_vs_pooled_threshold_h10.csv", index=False)

# ============================================================ 5. row weight per attacker (objective discussion)
w = pd.DataFrame({"user": M["user"], "y10": L["y_10"]}).groupby("user")["y10"].sum()
w = w[users.index[att_idx]].rename("positive_rows_y10").to_frame().join(dur)
w.groupby("window_workdays")["positive_rows_y10"].agg(["count", "mean", "sum"]).assign(
    share_of_all_positive_rows=lambda t: t["sum"] / t["sum"].sum()).to_csv(OUT / "5_positive_rows_by_duration.csv")
print("audit done")
