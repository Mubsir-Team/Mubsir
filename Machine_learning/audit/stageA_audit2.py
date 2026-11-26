"""
Stage A audit, part 2 (dev OOF only; read-only with respect to the pipeline).

  1  Missed 5-workday attackers (LightGBM, y_10, 1% user-FPR threshold as in Stage A)
  2  Candidate optimisation objectives compared on dev OOF (all 5 algorithms, y_10)
  3  Observation length and the user-level maximum score
  4  Latency with (a) the Stage A pooled threshold and (b) a frozen leave-fold-out threshold,
     by duration and stealth

Outputs: results/audit_stageA2/*.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from common import (LABELS, EVAL_META, FINAL_SPLIT, CV_FOLDS, USER_META, WORK, RESULTS, ROOT, NAS,
                    load_dataset)  # noqa: E402
from metrics import threshold_at_fpr  # noqa: E402
from models import ALGOS  # noqa: E402

OUT = RESULTS / "audit_stageA2"
OUT.mkdir(exist_ok=True)

df = load_dataset()
lab = pd.read_parquet(LABELS)
meta = pd.read_parquet(EVAL_META)
split = pd.read_csv(FINAL_SPLIT)
umeta = pd.read_csv(USER_META).set_index("user")
folds = pd.read_csv(CV_FOLDS)
dev_users = sorted(split.loc[split.part == "dev", "user"])
dev = df["user"].astype(str).isin(set(dev_users)).to_numpy()
D = df.loc[dev].reset_index(drop=True)
D["user"] = D["user"].astype(str)
M = meta.loc[dev].reset_index(drop=True)
M["user"] = M["user"].astype(str)
L = lab.loc[dev].reset_index(drop=True)
U = umeta.loc[dev_users]
is_att = U["is_attacker"].to_numpy() == 1
valid_of = {rep: folds[(folds.role == "valid") & (folds.repeat == rep)].set_index("user")["fold"] for rep in range(3)}


def oof(algo, h=10):
    return pd.read_parquet(WORK / "oof" / f"{algo}_h{h}.parquet")


def user_scores(score):
    f = pd.DataFrame({"user": M["user"], "s": score, "w": M["in_window"]})
    peak = f.groupby("user")["s"].max().loc[dev_users]
    inwin = f[f["w"]].groupby("user")["s"].max().reindex(dev_users)
    return peak, inwin


# ============================================================ 1. missed 5-day attackers
lg = oof("lightgbm")
det = {}
thr = {}
for rep, c in enumerate(lg.columns):
    peak, inwin = user_scores(lg[c].to_numpy())
    t = threshold_at_fpr(peak[~is_att].to_numpy(), 0.01)
    thr[rep] = t
    det[rep] = (inwin > t) & pd.Series(is_att, index=dev_users)
det = pd.DataFrame(det)
att5 = U.index[(U["is_attacker"] == 1) & (U["window_workdays"] == 5)]
d5 = det.loc[att5]
consistency = pd.DataFrame({
    "repeat": [0, 1, 2],
    "detected_5day": [int(d5[r].sum()) for r in range(3)],
    "n_5day": len(att5),
    "rate": [d5[r].mean() for r in range(3)],
    "threshold": [thr[r] for r in range(3)],
})
consistency.to_csv(OUT / "1_5day_detection_by_repeat.csv", index=False)
n_det = d5.sum(axis=1)

# per-attacker profile (evaluation-only columns are used here; none is a predictor)
gt_u = pd.read_parquet(NAS / "11_Ground_Truth_and_Quality_Validation" / "ground_truth_user_days_r62.parquet",
                       columns=["user", "day", "injected_channels", "executed_channel_count", "sync_type",
                                "total_injected_events", "is_switch", "primary_channel"])
gt_c = pd.read_parquet(NAS / "11_Ground_Truth_and_Quality_Validation" / "ground_truth_channel_days_r62.parquet",
                       columns=["user", "day", "channel", "baseline_p95", "count_before", "count_added"])
gt_u["user"] = gt_u["user"].astype(str)
gt_c["user"] = gt_c["user"].astype(str)

# injected deviation: E with vs without injection (NB12 E rule), from the source tables
CH = [("external_email_count", "email_count_p50", "email_count_p95"),
      ("usb_copy_count", "usb_copy_p50", "usb_copy_p95"),
      ("www_upload_count", "http_upload_p50", "http_upload_p95")]
cols = ["user", "day"] + [c for t in CH for c in t]
inj = pd.read_parquet(NAS / "12_Feature_Engineering_and_Validation" / "behavior_features_r62_ml_final_new.parquet", columns=cols)
pre = pd.read_parquet(ROOT / "Data_engineering" / "data" / "features" / "behavior_features_r62_v1.parquet", columns=cols)


def E(frame):
    out = np.zeros(len(frame))
    for x, p50, p95 in CH:
        v = frame[p50].notna() & frame[p95].notna() & (frame[p50] > 0) & (frame[p95] > 0) & (frame[p95] >= frame[p50])
        out += np.where(v, (np.maximum(0.0, frame[x] - frame[p50]) / np.maximum(1.0, frame[p95] - frame[p50])).fillna(0), 0)
    return out


for fr in (inj, pre):
    fr["user"] = fr["user"].astype(str)
    fr.sort_values(["user", "day"], inplace=True)
    fr.reset_index(drop=True, inplace=True)
inj["E"] = E(inj)
inj["dE"] = inj["E"] - E(pre)
inj = inj[inj["user"].isin(set(dev_users))].reset_index(drop=True)
assert (inj["user"].to_numpy() == D["user"].to_numpy()).all()

mean_score = lg.mean(axis=1).to_numpy()        # only for description, never for decisions
R = pd.DataFrame({"user": D["user"], "y": L["y"], "in_window": M["in_window"], "day_type": M["day_type"],
                  "E": inj["E"], "dE": inj["dE"], "score": mean_score,
                  "cd10": D["cumulative_deviation_10wd"], "cd5": D["cumulative_deviation_5wd"],
                  "nr_m_5wd": D["nr_m_5wd"], "nr_alt_5wd": D["nr_alt_5wd"],
                  "active_channel_count": D["active_channel_count"]})
win = R[R["in_window"]]
prew = R[R["day_type"] == "pre_window"]
prof = pd.DataFrame(index=att5)
prof["detected_repeats"] = n_det
prof["status"] = np.select([n_det == 3, n_det == 0], ["detected_3of3", "missed_3of3"], "unstable")
for c in ("stealth_level", "channel_configuration", "switching_level", "schedule_signature", "window_start"):
    prof[c] = U.loc[att5, c]
prof["start_month"] = pd.to_datetime(prof["window_start"]).dt.month
g = win[win.user.isin(att5)].groupby("user")
prof["sum_injected_dE"] = g["dE"].sum()
prof["max_cd10_in_window"] = g["cd10"].max()
prof["max_cd5_in_window"] = g["cd5"].max()
prof["max_score_in_window"] = g["score"].max()
prof["max_nr_m_5wd_in_window"] = g["nr_m_5wd"].max()
prof["max_nr_alt_5wd_in_window"] = g["nr_alt_5wd"].max()
gp = prew[prew.user.isin(att5)].groupby("user")
prof["pre_window_days"] = gp.size().reindex(att5).fillna(0)
prof["pre_window_max_cd10"] = gp["cd10"].max()
prof["pre_window_max_score"] = gp["score"].max()
gu = gt_u[gt_u.user.isin(att5)].sort_values(["user", "day"]).groupby("user")
prof["channel_sequence"] = gu["injected_channels"].agg(lambda s: ">".join(map(str, s)))
prof["multi_channel_days"] = gu["executed_channel_count"].agg(lambda s: int((s > 1).sum()))
prof["switches"] = gu["is_switch"].sum()
prof["total_injected_events"] = gu["total_injected_events"].sum()
gc = gt_c[gt_c.user.isin(att5)]
prof["mean_count_added"] = gc.groupby("user")["count_added"].mean()
prof["mean_baseline_p95"] = gc.groupby("user")["baseline_p95"].mean()
prof["mean_count_before"] = gc.groupby("user")["count_before"].mean()
prof.to_csv(OUT / "1_5day_attacker_profiles.csv")

# compare missed vs detected (3/3), plus all longer-duration attackers and benign users as reference
num = ["sum_injected_dE", "max_cd10_in_window", "max_cd5_in_window", "max_score_in_window",
       "max_nr_m_5wd_in_window", "max_nr_alt_5wd_in_window", "pre_window_days", "pre_window_max_cd10",
       "pre_window_max_score", "multi_channel_days", "switches", "total_injected_events", "mean_count_added",
       "mean_baseline_p95", "mean_count_before"]
cmp = prof.groupby("status")[num].median().T.astype(float)
cmp.loc["users"] = prof["status"].value_counts().reindex(cmp.columns).astype(float)
ben = R[R.user.isin(U.index[~is_att])].groupby("user")
cmp["benign_users_p99_of_peak"] = np.nan
cmp.loc["max_cd10_in_window", "benign_users_p99_of_peak"] = ben["cd10"].max().quantile(0.99)
cmp.loc["max_cd5_in_window", "benign_users_p99_of_peak"] = ben["cd5"].max().quantile(0.99)
cmp.loc["max_score_in_window", "benign_users_p99_of_peak"] = ben["score"].max().quantile(0.99)
cmp.to_csv(OUT / "1_5day_missed_vs_detected_medians.csv")

for c in ("stealth_level", "channel_configuration", "switching_level", "start_month"):
    pd.crosstab(prof[c], prof["status"], margins=True).to_csv(OUT / f"1_5day_by_{c}.csv")

# injected deviation needed: total injected dE in window by duration (all dev attackers)
allw = win[win.user.isin(U.index[is_att])].groupby("user").agg(sum_dE=("dE", "sum"), max_cd10=("cd10", "max"))
allw = allw.join(U[["window_workdays", "stealth_level"]])
allw.groupby(["window_workdays", "stealth_level"])["sum_dE"].median().unstack().to_csv(OUT / "1_sum_injected_dE_by_duration_stealth.csv")
allw.groupby(["window_workdays", "stealth_level"])["max_cd10"].median().unstack().to_csv(OUT / "1_max_cd10_by_duration_stealth.csv")

# ============================================================ 2. objectives
dur = U["window_workdays"]
row_user = M["user"].to_numpy()
row_dur = pd.Series(row_user).map(dur).to_numpy()
y10 = L["y_10"].to_numpy()
pos_per_user = pd.Series(y10).groupby(row_user).transform("sum").to_numpy()
w_att = np.where(y10 == 1, 1.0 / np.maximum(pos_per_user, 1), 1.0)
benign_row = M["is_attacker"].to_numpy() == 0
DURS = (5, 10, 20, 30, 40, 50)


def objectives(score, rows, users_in):
    """All candidate objectives on one evaluation set (rows: boolean mask; users_in: list)."""
    s, y = score[rows], y10[rows]
    out = {"row_pr_auc": average_precision_score(y, s),
           "attacker_weighted_row_pr_auc": average_precision_score(y, s, sample_weight=w_att[rows])}
    f = pd.DataFrame({"user": row_user[rows], "s": s})
    peak = f.groupby("user")["s"].max().loc[users_in]
    ya = U.loc[users_in, "is_attacker"].to_numpy()
    out["user_pr_auc"] = average_precision_score(ya, peak)
    out["user_roc_auc"] = roc_auc_score(ya, peak)
    rmac, umac = [], []
    for d_ in DURS:
        m = rows & (benign_row | (row_dur == d_))
        rmac.append(average_precision_score(y10[m], score[m]))
        uu = [u for u in users_in if U.loc[u, "is_attacker"] == 0 or dur[u] == d_]
        umac.append(average_precision_score(U.loc[uu, "is_attacker"], peak.loc[uu]))
    out["duration_macro_row_pr_auc"] = float(np.mean(rmac))
    out["duration_macro_user_pr_auc"] = float(np.mean(umac))
    for d_, v in zip(DURS, rmac):
        out[f"row_pr_auc_d{d_}"] = v
    for d_, v in zip(DURS, umac):
        out[f"user_pr_auc_d{d_}"] = v
    return out


obj_rows = []
for algo in ALGOS:
    o = oof(algo)
    for rep, c in enumerate(o.columns):
        sc = o[c].to_numpy()
        for fold in range(5):
            fu = valid_of[rep][valid_of[rep] == fold].index.tolist()
            rows = pd.Series(row_user).isin(set(fu)).to_numpy()
            obj_rows.append(dict(algo=algo, repeat=rep, fold=fold, level="fold", **objectives(sc, rows, fu)))
        obj_rows.append(dict(algo=algo, repeat=rep, fold=-1, level="pooled_repeat",
                             **objectives(sc, np.ones(len(sc), bool), dev_users)))
    print("objectives", algo, flush=True)
obj = pd.DataFrame(obj_rows)
obj.to_csv(OUT / "2_objectives_all.csv", index=False)
keys = ["row_pr_auc", "attacker_weighted_row_pr_auc", "user_pr_auc", "duration_macro_row_pr_auc",
        "duration_macro_user_pr_auc", "user_roc_auc"]
summ = obj[obj.level == "fold"].groupby("algo")[keys].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ.to_csv(OUT / "2_objectives_fold_mean_sd.csv")
# stability: paired lgbm-vs-xgboost and lgbm-vs-catboost differences per fold, relative to their sd (signal/noise)
sn = []
fo = obj[obj.level == "fold"].set_index(["algo", "repeat", "fold"])
for other in ("xgboost", "catboost", "random_forest"):
    for k in keys:
        dd = fo.loc[other, k] - fo.loc["lightgbm", k]
        sn.append(dict(pair=f"{other}-lightgbm", objective=k, mean_diff=dd.mean(), sd_diff=dd.std(),
                       t_like=dd.mean() / (dd.std() / np.sqrt(len(dd))), other_wins=int((dd > 0).sum())))
pd.DataFrame(sn).to_csv(OUT / "2_objective_paired_signal_noise.csv", index=False)
# share of positive rows by duration (what drives row PR-AUC)
pr = pd.DataFrame({"dur": row_dur, "y": y10})
pr[pr.y == 1].groupby("dur").size().rename("y10_positive_rows").to_frame().assign(
    share=lambda t: t.iloc[:, 0] / t.iloc[:, 0].sum(),
    attackers=U[is_att].groupby("window_workdays").size()).to_csv(OUT / "2_positive_rows_by_duration.csv")
obj[(obj.level == "pooled_repeat")].groupby("algo")[[f"row_pr_auc_d{d}" for d in DURS] +
                                                    [f"user_pr_auc_d{d}" for d in DURS]].mean().to_csv(
    OUT / "2_per_duration_pr_auc_pooled.csv")

# ============================================================ 3. observation length
nd = M.groupby("user").size().loc[dev_users].rename("n_days").to_frame()
nd["is_attacker"] = U["is_attacker"]
nd.groupby("is_attacker")["n_days"].describe(percentiles=[.05, .25, .5, .75, .95]).to_csv(OUT / "3_days_per_user.csv")
peaks = pd.DataFrame({f"rep{r}": user_scores(lg[c].to_numpy())[0] for r, c in enumerate(lg.columns)})
bn = nd[nd.is_attacker == 0].join(peaks)
bn["n_days_band"] = pd.cut(bn["n_days"], [0, 60, 90, 104, 200], labels=["<=60", "61-90", "91-104", ">=105"])
rows3 = []
for band, g_ in bn.groupby("n_days_band", observed=True):
    alarmed = np.mean([(g_[f"rep{r}"] > thr[r]).mean() for r in range(3)])
    rows3.append(dict(band=band, users=len(g_), days_min=g_.n_days.min(), days_max=g_.n_days.max(),
                      median_peak=g_[[f"rep{r}" for r in range(3)]].mean(axis=1).median(),
                      share_benign_alarmed_at_1pct_thr=alarmed))
pd.DataFrame(rows3).to_csv(OUT / "3_benign_peak_vs_days.csv", index=False)
# per-day alarm rate of benign rows vs probability of >=1 alarm over n days (independence approximation)
per_day = np.mean([(lg[c].to_numpy()[benign_row] > thr[r]).mean() for r, c in enumerate(lg.columns)])
pd.DataFrame({"benign_row_alarm_rate_at_user_1pct_thr": [per_day],
              "implied_P_any_alarm_104_days_if_independent": [1 - (1 - per_day) ** 104],
              "observed_benign_user_alarm_share": [5 / 511]}).to_csv(OUT / "3_row_vs_user_alarm.csv", index=False)

# ============================================================ 4. latency: pooled vs frozen leave-fold-out thresholds
lat = []
for rep, c in enumerate(lg.columns):
    sc = lg[c].to_numpy()
    peak, _ = user_scores(sc)
    f = pd.DataFrame({"user": M["user"], "s": sc, "w": M["in_window"], "wd": M["wd_from_first_attack"]})
    fw = f[f.w]
    vf = valid_of[rep]
    for scheme in ("pooled_same_set", "frozen_leave_fold_out"):
        for fold in range(5):
            fu = vf[vf == fold].index
            if scheme == "pooled_same_set":
                t = thr[rep]
            else:   # threshold from benign users of the OTHER 4 folds only, then applied to this fold
                other = vf[vf != fold].index
                t = threshold_at_fpr(peak.loc[[u for u in other if U.loc[u, "is_attacker"] == 0]].to_numpy(), 0.01)
            bfold = [u for u in fu if U.loc[u, "is_attacker"] == 0]
            fp = int((peak.loc[bfold] > t).sum())
            first = fw[fw.user.isin(set(fu)) & (fw.s > t)].groupby("user")["wd"].min()
            for u in fu:
                if U.loc[u, "is_attacker"] == 1:
                    lat.append(dict(scheme=scheme, repeat=rep, fold=fold, user=u, duration=dur[u],
                                    stealth=U.loc[u, "stealth_level"], detected=u in first.index,
                                    latency=float(first.get(u, np.nan)), fold_fp=fp, fold_benign=len(bfold)))
lat = pd.DataFrame(lat)
lat.to_csv(OUT / "4_latency_rows.csv", index=False)


def q(s, p):
    return s.quantile(p) if len(s.dropna()) else np.nan


def lat_table(g_):
    return pd.Series({"attacker_runs": len(g_), "detected": g_.detected.mean(),
                      "p25": q(g_.latency, .25), "median": q(g_.latency, .5), "p75": q(g_.latency, .75),
                      "p90": q(g_.latency, .9)})


for by in ("duration", "stealth"):
    lat.groupby(["scheme", by]).apply(lat_table).to_csv(OUT / f"4_latency_by_{by}.csv")
lat.groupby(["scheme", "duration", "stealth"]).apply(lat_table).to_csv(OUT / "4_latency_by_duration_x_stealth.csv")
fpsum = lat.drop_duplicates(["scheme", "repeat", "fold"]).groupby(["scheme", "repeat"])[["fold_fp", "fold_benign"]].sum()
fpsum.to_csv(OUT / "4_fp_counts_by_scheme.csv")
print("done")
