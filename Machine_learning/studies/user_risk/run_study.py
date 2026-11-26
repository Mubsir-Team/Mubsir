"""
MUBSIR — User risk aggregation & alert threshold validation study (DEV ONLY).

Inputs (read only): Stage A LightGBM y_10 out-of-fold scores (work/oof/lightgbm_h10.parquet, 3 repeats, 859 dev
users), the frozen Platt calibration (model/manifest.json), evaluation metadata (work/eval_meta.parquet),
user metadata (config/user_meta.csv), the dev/test split and CV folds (config/).
Test users are dropped at load; no test score file is opened.

Every user-level number is computed per OOF repeat (each dev row scored once per repeat by a model that never saw
that user) and then averaged over the 3 repeats. Bootstrap CIs resample users (stratified attacker/benign) and
average the metric over the 3 repeats inside each resample.

Outputs: results/*.csv, params.json (all experiment parameters).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import spearmanr, beta
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent  # studies/<name>/ -> Model/ (moved 2026-11-26)
RES = HERE / "results"
RES.mkdir(exist_ok=True)

PARAMS = {
    "seed": 42,
    "bootstrap_resamples": 1000,
    "eps_clip": 1e-7,
    "baseline": {"aggregation": "max daily risk over the whole observation period", "threshold": 85.0},
    "max_thresholds": [70, 75, 80, 85, 90, 92.63198568310092],
    "score_thresholds": [60, 70, 75, 80, 85, 90],
    "percentile_thresholds": [50, 60, 70, 80, 85],
    "ewma_thresholds": [60, 70, 80, 85, 90],
    "ewma_half_lives_workdays": [5, 10],
    "ewma_formula": "e_t = lam * risk_t + (1 - lam) * e_(t-1), e_0 = risk_0, lam = 1 - 2^(-1/half_life)",
    "top_k": [3, 5],
    "percentiles": [90, 95],
    "fixed_windows": {"first": [30, 60], "last": [30, 60]},
    "persistence": {"thresholds": [80, 85, 90], "rules_k_of_w": [[1, 1], [2, 2], [2, 3], [3, 5]],
                    "escalation_on_2of3": {"thresholds": [80, 85], "escalate_above": [95, 98]}},
    "iso_fpr_targets": [0.01, 0.03],
    "iso_fpr_method": "threshold per OOF repeat and validation fold, from the scores of benign users in the OTHER 4 folds",
    "exposure_adjusted": {"alphas": [0.01, 0.03],
                          "reference": "benign users of the other 4 folds (same repeat); all sliding windows of the "
                                       "user's length; each reference user weighted equally",
                          "rule": "alert iff max daily risk over n days > T_alpha(n), T_alpha(n) = weighted (1-alpha) "
                                  "quantile of benign window maxima of length n"},
    "horizons_workdays": [10, 20, 30, 60, 90],
    "horizon_start_rule": "benign: the first H workdays of the observation period (and, as robustness, every sliding "
                          "H-window); attackers: the H workdays starting at the first injected attack day",
    "mixed_duration_experiment": {"draws": 20, "horizons": [10, 20, 30, 60, 90],
                                  "rule": "each dev user gets one H uniformly at random; benign: window start uniform "
                                          "over valid starts; attackers: window start uniform among starts whose "
                                          "window contains the first injected day"},
    "incident_cooldown_workdays": 5,
    "workdays_per_employee_month": 21,
    "duration_groups_workdays": [0, 30, 60, 90, 104, 200],
}
(HERE / "params.json").write_text(json.dumps(PARAMS, indent=1), encoding="utf-8")
rng = np.random.default_rng(PARAMS["seed"])

# ====================================================================== load (dev only)
man = json.loads((MODEL_DIR / "model" / "manifest.json").read_text(encoding="utf-8"))
A, B = man["calibration"]["a"], man["calibration"]["b"]
split = pd.read_csv(MODEL_DIR / "config" / "final_split.csv")
dev_users = set(split.loc[split.part == "dev", "user"])
test_users = set(split.loc[split.part == "test", "user"])
folds = pd.read_csv(MODEL_DIR / "config" / "cv_folds.csv")
um = pd.read_csv(MODEL_DIR / "config" / "user_meta.csv").set_index("user")
meta = pd.read_parquet(MODEL_DIR / "work" / "eval_meta.parquet")
meta["user"] = meta["user"].astype(str)
dev_mask = meta["user"].isin(dev_users).to_numpy()
M = meta.loc[dev_mask].reset_index(drop=True)
del meta
assert not set(M["user"]) & test_users and M["user"].nunique() == 859
oof = pd.read_parquet(MODEL_DIR / "work" / "oof" / "lightgbm_h10.parquet")
assert len(oof) == len(M)


def risk_of(p):
    p = np.clip(p, PARAMS["eps_clip"], 1 - PARAMS["eps_clip"])
    return 100.0 / (1.0 + np.exp(-(A * np.log(p / (1 - p)) + B)))


RISK = [risk_of(oof[c].to_numpy()) for c in oof.columns]
R = len(RISK)
users = sorted(dev_users)
first_idx = pd.Series(np.arange(len(M))).groupby(M["user"].to_numpy()).agg(["min", "max"]).loc[users]
SL = [(int(a), int(b) + 1) for a, b in first_idx.to_numpy()]
U = pd.DataFrame({"user": users})
U["is_attacker"] = U["user"].map(um["is_attacker"]).astype(int)
U["n_days"] = [b - a for a, b in SL]
U["stealth"] = U["user"].map(um["stealth_level"])
U["duration"] = U["user"].map(um["window_workdays"])
y = U["is_attacker"].to_numpy()
att = np.where(y == 1)[0]
ben = np.where(y == 0)[0]
P_, N_ = len(att), len(ben)
fold_of = np.zeros((R, len(users)), dtype=int)
for r in range(R):
    vf = folds[(folds.repeat == r) & (folds.role == "valid")].set_index("user")["fold"]
    fold_of[r] = vf.loc[users].to_numpy()
INW = [M["in_window"].to_numpy()[a:b] for a, b in SL]
WD = [M["wd_from_first_attack"].astype(float).to_numpy()[a:b] for a, b in SL]
YD = [M["y"].to_numpy()[a:b] for a, b in SL]
FA = [int(np.argmax(yd)) if yd.any() else -1 for yd in YD]   # position of the first injected day
X = [[RISK[r][a:b] for a, b in SL] for r in range(R)]          # X[r][u] = daily calibrated risk series
EM = PARAMS["workdays_per_employee_month"]
print(f"dev users {len(users)} (attackers {P_}, benign {N_}), rows {len(M):,}, repeats {R}")


# ====================================================================== helpers
def threshold_at_fpr(neg, fpr):
    neg = np.sort(np.asarray(neg, dtype=float))[::-1]
    k = int(np.floor(fpr * len(neg)))
    return neg[min(k, len(neg) - 1)]


def cp(k, n):
    if n == 0:
        return (np.nan, np.nan)
    return (beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


def k_of_w(x, T, k, w):
    e = (x > T).astype(int)
    c = np.cumsum(np.r_[0, e])
    cnt = c[1:] - c[np.maximum(0, np.arange(1, len(x) + 1) - w)]
    return cnt >= k


def ewma(x, h):
    lam = 1 - 2 ** (-1 / h)
    out = np.empty(len(x))
    acc = x[0]
    for i, v in enumerate(x):
        acc = v if i == 0 else lam * v + (1 - lam) * acc
        out[i] = acc
    return out


def incidents(fire, cooldown):
    idx = np.flatnonzero(fire)
    if len(idx) == 0:
        return 0
    return int(1 + np.sum(np.diff(idx) > cooldown))


def weighted_quantile(values, weights, q):
    o = np.argsort(values)
    v, w = values[o], weights[o]
    cw = np.cumsum(w) / w.sum()
    return v[np.searchsorted(cw, q, side="left").clip(0, len(v) - 1)]


BOOT_A = rng.integers(0, P_, size=(PARAMS["bootstrap_resamples"], P_))
BOOT_B = rng.integers(0, N_, size=(PARAMS["bootstrap_resamples"], N_))


def confusion_metrics(fired_r):
    """fired_r: (R, U) bool -> dict of repeat-mean metrics + bootstrap CIs (B, metric)."""
    fa = fired_r[:, att].astype(float)
    fb = fired_r[:, ben].astype(float)

    def from_counts(tp, fn, fp, tn):
        rec = tp / (tp + fn)
        spec = tn / (tn + fp)
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
            f2 = np.where(4 * prec + rec > 0, 5 * prec * rec / (4 * prec + rec), 0.0)
            den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)
        return dict(recall=rec, specificity=spec, precision=prec, F1=f1, F2=f2, balanced_accuracy=(rec + spec) / 2,
                    MCC=mcc, FPR=fp / (fp + tn))

    tp = fa.sum(1); fn = P_ - tp; fp = fb.sum(1); tn = N_ - fp
    point = {k: float(np.nanmean(v)) for k, v in from_counts(tp, fn, fp, tn).items()}
    point.update(TP=float(tp.mean()), FN=float(fn.mean()), FP=float(fp.mean()), TN=float(tn.mean()))
    # bootstrap: counts per resample and repeat
    tpb = np.stack([fa[r][BOOT_A].sum(1) for r in range(R)], 1)        # (B, R)
    fpb = np.stack([fb[r][BOOT_B].sum(1) for r in range(R)], 1)
    mb = from_counts(tpb, P_ - tpb, fpb, N_ - fpb)
    boot = {k: np.nanmean(v, axis=1) for k, v in mb.items()}
    return point, boot


RECORDS, BOOTS, PER_USER = [], {}, {}


def record(pid, family, aggregation, rule, threshold, fired, det=None, lat=None, events=None, inc=None,
           population="all 859 dev users", note=""):
    """Store a policy result. fired/det: (R, U) bool; lat: (R, U) float; events/inc: (R, U) int."""
    point, boot = confusion_metrics(fired)
    row = dict(policy_id=pid, family=family, aggregation=aggregation, alert_rule=rule, threshold=threshold,
               population=population, n_users=len(users), n_attackers=P_, n_benign=N_, **point)
    for k, v in boot.items():
        row[f"{k}_ci_low"], row[f"{k}_ci_high"] = np.nanquantile(v, [0.025, 0.975])
    row["employees_alerted"] = row["TP"] + row["FP"]
    row["benign_alerted"] = row["FP"]
    row["attackers_detected_user_level"] = row["TP"]
    row["attackers_missed_user_level"] = row["FN"]
    if det is not None:
        row["attack_window_detection"] = float(det[:, att].mean())
        lat_all = lat[:, att][det[:, att]]
        row["latency_median"] = float(np.median(lat_all)) if len(lat_all) else np.nan
        row["latency_p75"] = float(np.quantile(lat_all, .75)) if len(lat_all) else np.nan
        row["latency_p90"] = float(np.quantile(lat_all, .9)) if len(lat_all) else np.nan
        row["attackers_alerted_only_outside_window"] = float((fired[:, att] & ~det[:, att]).sum(1).mean())
    for name, mask in (("short_5d", U["duration"].to_numpy() == 5), ("high_stealth", U["stealth"].to_numpy() == "High")):
        row[f"recall_{name}"] = float(fired[:, mask].mean())
        if det is not None:
            row[f"window_detection_{name}"] = float(det[:, mask].mean())
    if events is not None:
        b_months = U.loc[ben, "n_days"].sum() / EM
        all_months = U["n_days"].sum() / EM
        row["alert_events_per_100_employee_months"] = float(events.sum(1).mean() / all_months * 100)
        row["benign_alert_events_per_100_benign_employee_months"] = float(events[:, ben].sum(1).mean() / b_months * 100)
        row["incidents_per_100_employee_months"] = float(inc.sum(1).mean() / all_months * 100)
        row["benign_incidents_per_100_benign_employee_months"] = float(inc[:, ben].sum(1).mean() / b_months * 100)
        al = events[fired]
        il = inc[fired]
        row["events_per_alerted_employee_mean"] = float(al.mean()) if len(al) else np.nan
        row["events_per_alerted_employee_median"] = float(np.median(al)) if len(al) else np.nan
        row["incidents_per_alerted_employee_mean"] = float(il.mean()) if len(il) else np.nan
        bal = events[:, ben][fired[:, ben]]
        row["events_per_alerted_benign_employee_mean"] = float(bal.mean()) if len(bal) else np.nan
    row["unique_employees_alerted_per_100"] = row["employees_alerted"] / len(users) * 100
    row["benign_employees_alerted_per_100_benign"] = row["FP"] / N_ * 100
    row["note"] = note
    RECORDS.append(row)
    BOOTS[pid] = boot
    PER_USER[pid] = dict(fired=fired, det=det, lat=lat)


# ====================================================================== daily-rule policies
def run_daily(pid, family, aggregation, rule, threshold, fire_fn):
    fired = np.zeros((R, len(users)), bool)
    det = np.zeros_like(fired)
    lat = np.full(fired.shape, np.nan)
    ev = np.zeros(fired.shape, int)
    inc = np.zeros(fired.shape, int)
    for r in range(R):
        for u in range(len(users)):
            f = fire_fn(X[r][u])
            fired[r, u] = f.any()
            ev[r, u] = f.sum()
            inc[r, u] = incidents(f, PARAMS["incident_cooldown_workdays"])
            if y[u]:
                fw = f & INW[u]
                det[r, u] = fw.any()
                if det[r, u]:
                    lat[r, u] = WD[u][fw].min()
    record(pid, family, aggregation, rule, threshold, fired, det, lat, ev, inc)


for T in PARAMS["max_thresholds"]:
    run_daily(f"max_full>{T:.2f}", "A current max", "max daily risk, whole period",
              f"risk_peak > {T:.2f} (any single day)", T, lambda x, T=T: x > T)
for T in PARAMS["persistence"]["thresholds"]:
    for k, w in PARAMS["persistence"]["rules_k_of_w"]:
        if (k, w) == (1, 1):
            continue
        run_daily(f"persist_{k}of{w}>{T}", "G persistence", "daily risk",
                  f">= {k} days above {T} in the last {w} workdays", T, lambda x, T=T, k=k, w=w: k_of_w(x, T, k, w))
for T in PARAMS["persistence"]["escalation_on_2of3"]["thresholds"]:
    for E in PARAMS["persistence"]["escalation_on_2of3"]["escalate_above"]:
        run_daily(f"persist_2of3>{T}_or_day>{E}", "G persistence + escalation", "daily risk",
                  f">= 2 of 3 days above {T}, or any single day above {E}", T,
                  lambda x, T=T, E=E: k_of_w(x, T, 2, 3) | (x > E))
EW = {h: [[ewma(X[r][u], h) for u in range(len(users))] for r in range(R)] for h in PARAMS["ewma_half_lives_workdays"]}
for h in PARAMS["ewma_half_lives_workdays"]:
    for T in PARAMS["ewma_thresholds"]:
        fired = np.zeros((R, len(users)), bool); det = np.zeros_like(fired); lat = np.full(fired.shape, np.nan)
        ev = np.zeros(fired.shape, int); inc = np.zeros(fired.shape, int)
        for r in range(R):
            for u in range(len(users)):
                f = EW[h][r][u] > T
                fired[r, u] = f.any(); ev[r, u] = f.sum(); inc[r, u] = incidents(f, PARAMS["incident_cooldown_workdays"])
                if y[u]:
                    fw = f & INW[u]
                    det[r, u] = fw.any()
                    if det[r, u]:
                        lat[r, u] = WD[u][fw].min()
        record(f"ewma_h{h}>{T}", "E EWMA", f"EWMA of daily risk, half-life {h} workdays",
               f"EWMA > {T} on any day", T, fired, det, lat, ev, inc)


# ====================================================================== user-score policies
def user_scores(fn):
    return np.array([[fn(X[r][u], u, r) for u in range(len(users))] for r in range(R)])


SCORES = {
    "max_full": ("A current max", "max daily risk, whole period", user_scores(lambda x, u, r: x.max())),
    "max_first30": ("B fixed window", "max over the FIRST 30 workdays", user_scores(lambda x, u, r: x[:30].max())),
    "max_first60": ("B fixed window", "max over the FIRST 60 workdays", user_scores(lambda x, u, r: x[:60].max())),
    "max_last30": ("B fixed window", "max over the LAST 30 workdays", user_scores(lambda x, u, r: x[-30:].max())),
    "max_last60": ("B fixed window", "max over the LAST 60 workdays", user_scores(lambda x, u, r: x[-60:].max())),
    "top3_mean": ("C top-k", "mean of the 3 highest daily risks", user_scores(lambda x, u, r: np.sort(x)[-3:].mean())),
    "top5_mean": ("C top-k", "mean of the 5 highest daily risks", user_scores(lambda x, u, r: np.sort(x)[-5:].mean())),
    "p95": ("D percentile", "95th percentile of daily risk", user_scores(lambda x, u, r: np.quantile(x, 0.95))),
    "p90": ("D percentile", "90th percentile of daily risk", user_scores(lambda x, u, r: np.quantile(x, 0.90))),
}
for h in PARAMS["ewma_half_lives_workdays"]:
    SCORES[f"ewma_h{h}_max"] = ("E EWMA", f"max of EWMA (half-life {h})",
                                np.array([[EW[h][r][u].max() for u in range(len(users))] for r in range(R)]))
short_users = int((U["n_days"] < 30).sum())
for key, (fam, agg, S) in SCORES.items():
    if key == "max_full" or key.startswith("ewma"):
        pass   # fixed-threshold versions recorded above as daily rules
    else:
        grid = PARAMS["percentile_thresholds"] if key.startswith("p9") else PARAMS["score_thresholds"]
        for T in grid:
            note = ""
            if key.startswith("max_first") or key.startswith("max_last"):
                note = ("snapshot at one fixed position: attackers whose window lies outside it cannot be detected; "
                        f"{short_users} users have < 30 days and use all their days")
            record(f"{key}>{T}", fam, agg, f"score > {T}", T, S > T, note=note)
    for tgt in PARAMS["iso_fpr_targets"]:
        fired = np.zeros(S.shape, bool)
        thr_used = []
        for r in range(R):
            for k in range(5):
                inn = fold_of[r] == k
                ref = S[r][(~inn) & (y == 0)]
                t = threshold_at_fpr(ref, tgt)
                thr_used.append(t)
                fired[r, inn] = S[r][inn] > t
        record(f"{key}@isoFPR{int(tgt * 100)}%", fam + " (iso-FPR)", agg,
               f"score > cross-fitted threshold for {tgt:.0%} benign-user FPR", float(np.mean(thr_used)), fired,
               note=f"threshold from benign users of the other folds; mean threshold {np.mean(thr_used):.2f} "
                    f"(range {np.min(thr_used):.2f}-{np.max(thr_used):.2f})")


# ====================================================================== exposure-adjusted maximum
def window_maxima(x, n):
    return sliding_window_view(x, n).max(axis=1) if len(x) >= n else np.array([])


def exposure_threshold(r, ref_users, n, alpha, cache):
    key = (r, tuple(ref_users[:3]), len(ref_users), n, alpha)
    if key in cache:
        return cache[key]
    vals, wts = [], []
    for u in ref_users:
        wm = window_maxima(X[r][u], n)
        if len(wm):
            vals.append(wm); wts.append(np.full(len(wm), 1.0 / len(wm)))
    if not vals:
        cache[key] = (np.nan, 0)
    else:
        cache[key] = (float(weighted_quantile(np.concatenate(vals), np.concatenate(wts), 1 - alpha)), len(vals))
    return cache[key]


cache = {}
for alpha in PARAMS["exposure_adjusted"]["alphas"]:
    fired = np.zeros((R, len(users)), bool)
    nref = []
    for r in range(R):
        for k in range(5):
            ref = [u for u in ben if fold_of[r][u] != k]
            for u in np.where(fold_of[r] == k)[0]:
                t, n_ref = exposure_threshold(r, ref, U.loc[u, "n_days"], alpha, cache)
                nref.append(n_ref)
                fired[r, u] = X[r][u].max() > t
    record(f"exposure_adj_max@alpha{alpha}", "F exposure-adjusted", "max daily risk vs benign max over the same n days",
           f"risk_peak > T_alpha(n_days), alpha = {alpha}", np.nan, fired,
           note=f"reference benign users per threshold: median {int(np.median(nref))}, min {int(np.min(nref))}")

cmp = pd.DataFrame(RECORDS)

# paired comparisons against the baseline max_full>85
base = "max_full>85.00"
pairs = []
for pid in cmp.policy_id:
    if pid == base:
        continue
    d = {}
    for k in ("recall", "FPR", "MCC", "balanced_accuracy"):
        diff = BOOTS[pid][k] - BOOTS[base][k]
        d[f"d_{k}"] = float(cmp.loc[cmp.policy_id == pid, k].iloc[0] - cmp.loc[cmp.policy_id == base, k].iloc[0])
        d[f"d_{k}_ci_low"], d[f"d_{k}_ci_high"] = np.nanquantile(diff, [0.025, 0.975])
    pairs.append(dict(policy_id=pid, **d))
cmp = cmp.merge(pd.DataFrame(pairs), on="policy_id", how="left")
cmp.to_csv(RES / "policy_comparison.csv", index=False)

# per-group performance (duration x stealth) for every policy
grp = []
for pid, d in PER_USER.items():
    for gname, col in (("attack_duration", "duration"), ("stealth", "stealth")):
        for g in sorted(U.loc[att, col].dropna().unique(), key=str):
            m = (U[col].to_numpy() == g) & (y == 1)
            row = dict(policy_id=pid, group_type=gname, group=g, n_attackers=int(m.sum()),
                       user_level_recall=float(d["fired"][:, m].mean()))
            if d["det"] is not None:
                row["attack_window_detection"] = float(d["det"][:, m].mean())
                l = d["lat"][:, m][d["det"][:, m]]
                row["latency_median"] = float(np.median(l)) if len(l) else np.nan
            grp.append(row)
pd.DataFrame(grp).to_csv(RES / "policy_by_duration_stealth.csv", index=False)
print("policies evaluated:", len(cmp))

# ====================================================================== observation-duration bias
dur_rows = []
nd = U["n_days"]
for who, m in (("all", np.ones(len(U), bool)), ("benign", y == 0), ("attackers", y == 1)):
    s = nd[m]
    dur_rows.append(dict(section="distribution", subset=who, n=int(m.sum()), min=s.min(), p1=s.quantile(.01),
                         p5=s.quantile(.05), p10=s.quantile(.10), p25=s.quantile(.25), median=s.median(),
                         mean=s.mean(), p75=s.quantile(.75), p90=s.quantile(.9), max=s.max()))
bins = PARAMS["duration_groups_workdays"]
U["duration_group"] = pd.cut(U["n_days"], bins).astype(str)
peak = SCORES["max_full"][2]                                  # (R, U)
for who, m in (("benign", y == 0), ("attackers", y == 1)):
    for g, idx in U[m].groupby("duration_group").groups.items():
        ii = np.array(list(idx))
        share = (peak[:, ii] > 85).mean()
        k = int(round(share * len(ii)))
        lo, hi = cp(k, len(ii))
        dur_rows.append(dict(section="cross_sectional_by_duration_group", subset=who, group=g, n=len(ii),
                             median_risk_peak=float(np.median(peak[:, ii])), share_risk_peak_gt_85=float(share),
                             ci_low=lo, ci_high=hi))
for who, m in (("benign", y == 0), ("attackers", y == 1)):
    rho = np.mean([spearmanr(nd[m], peak[r][m]).statistic for r in range(R)])
    ii = np.where(m)[0]
    bs = []
    for _ in range(PARAMS["bootstrap_resamples"]):
        s = rng.choice(ii, len(ii))
        bs.append(np.mean([spearmanr(nd.to_numpy()[s], peak[r][s]).statistic for r in range(R)]))
    dur_rows.append(dict(section="spearman_ndays_vs_risk_peak", subset=who, n=int(m.sum()), rho=rho,
                         ci_low=np.nanquantile(bs, .025), ci_high=np.nanquantile(bs, .975),
                         note="n_days varies little: 838 of 859 users have >= 105 workdays"))
# logistic regression: alert(85) ~ log(n_days) + attacker, coefficients averaged over repeats, bootstrap CI
Xlr = np.c_[np.log(nd.to_numpy()), y]
coefs = []
for r in range(R):
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(Xlr, peak[r] > 85)
    coefs.append(lr.coef_[0])
coef = np.mean(coefs, axis=0)
bs = []
for _ in range(300):
    s = np.r_[rng.choice(att, P_), rng.choice(ben, N_)]
    cr = []
    for r in range(R):
        yy = peak[r][s] > 85
        if yy.all() or (~yy).all():
            continue
        cr.append(LogisticRegression(C=1e6, max_iter=2000).fit(Xlr[s], yy).coef_[0])
    bs.append(np.mean(cr, axis=0))
bs = np.array(bs)
for j, name in enumerate(["log(n_days)", "is_attacker"]):
    dur_rows.append(dict(section="logistic_alert85", subset="all", term=name, coef=coef[j],
                         odds_ratio=float(np.exp(coef[j])), ci_low=float(np.quantile(bs[:, j], .025)),
                         ci_high=float(np.quantile(bs[:, j], .975)),
                         note="alert(risk_peak>85) ~ log(n_days) + is_attacker; CI on the coefficient"))

# ---- controlled exposure experiment: benign max over fixed horizons
p_day85 = np.mean([np.concatenate([X[r][u] for u in ben]).__gt__(85).mean() for r in range(R)])
horizons = PARAMS["horizons_workdays"] + ["full"]
horizon_curve = []
for T in (80, 85, 90):
    p_day = np.mean([(np.concatenate([X[r][u] for u in ben]) > T).mean() for r in range(R)])
    for H in horizons:
        elig = [u for u in ben if H == "full" or U.loc[u, "n_days"] >= H]
        excl = N_ - len(elig)
        first_max = np.array([[X[r][u][:(None if H == 'full' else H)].max() for u in elig] for r in range(R)])
        share_first = (first_max > T).mean(axis=0)                    # per user, mean over repeats
        bs = [share_first[rng.integers(0, len(elig), len(elig))].mean() for _ in range(PARAMS["bootstrap_resamples"])]
        if H == "full":
            sl = share_first.mean()
        else:
            sl_u = np.array([np.mean([(window_maxima(X[r][u], H) > T).mean() for r in range(R)]) for u in elig])
            sl = sl_u.mean()
        Hn = np.median([U.loc[u, "n_days"] for u in elig]) if H == "full" else H
        horizon_curve.append(dict(section="benign_horizon_curve", threshold=T, horizon=str(H), n_benign_eligible=len(elig),
                                  n_benign_excluded_short_history=excl,
                                  share_first_H_max_above=float(share_first.mean()), ci_low=float(np.quantile(bs, .025)),
                                  ci_high=float(np.quantile(bs, .975)), share_any_H_window_max_above=float(sl),
                                  benign_daily_exceedance_rate=float(p_day),
                                  independence_prediction=float(1 - (1 - p_day) ** Hn),
                                  max_p50=float(np.median(first_max)), max_p90=float(np.quantile(first_max, .9)),
                                  max_p95=float(np.quantile(first_max, .95)), max_p99=float(np.quantile(first_max, .99))))
# attackers: alert within H workdays from the first injected day
for T in (80, 85, 90):
    for H in PARAMS["horizons_workdays"]:
        hit = np.array([[(X[r][u][FA[u]:FA[u] + H] > T).any() for u in att] for r in range(R)])
        horizon_curve.append(dict(section="attacker_alert_within_H_of_first_attack_day", threshold=T, horizon=str(H),
                                  n_attackers=P_, share=float(hit.mean())))
# benign max-risk distribution by horizon (for the ECDF figure): first-H window, repeat 0
ecdf_rows = []
for H in horizons:
    elig = [u for u in ben if H == "full" or U.loc[u, "n_days"] >= H]
    for r in range(R):
        for u in elig:
            ecdf_rows.append(dict(horizon=str(H), repeat=r, user=users[u],
                                  max_risk=float(X[r][u][:(None if H == 'full' else H)].max())))
pd.DataFrame(ecdf_rows).to_csv(RES / "benign_max_by_horizon_long.csv", index=False)

# ---- mixed-duration experiment: does each policy keep the same benign FPR across durations?
mixed = []
cache_m = {}
for draw in range(PARAMS["mixed_duration_experiment"]["draws"]):
    Hs = rng.choice(PARAMS["mixed_duration_experiment"]["horizons"], size=len(users))
    starts = np.zeros(len(users), int)
    for u in range(len(users)):
        n, H = U.loc[u, "n_days"], Hs[u]
        if n < H:
            starts[u] = -1
            continue
        if y[u] == 0:
            starts[u] = rng.integers(0, n - H + 1)
        else:
            lo, hi = max(0, FA[u] - H + 1), min(FA[u], n - H)
            starts[u] = rng.integers(lo, hi + 1) if hi >= lo else -1
    for r in range(R):
        for u in range(len(users)):
            if starts[u] < 0:
                continue
            x = X[r][u][starts[u]:starts[u] + Hs[u]]
            k = fold_of[r][u]
            ref = [v for v in ben if fold_of[r][v] != k]
            row = dict(draw=draw, repeat=r, user=users[u], is_attacker=int(y[u]), H=int(Hs[u]),
                       stealth=U.loc[u, "stealth"], duration=U.loc[u, "duration"])
            row["max>85"] = x.max() > 85
            row["max>90"] = x.max() > 90
            row["persist_2of3>85"] = k_of_w(x, 85, 2, 3).any()
            row["ewma_h5>80"] = (ewma(x, 5) > 80).any()
            row["top3_mean>85"] = np.sort(x)[-3:].mean() > 85
            for alpha in PARAMS["exposure_adjusted"]["alphas"]:
                t, _ = exposure_threshold(r, ref, int(Hs[u]), alpha, cache_m)
                row[f"exposure_adj@alpha{alpha}"] = x.max() > t
            mixed.append(row)
mixed = pd.DataFrame(mixed)
mixed.to_csv(RES / "mixed_duration_rows.csv.gz", index=False, compression="gzip")
pol_cols = [c for c in mixed.columns if c not in ("draw", "repeat", "user", "is_attacker", "H", "stealth", "duration")]
mix_sum = []
for c in pol_cols:
    for H, g in mixed.groupby("H"):
        mix_sum.append(dict(section="mixed_duration", policy=c, H=H, benign_FPR=g.loc[g.is_attacker == 0, c].mean(),
                            attacker_recall=g.loc[g.is_attacker == 1, c].mean(),
                            n_benign_windows=int((g.is_attacker == 0).sum()), n_attacker_windows=int((g.is_attacker == 1).sum())))
    g = mixed
    mix_sum.append(dict(section="mixed_duration", policy=c, H="all", benign_FPR=g.loc[g.is_attacker == 0, c].mean(),
                        attacker_recall=g.loc[g.is_attacker == 1, c].mean(),
                        high_stealth_recall=g.loc[(g.is_attacker == 1) & (g.stealth == "High"), c].mean(),
                        short_5d_recall=g.loc[(g.is_attacker == 1) & (g.duration == 5), c].mean(),
                        fpr_ratio_H90_vs_H10=np.nan))
mix_sum = pd.DataFrame(mix_sum)
for c in pol_cols:
    f = mix_sum[(mix_sum.policy == c) & (mix_sum.H != "all")].set_index("H")["benign_FPR"]
    mix_sum.loc[(mix_sum.policy == c) & (mix_sum.H == "all"), "fpr_ratio_H90_vs_H10"] = f.loc[90] / f.loc[10] if f.loc[10] > 0 else np.inf

# exposure-adjusted thresholds T_alpha(H) (all benign dev users, mean over repeats) + benign alert rate per
# employee-month by horizon (does the PER-MONTH rate stay constant? if alerts were a constant-rate process it would)
thr_rows = []
for H in PARAMS["horizons_workdays"] + [105]:
    for alpha in PARAMS["exposure_adjusted"]["alphas"]:
        ts = [exposure_threshold(r, list(ben), H, alpha, {})[0] for r in range(R)]
        thr_rows.append(dict(section="exposure_adjusted_threshold", horizon=str(H), alpha=alpha,
                             T_alpha=float(np.mean(ts)), n_benign_reference=int(sum(U.loc[ben, 'n_days'] >= H))))
for c in pol_cols:
    for H in PARAMS["mixed_duration_experiment"]["horizons"]:
        f = mix_sum[(mix_sum.policy == c) & (mix_sum.H == H)]["benign_FPR"].iloc[0]
        thr_rows.append(dict(section="benign_alerted_per_100_benign_employee_months_by_H", policy=c, horizon=str(H),
                             benign_FPR=float(f), per_100_employee_months=float(f / (H / EM) * 100)))
dur = pd.concat([pd.DataFrame(dur_rows), pd.DataFrame(horizon_curve), mix_sum, pd.DataFrame(thr_rows)], ignore_index=True)
dur.to_csv(RES / "duration_bias_analysis.csv", index=False)

# ====================================================================== persistence + volume tables
per = cmp[cmp.family.str.startswith(("A current", "G persistence", "E EWMA"))].copy()
per_cols = ["policy_id", "family", "alert_rule", "threshold", "recall", "recall_ci_low", "recall_ci_high", "FPR",
            "FPR_ci_low", "FPR_ci_high", "precision", "employees_alerted", "benign_alerted",
            "attackers_missed_user_level", "attack_window_detection", "latency_median", "latency_p75", "latency_p90",
            "recall_short_5d", "window_detection_short_5d", "recall_high_stealth", "window_detection_high_stealth",
            "d_recall", "d_recall_ci_low", "d_recall_ci_high", "d_FPR", "d_FPR_ci_low", "d_FPR_ci_high"]
per[per_cols].to_csv(RES / "persistence_policies.csv", index=False)
vol_cols = ["policy_id", "family", "alert_rule", "threshold", "unique_employees_alerted_per_100",
            "benign_employees_alerted_per_100_benign", "alert_events_per_100_employee_months",
            "benign_alert_events_per_100_benign_employee_months", "incidents_per_100_employee_months",
            "benign_incidents_per_100_benign_employee_months", "events_per_alerted_employee_mean",
            "events_per_alerted_employee_median", "incidents_per_alerted_employee_mean",
            "events_per_alerted_benign_employee_mean"]
per[vol_cols].to_csv(RES / "operational_alert_volume.csv", index=False)
meta_out = dict(dev_users=len(users), attackers=P_, benign=N_, rows=int(len(M)), repeats=R,
                benign_employee_months=float(U.loc[ben, "n_days"].sum() / EM),
                all_employee_months=float(U["n_days"].sum() / EM), benign_daily_exceedance_rate_85=float(p_day85))
(RES / "study_meta.json").write_text(json.dumps(meta_out, indent=1), encoding="utf-8")
print(json.dumps(meta_out, indent=1))
print("done")
