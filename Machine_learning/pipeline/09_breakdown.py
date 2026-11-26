"""
09 — Performance breakdown by simulation dimension (evaluation only; no decision is taken here).

Two sources, always reported side by side:
  dev   Stage A LightGBM y_10 out-of-fold scores (3 repeats), calibrated with the frozen Platt mapping
        (859 users: 348 attackers). Rates are means over the 3 repeats. Main source for small cells.
  test  the single frozen test scoring (work/test/test_scores.parquet; 52 attackers, 84 benign).
Thresholds: the frozen bands (risk > 85 alert, > 60) and the dev 1% user-FPR point, all from the manifest.

Dimensions (attacker level): stealth level, attack duration, switching level, channel configuration,
duration x stealth. Day level: injected channel set and same-day synchronisation of attack days, and the
alarm rate on non-attack day types (benign, pre-window, forced off-day, post-window). Active-days-only
row PR-AUC (audit M2).

Outputs: results/09_*.csv, reports/09_breakdown.md
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import average_precision_score

from common import WORK, RESULTS, REPORTS, MODEL_DIR, LABELS, EVAL_META, USER_META, FINAL_SPLIT, load_dataset, md_table

EPS = 1e-7
man = json.loads((MODEL_DIR / "model" / "manifest.json").read_text(encoding="utf-8"))
A, B = man["calibration"]["a"], man["calibration"]["b"]
T85, T60, T1 = 85.0, 60.0, man["thresholds"]["dev_1pct_user_fpr_risk"]
STEALTH = {"High": "High (0.25)", "Medium": "Medium (0.50)", "Low": "Low (0.75)"}


def risk_of(p):
    p = np.clip(p, EPS, 1 - EPS)
    return 100.0 / (1.0 + np.exp(-(A * np.log(p / (1 - p)) + B)))


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return f"{lo:.2f}-{hi:.2f}"


# ------------------------------------------------------------------ frames
df = load_dataset()
lab = pd.read_parquet(LABELS)
meta = pd.read_parquet(EVAL_META)
um = pd.read_csv(USER_META).set_index("user")
split = pd.read_csv(FINAL_SPLIT)
base = pd.DataFrame({"user": df["user"].astype(str), "day": df["day"], "y": lab["y"], "y_10": lab["y_10"],
                     "in_window": meta["in_window"], "day_type": meta["day_type"],
                     "wd": meta["wd_from_first_attack"].astype(float), "injected_channels": meta["injected_channels"],
                     "sync_type": meta["sync_type"], "active": df["active_channel_count"] > 0})
base["is_attacker"] = base["user"].map(um["is_attacker"])
for c in ("stealth_level", "window_workdays", "switching_level", "channel_configuration"):
    base[c] = base["user"].map(um[c])
base["stealth_level"] = base["stealth_level"].map(STEALTH)

dev_mask = base["user"].isin(set(split.loc[split.part == "dev", "user"])).to_numpy()
oof = pd.read_parquet(WORK / "oof" / "lightgbm_h10.parquet")
dev_frames = [base[dev_mask].reset_index(drop=True).assign(risk=risk_of(oof[c].to_numpy())) for c in oof.columns]
ts = pd.read_parquet(WORK / "test" / "test_scores.parquet")
test = base.merge(ts[["user", "day", "risk"]], on=["user", "day"], how="inner")
assert len(test) == len(ts) and test["user"].nunique() == 136


# ------------------------------------------------------------------ attacker-level breakdown
def attacker_table(frame, col, t):
    att = frame[frame.is_attacker == 1]
    inwin = att[att.in_window].groupby("user")["risk"].max()
    first = att[att.in_window & (att.risk > t)].groupby("user")["wd"].min()
    u = att.groupby("user")[col].first().to_frame("group")
    u["detected"] = inwin.reindex(u.index) > t
    u["latency"] = first.reindex(u.index)
    return u


def breakdown(col):
    out = []
    dev_tabs = [attacker_table(f, col, T85) for f in dev_frames]
    test_tab = attacker_table(test, col, T85)
    groups = sorted(dev_tabs[0]["group"].dropna().unique(), key=lambda g: (str(type(g)), g))
    benign_dev = [f[f.is_attacker == 0] for f in dev_frames]
    benign_test = test[test.is_attacker == 0]
    for g in groups:
        n_dev = int((dev_tabs[0]["group"] == g).sum())
        rates = [t_.loc[t_.group == g, "detected"].mean() for t_ in dev_tabs]
        lat = pd.concat([t_.loc[(t_.group == g) & t_.detected, "latency"] for t_ in dev_tabs])
        ap_dev = np.mean([average_precision_score(
            pd.concat([f[(f.is_attacker == 1) & (f[col] == g)], b])["y_10"],
            pd.concat([f[(f.is_attacker == 1) & (f[col] == g)], b])["risk"]) for f, b in zip(dev_frames, benign_dev)])
        tt = test_tab[test_tab.group == g]
        k, n = int(tt.detected.sum()), len(tt)
        sub = pd.concat([test[(test.is_attacker == 1) & (test[col] == g)], benign_test])
        out.append({col: g, "dev_attackers": n_dev, "dev_detected@85": np.mean(rates),
                    "dev_range_3_repeats": f"{min(rates):.2f}-{max(rates):.2f}",
                    "dev_latency_median": lat.median(), "dev_latency_p90": lat.quantile(.9),
                    "dev_row_pr_auc": ap_dev,
                    "test_attackers": n, "test_detected@85": f"{k}/{n}", "test_rate_95CI": cp(k, n) if n else "",
                    "test_latency_median": tt.loc[tt.detected, "latency"].median(),
                    "test_row_pr_auc": average_precision_score(sub["y_10"], sub["risk"]) if n else np.nan,
                    "small_cell": "yes" if n < 10 else ""})
    return pd.DataFrame(out)


tables = {c: breakdown(c) for c in ("stealth_level", "window_workdays", "switching_level", "channel_configuration")}
for c, t_ in tables.items():
    t_.to_csv(RESULTS / f"09_by_{c}.csv", index=False)

# duration x stealth (dev, mean of 3 repeats)
cross = []
for f in dev_frames:
    a = attacker_table(f, "window_workdays", T85).join(f.groupby("user")["stealth_level"].first())
    cross.append(a.pivot_table(index="group", columns="stealth_level", values="detected", aggfunc="mean"))
cross = sum(cross) / len(cross)
counts = attacker_table(dev_frames[0], "window_workdays", T85).join(dev_frames[0].groupby("user")["stealth_level"].first()) \
    .pivot_table(index="group", columns="stealth_level", values="detected", aggfunc="size")
cross.index.name = "duration"
cross.to_csv(RESULTS / "09_dev_duration_x_stealth.csv")


# ------------------------------------------------------------------ day-level: attack days by channel set / sync
def day_table(col):
    rows = []
    for name, frames in (("dev", dev_frames), ("test", [test])):
        for f in frames:
            a = f[f.y == 1]
            for g, s in a.groupby(col):
                rows.append(dict(source=name, group=g, attack_days=len(s), share_above_85=(s.risk > T85).mean(),
                                 share_above_60=(s.risk > T60).mean(), median_risk=s.risk.median()))
    r = pd.DataFrame(rows).groupby(["group", "source"]).mean().unstack("source")
    r.columns = [f"{b}_{a}" for a, b in r.columns]
    return r.sort_values("dev_attack_days", ascending=False)


ch = day_table("injected_channels")
sync = day_table("sync_type")
ch.to_csv(RESULTS / "09_attack_days_by_channels.csv")
sync.to_csv(RESULTS / "09_attack_days_by_sync.csv")

# ------------------------------------------------------------------ non-attack day types: alarm rates
neg = []
for name, frames in (("dev", dev_frames), ("test", [test])):
    for f in frames:
        for g, s in f[f.y == 0].groupby("day_type"):
            neg.append(dict(source=name, day_type=g, rows=len(s), alarm_rate_85=(s.risk > T85).mean(),
                            alarm_rate_60=(s.risk > T60).mean(), y10_positive_share=s.y_10.mean()))
neg = pd.DataFrame(neg).groupby(["day_type", "source"]).mean().unstack("source")
neg.columns = [f"{b}_{a}" for a, b in neg.columns]
neg.to_csv(RESULTS / "09_non_attack_day_alarms.csv")

# ------------------------------------------------------------------ active days only (audit M2)
act = []
for name, frames in (("dev", dev_frames), ("test", [test])):
    vals_all, vals_act, share = [], [], []
    for f in frames:
        vals_all.append(average_precision_score(f.y_10, f.risk))
        a = f[f.active]
        vals_act.append(average_precision_score(a.y_10, a.risk))
        share.append(f.active.mean())
    act.append(dict(source=name, row_pr_auc_all_rows=np.mean(vals_all), row_pr_auc_active_rows=np.mean(vals_act),
                    share_active_rows=np.mean(share)))
act = pd.DataFrame(act)
act.to_csv(RESULTS / "09_active_rows.csv", index=False)

# ------------------------------------------------------------------ report
show = ["dev_attackers", "dev_detected@85", "dev_range_3_repeats", "dev_latency_median", "dev_latency_p90",
        "dev_row_pr_auc", "test_attackers", "test_detected@85", "test_rate_95CI", "test_latency_median",
        "test_row_pr_auc", "small_cell"]
sections = "\n\n".join(f"### {title}\n{md_table(tables[c].set_index(c)[show])}"
                       for c, title in (("stealth_level", "By stealth level (injection multiplier)"),
                                        ("window_workdays", "By attack duration (workdays)"),
                                        ("switching_level", "By switching level"),
                                        ("channel_configuration", "By channel configuration")))
text = f"""# 09 — Breakdown by simulation dimension

Model: frozen LightGBM y_10 + Platt. Threshold for detection and latency: alert band risk > 85 (frozen).
**dev** = Stage A out-of-fold scores (348 attackers / 511 benign; mean of 3 repeats).
**test** = the single frozen test scoring (52 / 84). Test cells with fewer than 10 attackers are flagged; read them from dev.
`row_pr_auc` = y_10 average precision over the group's attackers' rows plus all benign rows.
Latency = workdays from the first injected day to the first in-window alarm.

{sections}

### Detection by duration x stealth (dev, share detected at 85, mean of 3 repeats)
{md_table(cross)}

Attackers per cell (dev): {counts.to_dict()}

### Attack days by injected channel set (share of attack days above the threshold)
{md_table(ch)}

### Attack days by same-day synchronisation
{md_table(sync)}

### Non-attack days: alarm rate by day type
{md_table(neg)}

`y10_positive_share`: forced off-days and early post-window days are y_10-positive by definition (an attack day in the
last 10 workdays); pre-window and benign rows never are.

### Active days only (audit M2: rows without any channel activity are easy negatives; about 26-28% of rows in the 995-user scope)
{md_table(act, index=False)}
"""
(REPORTS / "09_breakdown.md").write_text(text, encoding="utf-8")
print(text)
