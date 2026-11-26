"""
11 — Detection latency at the frozen thresholds (evaluation only).

Latency = workdays from the attacker's FIRST INJECTED day to the first in-window day whose calibrated risk is
above the threshold (0 = alarm on the first attack day; negative = alarm on a forced off-day that opens the
window, before any injection). Undetected attackers have no latency and are counted separately.

Sources: dev = Stage A OOF scores with the frozen Platt mapping (3 repeats pooled: each attacker appears 3 times);
test = the single frozen test scoring. Thresholds: risk > 85 (alert), > 60, > dev 1% user-FPR point (manifest).

Outputs: results/11_latency_*.csv, reports/11_latency.md
"""

import json

import numpy as np
import pandas as pd

from common import WORK, RESULTS, REPORTS, MODEL_DIR, EVAL_META, USER_META, FINAL_SPLIT, load_dataset, md_table

EPS = 1e-7
man = json.loads((MODEL_DIR / "model" / "manifest.json").read_text(encoding="utf-8"))
A, B = man["calibration"]["a"], man["calibration"]["b"]
THR = {"risk > 85 (alert)": 85.0, "risk > 60": 60.0, f"risk > {man['thresholds']['dev_1pct_user_fpr_risk']:.2f} (dev 1%)":
       man["thresholds"]["dev_1pct_user_fpr_risk"]}
STEALTH = {"High": "High (0.25)", "Medium": "Medium (0.50)", "Low": "Low (0.75)"}


def risk_of(p):
    p = np.clip(p, EPS, 1 - EPS)
    return 100.0 / (1.0 + np.exp(-(A * np.log(p / (1 - p)) + B)))


df = load_dataset()
meta = pd.read_parquet(EVAL_META)
um = pd.read_csv(USER_META).set_index("user")
split = pd.read_csv(FINAL_SPLIT)
base = pd.DataFrame({"user": df["user"].astype(str), "day": df["day"], "in_window": meta["in_window"],
                     "wd": meta["wd_from_first_attack"].astype(float)})
dev_mask = base["user"].isin(set(split.loc[split.part == "dev", "user"])).to_numpy()
oof = pd.read_parquet(WORK / "oof" / "lightgbm_h10.parquet")
dev_runs = [base[dev_mask].reset_index(drop=True).assign(risk=risk_of(oof[c].to_numpy()), run=c) for c in oof.columns]
ts = pd.read_parquet(WORK / "test" / "test_scores.parquet")
test_runs = [base.merge(ts[["user", "day", "risk"]], on=["user", "day"]).assign(run="test")]


def first_alarms(runs, t):
    rows = []
    for f in runs:
        win = f[f.in_window]
        first = win[win.risk > t].groupby("user")["wd"].min()
        for u in win["user"].unique():
            rows.append(dict(run=f["run"].iloc[0], user=u, latency=first.get(u, np.nan)))
    out = pd.DataFrame(rows)
    out["stealth"] = out["user"].map(um["stealth_level"]).map(STEALTH)
    out["duration"] = out["user"].map(um["window_workdays"]).astype(int)
    return out


def summary(x):
    d = x["latency"].dropna()
    return pd.Series({"attacker_runs": len(x), "detected": x["latency"].notna().mean(),
                      "p25": d.quantile(.25), "median": d.median(), "p75": d.quantile(.75), "p90": d.quantile(.9),
                      "max": d.max(), "on_day_0": (d == 0).mean() if len(d) else np.nan,
                      "within_3_days": (d <= 3).mean() if len(d) else np.nan,
                      "before_first_injection": int((d < 0).sum())})


tables = {}
for name, t in THR.items():
    for src, runs in (("dev", dev_runs), ("test", test_runs)):
        fa = first_alarms(runs, t)
        tables[(name, src, "overall")] = summary(fa).to_frame("all").T
        tables[(name, src, "stealth")] = fa.groupby("stealth").apply(summary)
        tables[(name, src, "duration")] = fa.groupby("duration").apply(summary)
        if src == "dev":
            tables[(name, src, "cross")] = fa.groupby(["duration", "stealth"]).apply(summary)["median"].unstack()
            tables[(name, src, "cross_det")] = fa.groupby(["duration", "stealth"]).apply(summary)["detected"].unstack()
for (name, src, kind), t_ in tables.items():
    tag = name.split(" ")[2].replace(".", "_")
    t_.to_csv(RESULTS / f"11_latency_{src}_{kind}_{tag}.csv")

main = list(THR)[0]
overall = pd.concat({f"{src} · {n}": tables[(n, src, "overall")].iloc[0] for n in THR for src in ("dev", "test")}, axis=1).T
text = f"""# 11 — Detection latency at the frozen thresholds

Latency = workdays from the first injected day to the first in-window alarm (0 = first attack day).
dev = Stage A OOF, 3 repeats pooled (348 attackers x 3 runs); test = single frozen scoring (52 attackers).
`detected` = share of attackers with any in-window alarm; latency percentiles are over detected attackers.

## Overall
{md_table(overall)}

## Alert threshold (risk > 85) by stealth
dev:
{md_table(tables[(main, 'dev', 'stealth')])}

test:
{md_table(tables[(main, 'test', 'stealth')])}

## Alert threshold (risk > 85) by attack duration
dev:
{md_table(tables[(main, 'dev', 'duration')])}

test (cells below 10 attackers are not reliable):
{md_table(tables[(main, 'test', 'duration')])}

## Dev, alert threshold: median latency by duration x stealth
{md_table(tables[(main, 'dev', 'cross')])}

Share detected in the same cells:
{md_table(tables[(main, 'dev', 'cross_det')])}

Per-threshold tables for risk > 60 and the dev 1% point: `results/11_latency_*`.
"""
(REPORTS / "11_latency.md").write_text(text, encoding="utf-8")
print(text)
