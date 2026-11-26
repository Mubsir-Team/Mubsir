"""
Read-only population check for the revised plan (PLAN.md). No model, no data generation, dev users only.
For every dev user: 2010 switching habit (rate = changes / linked pairs of deviation days, E with the contract rule used by
E_*_today: channels without a baseline contribute E = 0), eligibility (>= 20 linked pairs), and the user-level AUC of the
habit alone vs the attacker label (shortcut screen). Also counts the closed study's cases whose host is habit-eligible.
Writes results/population_check.json and results/habit_2010_dev_users.csv.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent  # studies/<name>/ -> Model/ (moved 2026-11-26)
ROOT = MODEL_DIR.parent
sys.path.insert(0, str(HERE))
from rot_features import _linked_stats  # noqa: E402

MIN_PAIRS = 20
CNT = ["external_email_count", "usb_copy_count", "www_upload_count"]
BASE = ["email_count", "usb_copy", "http_upload"]

split = pd.read_csv(MODEL_DIR / "config" / "final_split.csv")
dev = set(split.loc[split.part == "dev", "user"].astype(str))
um = pd.read_csv(MODEL_DIR / "config" / "user_meta.csv").set_index("user").loc[sorted(dev)]
bl = pd.read_parquet(ROOT / "Data_engineering" / "data" / "features" / "user_baseline_r62_v1.parquet")
bl = bl[bl.user.astype(str).isin(dev)].set_index("user")
ud = pd.read_parquet(ROOT / "Data_engineering" / "data" / "processed" / "user_day_r62_v1.parquet",
                     columns=["user", "day", "is_weekend", "logon_count"] + CNT)
ud = ud[ud.user.astype(str).isin(dev)]
ud["day"] = pd.to_datetime(ud["day"])
b10 = ud[(ud.day >= "2010-01-02") & (ud.day <= "2010-12-31") & (~ud.is_weekend.astype(bool)) & (ud.logon_count > 0)]
b10 = b10.sort_values(["user", "day"])

rows = []
for u, g in b10.groupby("user"):
    p50 = bl.loc[u, [f"{b}_p50" for b in BASE]].to_numpy(dtype=float)
    p95 = bl.loc[u, [f"{b}_p95" for b in BASE]].to_numpy(dtype=float)
    x = g[CNT].to_numpy(dtype=float)
    E = np.zeros_like(x)
    ok = np.isfinite(p50) & np.isfinite(p95)
    for c in range(3):
        if ok[c]:
            E[:, c] = np.maximum(0.0, x[:, c] - p50[c]) / max(1.0, p95[c] - p50[c])
    dv = E.sum(axis=1) > 0
    dom = np.full(len(E), -1)
    dom[dv] = E[dv].argmax(axis=1)
    lk, ch = _linked_stats(np.flatnonzero(dv), dom)
    rows.append(dict(user=u, baseline_channels=int(ok.sum()), dev_days=int(dv.sum()), linked_pairs=int(lk.sum()),
                     changes=int(ch.sum()), rate=float(ch.sum() / lk.sum()) if lk.sum() else np.nan))
H = pd.DataFrame(rows).set_index("user").join(um[["is_attacker", "observed_channel_set", "switching_level", "channel_count"]])
H["eligible"] = (H.linked_pairs >= MIN_PAIRS) & (H.baseline_channels >= 2)
H.to_csv(HERE / "results" / "habit_2010_dev_users.csv")

e = H[H.eligible]
auc = roc_auc_score(e.is_attacker, e.rate)
cases = pd.read_parquet(MODEL_DIR / "studies" / "controlled_rotation" / "data" / "counterfactual_cases.parquet",
                        columns=["case", "host", "cohort", "variant"]).drop_duplicates("case")
cases["eligible_host"] = cases.host.map(H.eligible).fillna(False).astype(bool)
cc = cases.groupby(["cohort", "variant"]).agg(cases=("case", "size"), eligible_cases=("eligible_host", "sum")).reset_index()
ch = cases.drop_duplicates(["cohort", "host"]).groupby("cohort").agg(hosts=("host", "size"), eligible_hosts=("eligible_host", "sum"))
out = dict(
    min_linked_pairs=MIN_PAIRS,
    dev_users=len(H), dev_attackers=int(H.is_attacker.sum()),
    eligible_users=int(H.eligible.sum()), eligible_attackers=int(e.is_attacker.sum()), eligible_benign=int((e.is_attacker == 0).sum()),
    eligible_by_channel_set=H.groupby("observed_channel_set").eligible.agg(["size", "sum"]).to_dict("index"),
    eligible_attackers_by_switching_level=H[H.is_attacker == 1].groupby("switching_level").eligible.agg(["size", "sum"]).to_dict("index"),
    eligible_attackers_by_channel_count=H[H.is_attacker == 1].groupby("channel_count").eligible.agg(["size", "sum"]).to_dict("index"),
    baseline_channels=H.baseline_channels.value_counts().to_dict(),
    habit_rate_eligible=dict(p10=float(e.rate.quantile(.1)), tertiles=[float(e.rate.quantile(1 / 3)), float(e.rate.quantile(2 / 3))],
                             p90=float(e.rate.quantile(.9)), median_benign=float(e[e.is_attacker == 0].rate.median()),
                             median_attacker=float(e[e.is_attacker == 1].rate.median())),
    shortcut_auc_rate_folded=float(max(auc, 1 - auc)),
    linked_pairs_median=dict(benign=float(e[e.is_attacker == 0].linked_pairs.median()), attacker=float(e[e.is_attacker == 1].linked_pairs.median())),
    closed_study_hosts=ch.to_dict("index"), closed_study_cases=cc.to_dict("records"))
(HERE / "results" / "population_check.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(json.dumps(out, indent=1, default=float))
