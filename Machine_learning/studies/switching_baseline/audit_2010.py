"""
Phase 0 audit of the switching-baseline study (read-only; no model, no test data).

Question for the audit: can each employee's habitual channel switching be measured in the FIXED 2010 baseline period, the
same period and the same deviation definition used for the per-channel volume baselines (P50/P95), and is it informative?

1. Re-derives how the 2010 baselines were computed (to compute E in 2010 exactly as in 2011).
2. For the 253 dev hosts with valid baselines on all 3 channels (the host pool of study_controlled_rotation; H146 subset
   flagged): 2010 deviation days, linked pairs, channel changes, switching rate, and the distribution (P50/P95) of the
   10-workday change count `rot_changes_10` over 2010.
3. Stability: 2010 habit vs the same host's 2011 pre-attack switching rate (unmodified history from the closed study's cases).
4. Shortcut check: does the 2010 habit itself separate attacker from benign users?
5. Position of the controlled ROT/BLK attacks relative to each host's 2010 habit.
Writes results/audit_* only.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent  # studies/<name>/ -> Model/ (moved 2026-11-26)
ROOT = MODEL_DIR.parent
sys.path.insert(0, str(HERE))
from rot_features import rotation_features, _linked_stats  # noqa: E402  (byte-identical copy, see EXPERIMENT_LOG)

RES = HERE / "results"
CNT = ["external_email_count", "usb_copy_count", "www_upload_count"]
BASE = ["email_count", "usb_copy", "http_upload"]
Ecols = ["E_email_today", "E_usb_today", "E_http_today"]

# ------------------------------------------------------------------ 1. baseline definition check
bl = pd.read_parquet(ROOT / "Data_engineering" / "data" / "features" / "user_baseline_r62_v1.parquet").set_index("user")
ud = pd.read_parquet(ROOT / "Data_engineering" / "data" / "processed" / "user_day_r62_v1.parquet",
                     columns=["user", "day", "is_weekend", "logon_count"] + CNT)
ud["user"] = ud["user"].astype(str)
ud["day"] = pd.to_datetime(ud["day"]).dt.normalize()
b10 = ud[(ud.day >= "2010-01-02") & (ud.day <= "2010-12-31") & (~ud.is_weekend.astype(bool)) & (ud.logon_count > 0)]
b10 = b10.sort_values(["user", "day"]).reset_index(drop=True)
defn = {}
defn["workdays_match"] = float((b10.groupby("user").size().reindex(bl.index) == bl["baseline_workdays"]).mean())
for c, b in zip(CNT, BASE):
    a = b10[b10[c] > 0].groupby("user")[c]
    for q, nm in ((0.5, "p50"), (0.95, "p95")):
        v = a.quantile(q).reindex(bl.index)
        defn[f"{b}_{nm}_match"] = float((np.isclose(v, bl[f"{b}_{nm}"]) | (v.isna() & bl[f"{b}_{nm}"].isna())).mean())
print("baseline definition (2010 weekday & logon>0; linear quantile over active days):", defn, flush=True)

# ------------------------------------------------------------------ 2. hosts and 2010 habit
split = pd.read_csv(MODEL_DIR / "config" / "final_split.csv")
um = pd.read_csv(MODEL_DIR / "config" / "user_meta.csv").set_index("user")
dev = sorted(set(split.loc[split.part == "dev", "user"].astype(str)))
P50 = bl[[f"{b}_p50" for b in BASE]].to_numpy(dtype=float)
P95 = bl[[f"{b}_p95" for b in BASE]].to_numpy(dtype=float)
pos = {u: i for i, u in enumerate(bl.index)}


def valid(u):
    i = pos[u]
    return bool(np.all(np.isfinite(P50[i]) & np.isfinite(P95[i]) & (P50[i] > 0) & (P95[i] >= P50[i])))


def is_int(v):
    return abs(v - round(v)) < 1e-9


hosts = [u for u in dev if u in pos and valid(u)]
H146 = {u for u in hosts if all(is_int(P50[pos[u], c]) and is_int(max(1.0, P95[pos[u], c] - P50[pos[u], c])) for c in range(3))}
assert len(hosts) == 253 and len(H146) == 146, (len(hosts), len(H146))


def E_of(x, i):
    return np.maximum(0.0, x - P50[i]) / np.maximum(1.0, P95[i] - P50[i])


def habit(E):
    dev_ = E.sum(axis=1) > 0
    d = np.flatnonzero(dev_)
    dom = np.full(len(E), -1)
    dom[dev_] = E[dev_].argmax(axis=1)
    linked, change = _linked_stats(d, dom)
    f = rotation_features(E)
    rc10 = f["rot_changes_10"][9:]                       # full 10-workday windows only
    return dict(workdays=len(E), dev_days=int(dev_.sum()), linked_pairs=int(linked.sum()), changes=int(change.sum()),
                rate=float(change.sum() / linked.sum()) if linked.sum() else np.nan,
                rc10_p50=float(np.percentile(rc10, 50)), rc10_p95=float(np.percentile(rc10, 95)),
                rc10_max=float(rc10.max()), share_windows_ge5=float((rc10 >= 5).mean()),
                multi_channel_dev_days=int(((E > 0).sum(axis=1) >= 2).sum()))


rows = []
g10 = {u: g for u, g in b10[b10.user.isin(hosts)].groupby("user")}
for u in hosts:
    E = E_of(g10[u][CNT].to_numpy(dtype=float), pos[u])
    rows.append(dict(user=u, H146=u in H146, is_attacker=int(um.loc[u, "is_attacker"]), **habit(E)))
H = pd.DataFrame(rows).set_index("user")
print(H.describe().T.round(3).to_string(), flush=True)

# ------------------------------------------------------------------ 3. stability vs 2011 pre-attack (unmodified) history
cases = pd.read_parquet(MODEL_DIR / "studies" / "controlled_rotation" / "data" / "counterfactual_cases.parquet",
                        columns=["case", "host", "cohort", "variant", "L", "row_index", "first_attack_index", "in_span"] + Ecols)
pre = cases[(cases.variant == "ROT") & (cases.L == 6) & cases.cohort.isin(["H146", "R253"])]
pre = pre[pre.row_index < pre.first_attack_index].sort_values(["host", "row_index"])
r11 = {}
for h, g in pre.groupby("host"):
    E = g[Ecols].to_numpy()
    dv = E.sum(axis=1) > 0
    dom = np.full(len(E), -1)
    dom[dv] = E[dv].argmax(axis=1)
    lk, ch = _linked_stats(np.flatnonzero(dv), dom)
    r11[h] = (ch.sum() / lk.sum()) if lk.sum() else np.nan
H["rate_2011_pre_attack"] = pd.Series(r11)
st = H.dropna(subset=["rate", "rate_2011_pre_attack"])
rho = spearmanr(st["rate"], st["rate_2011_pre_attack"])

# ------------------------------------------------------------------ 4. shortcut check (habit alone vs attacker label)
auc = roc_auc_score(H["is_attacker"], H["rate"].fillna(H["rate"].median()))
auc_rc = roc_auc_score(H["is_attacker"], H["rc10_p95"])

# ------------------------------------------------------------------ 5. controlled attacks vs 2010 habit (H146)
att = []
c146 = cases[(cases.cohort == "H146") & cases.variant.isin(["ROT", "BLK"])].sort_values(["case", "row_index"])
for cs, g in c146.groupby("case"):
    h = g["host"].iloc[0]
    f = rotation_features(g[Ecols].to_numpy())
    peak = float(f["rot_changes_10"][g["in_span"].to_numpy()].max())
    p50, p95 = H.loc[h, "rc10_p50"], H.loc[h, "rc10_p95"]
    att.append(dict(case=cs, host=h, variant=g["variant"].iloc[0], L=int(g["L"].iloc[0]), peak_rot_changes_10=peak,
                    habit_p50=p50, habit_p95=p95, E_switch=max(0.0, peak - p50) / max(1.0, p95 - p50),
                    above_habit_p95=peak > p95))
A = pd.DataFrame(att)
A["habit_tertile"] = pd.qcut(A["host"].map(H["rate"]), 3, labels=["low", "mid", "high"])
summ5 = A.groupby(["variant", "L"]).agg(peak=("peak_rot_changes_10", "mean"), E_switch=("E_switch", "mean"),
                                         share_above_p95=("above_habit_p95", "mean")).round(3)
by_t = A.groupby(["variant", "L", "habit_tertile"], observed=True).agg(
    E_switch=("E_switch", "mean"), share_above_p95=("above_habit_p95", "mean"), hosts=("host", "nunique")).round(3)
print(summ5.to_string(), "\n", by_t.to_string(), flush=True)

# ------------------------------------------------------------------ save
H.to_csv(RES / "audit_habit_2010_per_host.csv")
A.to_csv(RES / "audit_attacks_vs_habit.csv", index=False)
by_t.reset_index().to_csv(RES / "audit_attacks_vs_habit_by_tertile.csv", index=False)
h146 = H[H.H146]
q = lambda s, p: float(np.nanpercentile(s, p))  # noqa: E731
out = dict(
    baseline_definition=defn,
    hosts=dict(all=len(H), H146=int(H.H146.sum()), attackers=int(H.is_attacker.sum())),
    habit_2010_H146=dict(
        workdays_median=q(h146.workdays, 50), dev_days_median=q(h146.dev_days, 50), dev_days_min=float(h146.dev_days.min()),
        linked_pairs_p05=q(h146.linked_pairs, 5), linked_pairs_median=q(h146.linked_pairs, 50),
        linked_pairs_min=float(h146.linked_pairs.min()), hosts_lt_20_pairs=int((h146.linked_pairs < 20).sum()),
        hosts_lt_50_pairs=int((h146.linked_pairs < 50).sum()),
        rate_p10=q(h146.rate, 10), rate_tertiles=[q(h146.rate, 33.33), q(h146.rate, 66.67)], rate_p90=q(h146.rate, 90),
        rc10_p50_values=h146.rc10_p50.value_counts().sort_index().to_dict(),
        rc10_p95_values=h146.rc10_p95.value_counts().sort_index().to_dict()),
    stability_2010_vs_2011_pre_attack=dict(spearman=float(rho.statistic), p=float(rho.pvalue), hosts=int(len(st))),
    shortcut=dict(auc_rate_2010=float(max(auc, 1 - auc)), auc_rc10_p95=float(max(auc_rc, 1 - auc_rc))),
    controlled_attacks_vs_habit={f"{v}|L{L}": r for (v, L), r in summ5.to_dict("index").items()})
(RES / "audit_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("habit_2010_H146", "stability_2010_vs_2011_pre_attack", "shortcut")}, indent=1, default=float))
