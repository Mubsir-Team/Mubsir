"""
01 — Horizon labels and evaluation context.

Labels (handover §2.6): y_h = 1 if at least one attack day (y == 1) falls within
the last h workdays of the same user, current day included. Rows of the model
dataset are the user's eligible workdays in order, so a trailing window of h rows
is a trailing window of h workdays. Backward-looking only.

Evaluation context (never predictors):
  work/eval_meta.parquet  per row: day_type (benign / pre_window / attack / offday /
                          post_window), in_window, wd_from_start, executed channels,
                          sync type, injected channel set
  config/user_meta.csv    per user: attacker flag, scenario attributes,
                          schedule signature, 2010 channel set, legacy split

Outputs: work/labels.parquet, work/eval_meta.parquet, config/user_meta.csv,
         reports/01_labels.md
"""

import numpy as np
import pandas as pd

from common import (HORIZONS, KEYS, LABELS, EVAL_META, USER_META, OLD_SPLIT, REPORTS, SRC_SCENARIOS,
                    SRC_CANDIDATES, SRC_GT_USER_DAYS, CHECKS, check, load_dataset, md_table)

df = load_dataset()[KEYS + ["y"]]
check("Dataset rows", len(df) == 103_395, len(df))
check("Dataset users", df["user"].nunique() == 995, df["user"].nunique())
check("(user, day) unique", not df.duplicated(KEYS).any())
check("Positive days", int(df["y"].sum()) == 9_080, int(df["y"].sum()))

# ------------------------------------------------------------------ labels
for h in HORIZONS:
    df[f"y_{h}"] = (df.groupby("user")["y"].transform(lambda s: s.rolling(h, min_periods=1).max())
                    .astype("int8"))
    check(f"y_{h} >= y on every row", (df[f"y_{h}"] >= df["y"]).all())
check("Nested horizons y_5 <= y_10 <= y_20", ((df["y_5"] <= df["y_10"]) & (df["y_10"] <= df["y_20"])).all())

# ------------------------------------------------------------------ per-user context
sc = pd.read_parquet(SRC_SCENARIOS)
sc["schedule_signature"] = (sc["set_switch_bitmask"].astype(str) + "|" + sc["proposed_active_days"].astype(str)
                            + "|" + sc["window_workdays"].astype(int).astype(str))
cand = pd.read_parquet(SRC_CANDIDATES, columns=["user", "observed_channel_set"])
old = pd.read_csv(OLD_SPLIT, usecols=["user", "split"]).rename(columns={"split": "old_split"})

users = pd.DataFrame({"user": sorted(df["user"].unique())})
users = users.merge(cand, on="user", how="left").merge(old, on="user", how="left")
users = users.merge(sc[["user", "window_workdays", "window_start", "window_end", "stealth_level", "stealth_multiplier",
                        "switching_level", "channel_configuration", "channel_count", "forced_offdays",
                        "scenario_type_id", "schedule_signature"]], on="user", how="left")
users["is_attacker"] = users["window_start"].notna().astype(int)
check("Attackers = 400", users["is_attacker"].sum() == 400, users["is_attacker"].sum())
check("Every user has a 2010 channel set", users["observed_channel_set"].notna().all(),
      users["observed_channel_set"].value_counts().to_dict())
check("Every user has a legacy split", users["old_split"].notna().all())
# Benign users form their own group; attackers are grouped by identical schedule.
users["group"] = np.where(users["is_attacker"] == 1, "S:" + users["schedule_signature"].astype(str),
                          "U:" + users["user"])
users["stratum"] = np.where(
    users["is_attacker"] == 1,
    "A_" + users["window_workdays"].astype("Int64").astype(str) + "wd_" + users["stealth_level"].astype(str),
    "B_" + users["observed_channel_set"].astype(str))

# ------------------------------------------------------------------ per-row context
gt = pd.read_parquet(SRC_GT_USER_DAYS, columns=["user", "day", "executed_channel_count", "sync_type",
                                                 "injected_channels", "active_day_index"])
gt["day"] = gt["day"].astype("datetime64[ms]")
meta = df[KEYS + ["y"]].merge(users[["user", "is_attacker", "window_start", "window_end"]], on="user", how="left")
meta = meta.merge(gt, on=KEYS, how="left")
ws = meta["window_start"].astype("datetime64[ms]")
we = meta["window_end"].astype("datetime64[ms]")
meta["in_window"] = (meta["is_attacker"] == 1) & (meta["day"] >= ws) & (meta["day"] <= we)
meta["day_type"] = np.select(
    [meta["is_attacker"] == 0, meta["y"] == 1, meta["in_window"], meta["day"] < ws],
    ["benign", "attack", "offday", "pre_window"], default="post_window")
pos = meta.groupby("user").cumcount()
start_pos = pos.where(meta["in_window"]).groupby(meta["user"]).transform("min")
meta["wd_from_start"] = (pos - start_pos).where(meta["in_window"]).astype("Int64")
# Latency is measured from the first injected (attack) day; some windows open with a forced off-day,
# so this can be negative for an alarm on such a leading off-day.
first_attack_pos = pos.where(meta["y"] == 1).groupby(meta["user"]).transform("min")
meta["wd_from_first_attack"] = (pos - first_attack_pos).where(meta["in_window"]).astype("Int64")

counts = meta["day_type"].value_counts().to_dict()
check("Attack rows = 9,080", counts.get("attack") == 9_080, counts)
check("Forced off-day rows = 1,670", counts.get("offday") == 1_670, counts.get("offday"))
check("In-window rows = 10,750", int(meta["in_window"].sum()) == 10_750, int(meta["in_window"].sum()))
check("Every attack row has GT channel info", meta.loc[meta["y"] == 1, "executed_channel_count"].notna().all())
check("No GT info outside attack rows", meta.loc[meta["y"] == 0, "executed_channel_count"].isna().all())
first_in_window = meta.loc[meta["wd_from_start"] == 0].set_index("user")["day"]
check("First in-window row is the scenario window_start",
      (first_in_window == users.set_index("user").loc[first_in_window.index, "window_start"].astype("datetime64[ms]")).all())
lead_off = int((meta.loc[meta["wd_from_start"] == 0, "day_type"] == "offday").sum())
check("Every attacker has a first attack day", meta.groupby("user")["wd_from_first_attack"].min().loc[
    users.loc[users.is_attacker == 1, "user"]].notna().all(), f"windows opening with an off-day: {lead_off}")
wlen = meta[meta["in_window"]].groupby("user").size()
wexp = users.set_index("user").loc[wlen.index, "window_workdays"]
check("Window length in rows = window_workdays (attack days + forced off-days)", (wlen == wexp).all(),
      f"mismatch users: {int((wlen != wexp).sum())}")
n_att = meta[meta["y"] == 1].groupby("user").size()
n_exp = (users.set_index("user")["window_workdays"] - users.set_index("user")["forced_offdays"]).loc[n_att.index]
check("Attack days per user = window_workdays - forced_offdays", (n_att == n_exp).all(),
      f"mismatch users: {int((n_att != n_exp).sum())}")

# ------------------------------------------------------------------ write
df.to_parquet(LABELS, index=False)
meta.drop(columns=["window_start", "window_end"]).to_parquet(EVAL_META, index=False)
users.to_csv(USER_META, index=False)

rates = pd.DataFrame({f"y_{h}": [int(df[f"y_{h}"].sum()), df[f"y_{h}"].mean()] for h in ["", *HORIZONS] if h != ""},
                     index=["positives", "rate"]).T
rates.loc["y (attack day)"] = [int(df["y"].sum()), df["y"].mean()]
by_type = meta.groupby("day_type").agg(rows=("y", "size")).join(
    df.assign(day_type=meta["day_type"]).groupby("day_type")[[f"y_{h}" for h in HORIZONS]].mean())

report = f"""# 01 — Labels and evaluation context

**{len(CHECKS)}/{len(CHECKS)} checks PASS.**

y_h = trailing max of y over the last h workdays (rows) of the same user, current day included.

## Positive rates (995 users, {len(df):,} rows)
{md_table(rates)}

## Share of positives by day type
{md_table(by_type)}

Off-days and the first days after a window become positive under y_h; benign-user and pre-window rows never do.

{lead_off} of 400 attack windows open with a forced off-day (allowed by the scenario design). Detection
latency is therefore measured from the first injected day (`wd_from_first_attack`), not from `window_start`.

## Groups and strata (for the split in 02)
- Attackers grouped by schedule signature (`set_switch_bitmask|proposed_active_days|window_workdays`):
  {users.loc[users.is_attacker == 1, 'group'].nunique()} groups for 400 attackers;
  largest group {users.loc[users.is_attacker == 1, 'group'].value_counts().max()} users.
- Benign users: one group each. Strata: attackers duration x stealth, benign users by 2010 channel set
  ({users.loc[users.is_attacker == 0, 'observed_channel_set'].value_counts().to_dict()}).

## Checks
""" + "\n".join(f"- [{'PASS' if ok else 'FAIL'}] {n} {d}" for n, ok, d in CHECKS) + "\n"
(REPORTS / "01_labels.md").write_text(report, encoding="utf-8")
print("\n", rates)
