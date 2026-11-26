"""
02 — Locked test set and grouped cross-validation folds.

Groups: attackers that share an identical attack schedule
(`set_switch_bitmask|proposed_active_days|window_workdays`) form one group;
each benign user is its own group. A group never spans two parts.

Strata: attackers by duration x stealth; benign users by 2010 channel set (EU/EH/EUH).

Test (~15%): first fold of StratifiedGroupKFold(6, shuffle, seed 42) drawn only
from groups with no member in the legacy `val` split (legacy val users were used
for the feature decisions in audit §8.1/§8.4, so they stay in development).

Development: 5-fold x 3 repeats (seeds 42/43/44), grouped and stratified the same
way. Inside each fold, ~1/7 of the fold-training users (grouped) are held out
for early stopping (`es`); the rest are `fit`.

Outputs: config/final_split.csv, config/cv_folds.csv, reports/02_split.md
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import (SEED, CV_SEEDS, N_FOLDS, ES_FRACTION_SPLITS, USER_META, LABELS, FINAL_SPLIT, CV_FOLDS,
                    REPORTS, CHECKS, check, md_table)

TEST_SPLITS = 6

users = pd.read_csv(USER_META)
labels = pd.read_parquet(LABELS)

# ------------------------------------------------------------------ locked test
val_groups = set(users.loc[users["old_split"] == "val", "group"])
pool = users[~users["group"].isin(val_groups)].reset_index(drop=True)

# Large schedule groups (mostly 5-day windows) make a single grouped draw unbalanced, so the test
# draw is repeated over seeds SEED..SEED+N_SEED_SEARCH-1 and the draw whose composition is closest to
# TEST_TARGET in every category is kept. Only composition is used (no labels beyond attacker
# membership, no model scores), so the choice cannot favour any model.
TEST_TARGET = 0.15
N_SEED_SEARCH = 300
BALANCE_DIMS = {  # (column, population)
    "duration": ("window_workdays", 1), "stealth": ("stealth_level", 1), "switching": ("switching_level", 1),
    "channels": ("channel_configuration", 1), "benign_set": ("observed_channel_set", 0),
}


def balance_cost(test_set):
    is_test = users["user"].isin(test_set)
    devs = [is_test.mean() - TEST_TARGET]
    for col, pop in BALANCE_DIMS.values():
        sub = users[users["is_attacker"] == pop]
        devs += list(sub.groupby(col)["user"].apply(lambda s: s.isin(test_set).mean()) - TEST_TARGET)
    devs.append(users.loc[is_test, "is_attacker"].mean() - users["is_attacker"].mean())
    return float(np.sum(np.square(devs)))


best = None
for s in range(SEED, SEED + N_SEED_SEARCH):
    sgkf = StratifiedGroupKFold(n_splits=TEST_SPLITS, shuffle=True, random_state=s)
    _, idx = next(sgkf.split(pool, pool["stratum"], pool["group"]))
    cand = set(pool.loc[idx, "user"])
    cost = balance_cost(cand)
    if best is None or cost < best[0]:
        best = (cost, s, cand)
test_cost, test_seed, test_users = best
print(f"test draw: seed {test_seed}, balance cost {test_cost:.5f}")
users["part"] = np.where(users["user"].isin(test_users), "test", "dev")

dev = users[users["part"] == "dev"]
test = users[users["part"] == "test"]
check("Every user in exactly one part", users["part"].notna().all() and not users["user"].duplicated().any())
check("No group spans dev and test", not (set(dev["group"]) & set(test["group"])),
      len(set(dev["group"]) & set(test["group"])))
check("No legacy-val user in test", not (test["old_split"] == "val").any())
check("Test size 13-17% of users", 0.13 <= len(test) / len(users) <= 0.17, f"{len(test)} users ({len(test)/len(users):.1%})")
check("Test has every stealth level", test.loc[test.is_attacker == 1, "stealth_level"].nunique() == 3)

dur_share = users[users.is_attacker == 1].groupby("window_workdays")["part"].apply(lambda s: (s == "test").mean())
check("Every duration has 8-25% of its attackers in test", dur_share.between(0.08, 0.25).all(),
      dur_share.round(3).to_dict())
check("Test has every switching level", test.loc[test.is_attacker == 1, "switching_level"].nunique() == 3)
check("Test has every channel configuration", test.loc[test.is_attacker == 1, "channel_configuration"].nunique() == 4)
check("Test has every benign channel set", test.loc[test.is_attacker == 0, "observed_channel_set"].nunique() == 3)

# ------------------------------------------------------------------ CV folds on dev
dev = dev.reset_index(drop=True)
rows = []
for rep, seed in enumerate(CV_SEEDS):
    outer = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(outer.split(dev, dev["stratum"], dev["group"])):
        tr = dev.loc[tr_idx].reset_index(drop=True)
        inner = StratifiedGroupKFold(n_splits=ES_FRACTION_SPLITS, shuffle=True, random_state=seed + 1000 + fold)
        _, es_idx = next(inner.split(tr, tr["stratum"], tr["group"]))
        es_users = set(tr.loc[es_idx, "user"])
        for u in dev.loc[va_idx, "user"]:
            rows.append((u, rep, fold, "valid"))
        for u in tr["user"]:
            rows.append((u, rep, fold, "es" if u in es_users else "fit"))
folds = pd.DataFrame(rows, columns=["user", "repeat", "fold", "role"])

check("Each dev user has one role per (repeat, fold)",
      folds.groupby(["user", "repeat", "fold"]).size().eq(1).all() and
      folds.groupby(["repeat", "fold"])["user"].nunique().eq(len(dev)).all())
check("Valid folds partition dev in every repeat",
      folds[folds.role == "valid"].groupby("repeat")["user"].agg(lambda s: s.nunique() == len(s) == len(dev)).all())
grp = users.set_index("user")["group"]
bad = 0
for (rep, fold), f in folds.groupby(["repeat", "fold"]):
    g = {r: set(grp.loc[f.loc[f.role == r, "user"]]) for r in ("fit", "es", "valid")}
    bad += len(g["fit"] & g["es"]) + len(g["fit"] & g["valid"]) + len(g["es"] & g["valid"])
check("No group shared between fit / es / valid in any fold", bad == 0, bad)
check("No test user in CV folds", not folds["user"].isin(test_users).any())
es_share = folds[folds.role != "valid"].groupby(["repeat", "fold"])["role"].apply(lambda s: (s == "es").mean())
check("Early-stopping share 10-20% of fold-train users", es_share.between(0.10, 0.20).all(),
      f"{es_share.min():.3f}-{es_share.max():.3f}")

# ------------------------------------------------------------------ write
users[["user", "is_attacker", "group", "stratum", "old_split", "part"]].to_csv(FINAL_SPLIT, index=False)
folds.to_csv(CV_FOLDS, index=False)

lab = labels.merge(users[["user", "part"]], on="user")
row_tab = lab.groupby("part").agg(users=("user", "nunique"), rows=("y", "size"), attack_days=("y", "sum"),
                                  y_10_rate=("y_10", "mean"))
att = users[users.is_attacker == 1]


def mix(col):
    return att.pivot_table(index=col, columns="part", values="user", aggfunc="count", fill_value=0)


report = f"""# 02 — Split (locked test + grouped CV)

**{len(CHECKS)}/{len(CHECKS)} checks PASS.**

- Groups: attackers sharing an identical schedule form one group ({users.loc[users.is_attacker == 1, 'group'].nunique()} groups
  for 400 attackers); every benign user is its own group.
- Legacy-val users ({int((users.old_split == 'val').sum())}) and their groups ({len(val_groups)}) were excluded from the test draw.
- Test = fold 0 of StratifiedGroupKFold({TEST_SPLITS}); the seed was searched over {SEED}..{SEED + N_SEED_SEARCH - 1}
  and seed {test_seed} kept because its composition is closest to {TEST_TARGET:.0%} in every category
  (sum of squared share deviations = {test_cost:.5f}; composition only, no model scores). Dev CV = {N_FOLDS} folds x {len(CV_SEEDS)} repeats
  (seeds {CV_SEEDS}); early-stopping users = fold 0 of StratifiedGroupKFold({ES_FRACTION_SPLITS}) inside each fold-train set.

## Parts
{md_table(users.groupby(['part', 'is_attacker']).size().unstack().rename(columns={0: 'benign', 1: 'attackers'}))}

{md_table(row_tab)}

## Attackers by dimension
Stealth:
{md_table(mix('stealth_level'))}

Duration (workdays):
{md_table(mix('window_workdays'))}

Switching:
{md_table(mix('switching_level'))}

Channel configuration:
{md_table(mix('channel_configuration'))}

Benign users by 2010 channel set:
{md_table(users[users.is_attacker == 0].pivot_table(index='observed_channel_set', columns='part', values='user', aggfunc='count'))}

Legacy split of the new test users:
{md_table(test.groupby(['old_split', 'is_attacker']).size().unstack(fill_value=0))}

## Checks
""" + "\n".join(f"- [{'PASS' if ok else 'FAIL'}] {n} {d}" for n, ok, d in CHECKS) + "\n"
(REPORTS / "02_split.md").write_text(report, encoding="utf-8")
print(report)
