# 01 — Labels and evaluation context

**20/20 checks PASS.**

y_h = trailing max of y over the last h workdays (rows) of the same user, current day included.

## Positive rates (995 users, 103,395 rows)
| index | positives | rate |
|---|---|---|
| y_5 | 12161.0000 | 0.1176 |
| y_10 | 13960.0000 | 0.1350 |
| y_20 | 17158.0000 | 0.1659 |
| y (attack day) | 9080.0000 | 0.0878 |

## Share of positives by day type
| day_type | rows | y_5 | y_10 | y_20 |
|---|---|---|---|---|
| attack | 9080 | 1.0000 | 1.0000 | 1.0000 |
| benign | 61753 | 0.0000 | 0.0000 | 0.0000 |
| offday | 1670 | 0.9611 | 0.9611 | 0.9611 |
| post_window | 17637 | 0.0837 | 0.1857 | 0.3670 |
| pre_window | 13255 | 0.0000 | 0.0000 | 0.0000 |

Off-days and the first days after a window become positive under y_h; benign-user and pre-window rows never do.

59 of 400 attack windows open with a forced off-day (allowed by the scenario design). Detection
latency is therefore measured from the first injected day (`wd_from_first_attack`), not from `window_start`.

## Groups and strata (for the split in 02)
- Attackers grouped by schedule signature (`set_switch_bitmask|proposed_active_days|window_workdays`):
  318 groups for 400 attackers;
  largest group 16 users.
- Benign users: one group each. Strata: attackers duration x stealth, benign users by 2010 channel set
  ({'EU': 311, 'EH': 160, 'EUH': 124}).

## Checks
- [PASS] Dataset rows 103395
- [PASS] Dataset users 995
- [PASS] (user, day) unique 
- [PASS] Positive days 9080
- [PASS] y_5 >= y on every row 
- [PASS] y_10 >= y on every row 
- [PASS] y_20 >= y on every row 
- [PASS] Nested horizons y_5 <= y_10 <= y_20 
- [PASS] Attackers = 400 400
- [PASS] Every user has a 2010 channel set {'EU': 428, 'EUH': 291, 'EH': 276}
- [PASS] Every user has a legacy split 
- [PASS] Attack rows = 9,080 {'benign': 61753, 'post_window': 17637, 'pre_window': 13255, 'attack': 9080, 'offday': 1670}
- [PASS] Forced off-day rows = 1,670 1670
- [PASS] In-window rows = 10,750 10750
- [PASS] Every attack row has GT channel info 
- [PASS] No GT info outside attack rows 
- [PASS] First in-window row is the scenario window_start 
- [PASS] Every attacker has a first attack day windows opening with an off-day: 59
- [PASS] Window length in rows = window_workdays (attack days + forced off-days) mismatch users: 0
- [PASS] Attack days per user = window_workdays - forced_offdays mismatch users: 0
