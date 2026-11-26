# 02 — Split (locked test + grouped CV)

**14/14 checks PASS.**

- Groups: attackers sharing an identical schedule form one group (318 groups
  for 400 attackers); every benign user is its own group.
- Legacy-val users (147) and their groups (142) were excluded from the test draw.
- Test = fold 0 of StratifiedGroupKFold(6); the seed was searched over 42..341
  and seed 175 kept because its composition is closest to 15% in every category
  (sum of squared share deviations = 0.01264; composition only, no model scores). Dev CV = 5 folds x 3 repeats
  (seeds (42, 43, 44)); early-stopping users = fold 0 of StratifiedGroupKFold(7) inside each fold-train set.

## Parts
| part | benign | attackers |
|---|---|---|
| dev | 511 | 348 |
| test | 84 | 52 |

| part | users | rows | attack_days | y_10_rate |
|---|---|---|---|---|
| dev | 859 | 89471 | 7878 | 0.1354 |
| test | 136 | 13924 | 1202 | 0.1328 |

## Attackers by dimension
Stealth:
| stealth_level | dev | test |
|---|---|---|
| High | 117 | 17 |
| Low | 115 | 18 |
| Medium | 116 | 17 |

Duration (workdays):
| window_workdays | dev | test |
|---|---|---|
| 5.0000 | 46 | 4 |
| 10.0000 | 60 | 10 |
| 20.0000 | 60 | 10 |
| 30.0000 | 60 | 10 |
| 40.0000 | 61 | 9 |
| 50.0000 | 61 | 9 |

Switching:
| switching_level | dev | test |
|---|---|---|
| High | 115 | 18 |
| Low | 117 | 16 |
| Medium | 116 | 18 |

Channel configuration:
| channel_configuration | dev | test |
|---|---|---|
| E+H | 104 | 13 |
| E+U | 101 | 16 |
| E+U+H | 119 | 20 |
| U+H | 24 | 3 |

Benign users by 2010 channel set:
| observed_channel_set | dev | test |
|---|---|---|
| EH | 137 | 23 |
| EU | 267 | 44 |
| EUH | 107 | 17 |

Legacy split of the new test users:
| old_split | 0 | 1 |
|---|---|---|
| test | 17 | 11 |
| train | 67 | 41 |

## Checks
- [PASS] Every user in exactly one part 
- [PASS] No group spans dev and test 0
- [PASS] No legacy-val user in test 
- [PASS] Test size 13-17% of users 136 users (13.7%)
- [PASS] Test has every stealth level 
- [PASS] Every duration has 8-25% of its attackers in test {5.0: 0.08, 10.0: 0.143, 20.0: 0.143, 30.0: 0.143, 40.0: 0.129, 50.0: 0.129}
- [PASS] Test has every switching level 
- [PASS] Test has every channel configuration 
- [PASS] Test has every benign channel set 
- [PASS] Each dev user has one role per (repeat, fold) 
- [PASS] Valid folds partition dev in every repeat 
- [PASS] No group shared between fit / es / valid in any fold 0
- [PASS] No test user in CV folds 
- [PASS] Early-stopping share 10-20% of fold-train users 0.142-0.148
