# Switching-baseline study — Phase 0 audit

**Date:** 2026-11-26. Read-only; no model trained; test data never read. Code: `audit_2010.py`.
Outputs: `results/audit_*`.

## Why this study
The closed study `study_controlled_rotation/` found no independent switching signal. Its only "habit" features
(`sw_excess_*`) did not answer the question this study asks:
- **They compared with the wrong reference.** They compared switching with a rolling 2011 window (t−70 … t−11), not with
  the fixed 2010 baseline period that defines the volume deviation E (P50/P95). The two deviations were therefore not
  defined the same way, and late in the detection span the reference window overlapped the attack.
- **They were never tested alone.** They were bundled with 8 raw switching features; no arm isolated them.
- **They were never split by habit.** No analysis compared habitual switchers with non-switchers.

**Question:** is channel switching that deviates from an employee's own **2010** switching habit an independent detection
signal? In particular, is the switching signal larger for employees who rarely switch than for habitual switchers?

## 1. The 2010 baseline definition can be reproduced exactly
For all 3,995 users, recomputation matches `user_baseline_r62_v1` exactly (match rate 1.000 for the workday count and for
P50/P95 of all three channels). The definition is:
- **Workdays:** 2010-01-02 … 2010-12-31, weekday with logon_count > 0.
- **P50 / P95:** linear quantiles of the daily count over **active days** (count > 0).

So the daily deviation E_c = max(0, x − P50)/max(1, P95 − P50) can be computed on 2010 days exactly as on 2011 days, and a
switching habit can be measured in the same fixed period and with the same deviation definition. Daily 2010 counts are in
`Data_engineering/data/processed/user_day_r62_v1.parquet`.

## 2. The 2010 switching habit can be measured for nearly every host
Measured on the 253 dev hosts with valid baselines on all three channels; the H146 subset is used for exact matching.

| Measure (H146) | Value |
|---|---|
| 2010 workdays | median 251 |
| Deviation days (Σ_c E_c > 0) | median 85.5, min 13 |
| Linked pairs (≤ 2 quiet days apart) | median 60, 5th percentile 9, min 1 |
| Hosts with < 20 linked pairs (noisy rate) | 22 of 146 |
| Hosts with < 50 linked pairs | 56 of 146 |
| Switching rate (changes / linked pairs) | 10th percentile 0.28; tertile cuts 0.45 / 0.57; 90th percentile 0.66 |
| `rot_changes_10` P50 over 2010 | 0 (59 hosts), 1 (49), 2 (24), 3 (13) |
| `rot_changes_10` P95 over 2010 | 0–6, median 3 |
| 10-day windows with ≥ 5 changes | 2.6% on average |

**Feasibility:** yes for about 85% of hosts. A minimum number of linked pairs must be fixed in advance (for example ≥ 20,
which keeps 124 of the 146 hosts).

## 3. Habit is only moderately stable
- The 2010 switching rate and the same host's 2011 pre-attack switching rate have Spearman ρ = **0.37** (244 hosts,
  p = 2e-9).
- Switching habit is real but noisy, and a 2010 baseline is partly out of date by 2011. The volume baselines have the same
  limitation.

## 4. Shortcut risk: the attacker-selection process left a trace in 2010 activity
| 2010 measure alone, attacker vs benign host (folded AUC) | AUC |
|---|---:|
| switching **rate** (changes / linked pairs) | **0.58** |
| P95 of `rot_changes_10` (switching **count** spread) | **0.74** |
| deviation days | 0.76 |
| linked pairs | 0.76 |
| multi-channel deviation days | 0.75 |

Users later chosen as attackers were already more active across channels in 2010: median 73.5 linked pairs vs 39 for
benign hosts. Their switching **rate** is almost the same (0.53 vs 0.51).

**Consequence:** a habit measured with **counts** (P50/P95 of the number of changes) partly encodes "this user was selected
as an attacker". A model could learn that identity instead of attack-time switching. A habit measured as a **rate** is far
less exposed. A shortcut gate on the final features is needed, as in the closed study.

## 5. How the controlled attacks compare with the 2010 habit
This uses the closed study's matched H146 cases (read-only).
- The peak `rot_changes_10` in the detection span is compared with each host's 2010 P95.
- The habit tertile is the 2010 switching rate.

| Variant | L | Mean peak changes (10 days) | Share of hosts above their own 2010 P95 | E_switch: low / mid / high-habit tertile |
|---|---:|---:|---:|---|
| ROT | 6 | 6.1 | 94.5% | 3.9 / 2.4 / 2.2 |
| ROT | 12 | 8.6 | 100% | 6.2 / 3.6 / 3.1 |
| BLK | 6 | 3.7 | 57.2% | 2.0 / 1.2 / 1.2 |
| BLK | 12 | 3.3 | 43.4% | 1.9 / 1.1 / 1.0 |

`E_switch = max(0, peak − P50) / max(1, P95 − P50)`. Peaks include natural deviation days around the attack.

What this shows:
- **Habit changes the size of the excess as hypothesised.** For the same ROT attack, the excess is about twice as large for
  low-habit hosts as for high-habit hosts.
- **The contrast is limited, though.** The injected rotation is far above almost everyone's habit: ROT exceeds the host's
  own 2010 P95 for 88–100% of hosts, even in the high-habit tertile. Only about 2.6% of 2010 windows have ≥ 5 changes. So
  the existing ROT/BLK pair cannot separate "switching that is unusual for this person" from "switching that is unusual for
  anyone".
- **Implication for the closed study's null result.** Even raw switching features could flag ROT as extreme for almost
  every host. That makes it more likely that the bottleneck there was redundancy with aggregated deviation, not the lack of
  a personal reference. This is an expectation, not a result; the new study tests it.

## 6. Proposed design — SUPERSEDED by PLAN.md

> Owner decision (2026-11-26): no new attack data may be generated (no MIX3/MIX4 variants). The revised plan uses only
> 2010 activity, existing pre-attack activity, the existing Dev simulation and the closed study's existing cases. See
> `PLAN.md`. Sections 6-7 below are kept as the record of the original proposal.

1. **Habit (fixed, 2010):** per host, from the 2010 baseline workdays:
   - the switching rate r₀ (changes / linked pairs);
   - P50 and P95 of `rot_changes_10`, reported only.
   - Hosts need ≥ 20 linked pairs in 2010.
2. **New features (habit-relative, causal):**
   - `sw_dev_rate_10 = max(0, rate_10 − r₀)`;
   - `sw_dev_excess_10 = changes_10 − r₀ × linked_pairs_10`;
   - and the same over 20 days.
   - Rate-based, to avoid the count shortcut in §4.
3. **Arms:**
   - C+ (87, reference);
   - **H** = C+ + habit-relative features (91);
   - **R** = C+ + the same switching quantities without a habit reference (91).
   - Comparing H with R answers whether the comparison with the habit adds anything.
4. **Variants with a switching gradient:** on the same hosts, days and per-channel deviation as the closed study, generate
   orders with 2, 3, 4 and 5 changes (BLK, MIX3, MIX4, ROT). A 3-change attack is unusual for low-habit hosts and normal
   for high-habit ones, which is the contrast the question needs. New data generation takes about 15 minutes.
5. **Primary hypothesis (the question itself):** the gain of H over C+ on high-switch orders (ROT, MIX4) compared with
   BLK is larger for low-habit hosts (bottom tertile of r₀) than for high-habit hosts (top tertile). Secondary: H vs R.
6. **Gates:**
   - feature unit check on toy sequences;
   - shortcut gate: user-level AUC of the habit-relative features on pre-attack rows must be < 0.60;
   - minimum host count.
7. **Power, stated in advance:** each tertile has about 40–48 hosts, so only large effects (about ≥ 5–6 points) can be
   detected. A null result will be reported as "no large effect", not as "no effect".

## 7. Decisions needed before pre-registration
1. Habit measured as a **rate** (recommended; AUC 0.58) or as **count percentiles** (AUC 0.74 shortcut risk)?
2. Generate the **switching-gradient variants** (MIX3/MIX4; recommended), or reuse only ROT/BLK (limited contrast, §5)?
3. Primary hypothesis: **habit interaction** (low- vs high-habit hosts; recommended, the question itself) or **H vs R**?
