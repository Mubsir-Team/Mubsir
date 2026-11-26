# Stage A audit — part 2: short attacks, objectives, risk-score meaning, latency

Script: `audit/stageA_audit2.py` (dev OOF only; no test scores; no Optuna). Raw outputs: `results/audit_stageA2/*.csv`.
Split, labels, selection rules, features and training code are unchanged.
Model analysed: LightGBM, y_10 (Stage A selection). Threshold = 1% benign-user FPR, as in Stage A unless stated otherwise.

---

## 1. Missed 5-workday attackers

### 1.1 Consistency across repeats (`1_5day_detection_by_repeat.csv`)
| Repeat | Detected / 46 | Rate | Threshold (raw score) |
|---|---:|---:|---:|
| 0 | 21 | 45.7% | 0.933 |
| 1 | 23 | 50.0% | 0.927 |
| 2 | 23 | 50.0% | 0.927 |

The pattern is stable. **20 attackers are detected in all 3 repeats, 21 are missed in all 3, and 5 are unstable.** The 48.6% figure is not a fold artefact.

### 1.2 Who is missed
| Stealth | Detected 3/3 | Missed 3/3 | Unstable |
|---|---:|---:|---:|
| High (0.25) | 2 | **16** | 2 |
| Medium (0.50) | 9 | 5 | 2 |
| Low (0.75) | 9 | **0** | 1 |

- **Channel configuration:** E+U 4 / 9 / 1 · U+H 7 / 8 / 1 · E+H 4 / 2 / 0 · E+U+H 5 / 2 / 3. The misses lean towards USB-heavy configurations, which mostly coincide with High stealth.
- **Switching:** Low 11 / 13 / 1 · High 7 / 7 / 4 · Medium 2 / 1 / 0. No switching effect.
- **Pattern:** no 5-day attacker has a multi-channel day; all 46 inject one channel per day, so there is no same-day synchronisation to exploit.
- **Start month:** 36 of the 46 five-day windows start in January, with no pre-window history. This is the same for detected and missed attackers, so it is not a cause.

### 1.3 Missed vs detected (medians, `1_5day_missed_vs_detected_medians.csv`)
| Quantity | Detected 3/3 (20) | Missed 3/3 (21) | Benign users, p99 of their peak |
|---|---:|---:|---:|
| Total injected deviation in the window (Σ ΔE) | 6.0 | **2.8** | — |
| Max `cumulative_deviation_10wd` in the window | 8.9 | **5.5** | **10.2** |
| Max `cumulative_deviation_5wd` in the window | 8.3 | 4.4 | 7.5 |
| Max model score in the window (raw) | 0.985 | 0.628 | 0.926 |
| Mean events added per channel-day | 3.9 | 2.6 | — |
| Mean baseline P95 of injected channels | 6.7 | 7.4 | — |
| Max `nr_m_5wd` / `nr_alt_5wd` in the window | 3 / 0.75 | 2 / 1.0 | — |

### 1.4 Likely causes (no features or thresholds changed)
1. **The signal is below the benign noise floor.**
   - A missed 5-day attacker accumulates a median deviation of **5.5**. The noisiest 1% of benign users reach **10.2** on an ordinary day.
   - The 1% user-FPR threshold is set by those benign users (k = 5 of 511), so a 5-day, High-stealth attack is ranked below normal variation.
   - This is a separability limit of the attack as simulated, not a training failure.
2. **Stealth fixes the per-day signal.** Injected ΔE per day is about 0.5 (High), 1.0 (Medium) and 1.5 (Low), see audit part 1 §2.4. Five days of High stealth therefore add only about 2.5 in total.
   - Median Σ ΔE for 5-day attacks is 2.6 / 5.0 / 7.3 (High / Medium / Low), against 19.4 / 39.8 / 61.3 for 50-day attacks (`1_sum_injected_dE_by_duration_stealth.csv`).
3. **Smaller injected counts.** Missed attackers add 2.6 events per channel-day, against 3.9 for detected ones, on similar P95 levels.
4. **Rotation offers no compensation.** With 5 days, one channel per day and no overlap, rotation features carry little information. Missed attackers even have a higher `nr_alt_5wd`, but normal users also rotate (handover §2.7).

---

## 2. Optimisation objective (dev OOF, y_10, all five algorithms)

### 2.1 Why long attacks dominate the row objective (`2_positive_rows_by_duration.csv`)
| Duration | Attackers | Share of y_10-positive rows |
|---|---:|---:|
| 5 | 46 (13%) | **5.3%** |
| 10 | 60 | 9.0% |
| 20 | 60 | 13.9% |
| 30 | 60 | 18.6% |
| 40 | 61 | 23.9% |
| 50 | 61 | **29.3%** |

Row PR-AUC weights each attacker by the number of its positive rows. That favours long campaigns, and also repeated high scores on days after an attacker is already flagged.

Per-duration row PR-AUC (LightGBM, pooled): **0.50** (5 d) · 0.75 · 0.86 · 0.91 · 0.95 · 0.95 (50 d). The overall row value (0.944) hides the 5-day weakness almost completely.

### 2.2 Candidate objectives (15-fold mean ± sd, `2_objectives_fold_mean_sd.csv`)
| Objective | LightGBM | CatBoost | XGBoost | RF | DT |
|---|---|---|---|---|---|
| Row PR-AUC | 0.9439 ± .0046 | 0.9443 ± .0047 | 0.9428 ± .0041 | 0.9100 | 0.8771 |
| User PR-AUC (peak score, attacker vs benign) | 0.9847 ± .0041 | 0.9893 ± .0045 | 0.9847 ± .0050 | 0.9703 | 0.9363 |
| Duration-macro row PR-AUC | 0.8210 ± .0175 | 0.8271 ± .0147 | 0.8181 ± .0202 | 0.7503 | 0.6893 |
| Duration-macro user PR-AUC | 0.9311 ± .0229 | 0.9490 ± .0251 | 0.9322 ± .0262 | 0.8843 | 0.7439 |

Duration-macro = mean over the 6 duration groups; each group is its attackers plus all benign users.

**Paired fold differences (`2_objective_paired_signal_noise.csv`):**

| Comparison | Row PR-AUC | User PR-AUC | Duration-macro row | Duration-macro user |
|---|---|---|---|---|
| CatBoost − LightGBM | +0.0004, wins 9/15 | +0.0046, wins **13/15** | +0.0061, wins **13/15** | +0.0179, wins **13/15** |
| XGBoost − LightGBM | −0.0011, wins 2/15 | 0.0000, wins 7/15 | −0.0030, wins 6/15 | +0.0011, wins 8/15 |

**The objective changes the conclusion.**
- Row PR-AUC cannot distinguish CatBoost from LightGBM.
- Every per-employee or per-duration objective prefers CatBoost in 13 of 15 folds.
- This is reported only; the algorithm rule is not changed.

### 2.3 Strengths and weaknesses
| Objective | Strengths | Weaknesses |
|---|---|---|
| Row PR-AUC (handover) | most stable (sd 0.0046); rewards early in-window days | long attacks dominate (50 d = 29% of positives, 5 d = 5%); rewards re-flagging already-detected attackers; blind to the 5-day weakness |
| User PR-AUC | each employee counts once; matches "find the suspicious employee" | the peak over the whole period ignores timing (a late or post-window peak counts); near ceiling (0.985) so little room to optimise; 5-day attackers are still only 13% of the positives |
| Duration-macro row PR-AUC | each duration group weighs 1/6; keeps day-level timing (early in-window days must score high); directly exposes the short-attack weakness | about 4x noisier than row PR-AUC; the 5-day group has only about 9 attackers per fold; group PR-AUC depends on group prevalence (5-day has the fewest positives) |
| Duration-macro user PR-AUC | each duration equal and each employee once | noisiest (sd 0.023), coarse, and no timing |

### 2.4 Recommendation
The objective that best represents MUBSIR's goal (find the suspicious employee early, for every attack length) is the **duration-macro row PR-AUC on y_10**, computed on **pooled inner OOF scores** rather than as a per-fold mean, to reduce the noise of the small 5-day group.

It must be paired with guards:
- row PR-AUC must not fall by more than 0.002;
- dev user-level recall at a frozen 1% user-FPR threshold must not fall.

This is chosen on principle (equal weight per duration, timing preserved), not on scores. It would require updating `config/selection_rules.json` → `later_stages` **before** running Optuna, which needs your approval.

---

## 3. What the risk score means: three separate layers

### Layer 1: daily prediction (calibrated, per user-day)
- `risk(t) = 100 · P̂( at least one injected attack day among the user's last 10 workdays | features at day t )`
- Platt-calibrated at row level against **y_10**, estimated under the **dev population mix**: 13.5% positive rows, 40.5% of users are attackers.

**Why this is not P(employee is malicious):**
1. **It labels days, not people.** An attacker's own pre-window days (13,255 rows overall) and post-window days beyond 10 workdays are y_10 = 0. A malicious employee can correctly score near 0 on most days.
2. **Prevalence.** The calibration is relative to a dev set where 40.5% of users are attackers. In production the share is far smaller, so the same features would imply a much lower posterior. The number is calibrated only to the evaluation mix.
3. **Generator dependence.** It is calibrated to attacks produced by this simulator (audit part 1 §2.4), not to real insider behaviour.

### Layer 2: employee-level assessment (`risk_peak`, `first_alert_day`, `n_days`)
`risk_peak` = max of the daily risks over the monitored period. It is a **ranking statistic, not a calibrated probability:**
- It is a maximum over about 105 correlated daily values. At the dev 1% user-FPR threshold, a benign **row** alarms in 0.024% of rows, but a benign **user** alarms with probability 0.98%, about **40x higher** (`3_row_vs_user_alarm.csv`). If days were independent, the user rate would be 2.5%; alarms cluster within users.
- Its distribution depends on how many days are observed, so the same daily model gives different peak distributions for different observation lengths.
- An employee-level probability would need its own calibration (peak vs attacker, at production prevalence), which does not exist.

**Are observation periods comparable?** (`3_days_per_user.csv`, `3_benign_peak_vs_days.csv`)
- In this dataset, mostly yes: **500 of 511 benign dev users and almost all attackers have 105–107 workdays.**
- 11 benign users (2.2%) have 7–104 days (none alarmed); the shortest attacker has 28.
- In production this will not hold (new joiners, leave, rolling monitoring). The employee assessment should be defined over a fixed trailing period (for example, the last N workdays), with N reported next to `risk_peak`.

### Layer 3: alert generation (policy)
- Alert when the daily risk exceeds 85 (High) or 60 (Medium). These are fixed project decisions, not fitted.
- Its operating characteristics (user recall, user FPR, FP counts, latency) are properties of the policy, measured on dev OOF and later on test with dev-frozen settings. They are not model quality and not probabilities.

---

## 4. Latency

**Threshold provenance:**
- **Stage A:** derived from dev OOF benign users of the **same** repeat on which latency is measured. Dev only and test untouched, but in-sample for the threshold.
- **Here:** re-measured with a **frozen leave-fold-out** threshold. For each validation fold, the threshold comes only from the benign users of the other 4 folds and is then applied to that fold.
  - Realised FP: 6 / 5 / 9 of 511 benign users across the 3 repeats (1.2% / 1.0% / 1.8%), against the nominal 1%. This is the honest out-of-sample rate.

**By duration (frozen leave-fold-out; 3 repeats pooled; workdays from the first injected day):**

| Duration | Detected | P25 | Median | P75 | P90 |
|---|---:|---:|---:|---:|---:|
| 5 | **49.3%** | 0 | 1 | 3 | 4 |
| 10 | 83.9% | 1 | 3 | 6 | 8 |
| 20 | 90.0% | 2 | 4 | 7 | 13 |
| 30 | 93.9% | 1 | 4 | 7 | 9 |
| 40 | 99.5% | 1.25 | 3.5 | 6 | 11 |
| 50 | 96.7% | 1 | 3 | 7 | 13 |

**By stealth:**

| Stealth | Detected | P25 | Median | P75 | P90 |
|---|---:|---:|---:|---:|---:|
| High (0.25) | **67.0%** | 5 | **8** | 12 | **16** |
| Medium (0.50) | 95.1% | 2 | 4 | 6 | 8 |
| Low (0.75) | 99.4% | 0 | 1 | 3 | 5 |

The pooled in-sample threshold gives practically identical numbers (`4_latency_by_duration.csv`, `4_latency_by_stealth.csv`). Duration × stealth: `4_latency_by_duration_x_stealth.csv`.

- High stealth is the slower and weaker regime overall, not only for 5-day windows.
- When 5-day attacks are detected, it is fast (median 1 day), because detection only happens when the injected burst is large.

---

## 5. Recommendation for the next stage

1. **Before Optuna, pre-register Stage B rules in `config/selection_rules.json` (needs approval):**
   - Objective: duration-macro row PR-AUC on y_10 over pooled inner OOF (5 schedule-grouped inner folds on dev; 60 trials; default configuration enqueued as trial 0).
   - Adoption: Δ duration-macro row PR-AUC ≥ +0.005 **and** row PR-AUC ≥ default − 0.002 **and** dev user recall at the frozen leave-fold-out 1% user-FPR threshold ≥ default.
2. **Keep LightGBM** under the existing rule. Report CatBoost as a strong challenger and state plainly that it wins 13/15 folds on every per-employee or per-duration objective.
3. **Document the short and High-stealth limit as a property of the simulated attack**, not as a tuning target: the injected signal of 5-day High-stealth attacks is below the benign noise floor. Tuning may move the 5-day rate a few points, but should not be expected to close the gap. Feature work on this would be a separate, pre-registered experiment.
4. **Calibration as planned:** Platt on the stacked 3-repeat OOF scores, row level against y_10. Report it as Layer 1 only, with the three-layer wording above in all documentation.
5. **For test:** freeze all thresholds from dev; report FP counts out of 84 with exact CIs and the latency tables above.
