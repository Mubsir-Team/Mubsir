# Stage A — Methodological audit

Script: `audit/stageA_audit.py` (read-only with respect to the pipeline; dev users only; no test scores exist).
Raw outputs: `results/audit_stageA/*.csv`. Nothing in the selection rules, split, labels, features or training code was changed.

---

## 1. CatBoost vs LightGBM

### 1.1 How the 15 fold x 3 repeat results were aggregated
| Metric family | Unit | Aggregation |
|---|---|---|
| Row PR-AUC, ROC-AUC, Recall@1%/5%FPR (on y_h) | each of the 15 validation folds separately | mean (and sd) of the 15 fold values |
| User AUC, User Recall@1%/5% user-FPR, latency | pooled out-of-fold (OOF) scores of one repeat | computed once per repeat, then mean of the 3 repeats |

- Each repeat is a full 5-fold partition of the 859 dev users, so every dev row receives **exactly one OOF score per repeat** (3 scores in total, from 3 different models).
- The three repeated predictions are **never averaged together**. Each repeat is evaluated as a separate, complete OOF set; only the resulting metrics are averaged. Averaging scores across repeats would produce an ensemble-like, smoother score that no deployed model produces.
- Row-level Recall@FPR uses a per-fold threshold; user-level metrics use a pooled threshold per repeat (§4).

### 1.2 Paired bootstrap (2,000 resamples; CatBoost minus LightGBM; mean over the 3 repeats inside each resample)
Resampling is stratified (attackers and benign users separately). The same resampled users are applied to both models and all 3 repeats (paired). For User Recall, the 1% threshold is re-derived inside every resample. A second scheme resamples whole attacker schedule-groups (cluster bootstrap).

| h | Metric | LightGBM | CatBoost | Diff | 95% CI (user) | 95% CI (group) |
|---|---|---:|---:|---:|---|---|
| 5 | User AUC | 0.9884 | 0.9915 | +0.0031 | [+0.0012, +0.0051] | [+0.0009, +0.0057] |
| 5 | User Recall@1% | 0.8640 | 0.8860 | +0.0220 | [−0.0057, +0.0460] | [−0.0061, +0.0504] |
| **10** | **User AUC** | 0.9873 | 0.9914 | **+0.0041** | **[+0.0019, +0.0063]** | [+0.0016, +0.0067] |
| **10** | **User Recall@1%** | 0.8707 | 0.8841 | **+0.0134** | **[−0.0096, +0.0450]** | [−0.0085, +0.0471] |
| 20 | User AUC | 0.9849 | 0.9890 | +0.0041 | [+0.0019, +0.0067] | [+0.0013, +0.0071] |
| 20 | User Recall@1% | 0.8506 | 0.8716 | +0.0211 | [−0.0144, +0.0489] | [−0.0169, +0.0556] |

Paired row metrics per fold (`1_fold_paired_row_metrics.csv`): at h=10, CatBoost Recall@1%FPR is higher in **14 of 15 folds** (mean +0.0078, sd 0.0060). At h=5 it is higher in 8/15 (+0.0010) and at h=20 in 10/15 (+0.0019). PR-AUC differences are ≤ 0.0004 in every horizon.

**Reading:**
- The User AUC advantage of CatBoost is small but systematic: the CI excludes 0 in every horizon and under both schemes.
- The User Recall@1% difference (about 4–5 attackers out of 348) is **not significant**; every CI contains 0.
- These CIs cover user sampling only, given the trained models. Training randomness is only partly reflected, through the 3 repeats.

**Decision (unchanged):** LightGBM stays primary under the pre-declared rule A, and CatBoost is reported as a strong challenger.

**Concern (rule design, not changed):** rule A averages over h = 5/10/20 because it was written before the horizon was chosen. The production model uses only h=10, and at h=10 alone CatBoost's row Recall@1% gain (+0.0078) would exceed the +0.005 threshold. This is documented, not acted on.

---

## 2. `cumulative_deviation_10wd`

### 2.1 Calculation (NB12 §12.5.2, `rolling_sum_1252`; verified here independently)
```
E_c(t) = max(0, x_c(t) − P50_c) / max(1, P95_c − P50_c)     if P50_c, P95_c finite, P50_c > 0, P95_c > 0, P95_c ≥ P50_c
       = 0                                                   otherwise
daily_deviation(t) = E_email(t) + E_usb(t) + E_http(t)
cumulative_deviation_10wd(t) = Σ daily_deviation over the user's eligible workdays t−9 … t   (partial window at the start of 2011)
```
Source columns:
- Daily counts: `external_email_count`, `usb_copy_count`, `www_upload_count`, all of day t and earlier.
- Fixed 2010 baseline: `email_count_p50/p95`, `usb_copy_p50/p95`, `http_upload_p50/p95`.

NB12 asserts that `y` is absent from the frame used for feature calculation. The function receives only counts and baseline values. No ground-truth, schedule, window or scenario column is an argument.

### 2.2 Evidence (`2_cumulative_deviation_checks.csv`)
| Check | Result |
|---|---|
| Independent explicit-loop recomputation vs source column | max diff 2.8e-14 |
| Model dataset column vs source column | 0.0 |
| Baseline P50/P95 constant per user, and identical before/after injection | True / True |
| **Truncation test:** remove all rows after a random day (60 users); recompute | 0 values changed → no future rows used |
| **Counterfactual:** same rows recomputed from the pre-injection file (`behavior_features_r62_v1`) | |
| — rows whose value changed due to injection | 13,913 |
| — of these, rows with **no** injected day in the trailing 10 rows | **0** |
| — changed rows before an attacker's first injected day | **0** |
| — changed benign-user rows | **0** |
| — change minus trailing sum of the injected days' ΔE | 2.8e-14 (the change is fully explained by injected days ≤ t) |

So the only route from the attack to this feature is the **injected events themselves**, which are the simulated behaviour, and only after they occur. There is no direct or indirect path from the schedule, future attack dates, labels or window metadata.

### 2.3 Why AUC 0.944 is a behavioural signal (measured, not assumed)
| Same dev rows, same y_10 labels | Injected | Counterfactual (no injection) |
|---|---:|---:|
| All dev users | **0.944** | 0.601 |
| Attackers' rows only | **0.923** | **0.500** |

- Within attackers, the feature has zero separating power without the injected events (0.500). Its whole within-attacker signal is the exfiltration activity.
- The residual 0.601 on all users comes from attackers' **normal** behaviour differing from benign users. This is a population/selection effect (a residual of C2), not leakage, and it is part of what user-level metrics reward.
- By stealth (`2_cumulative_deviation_by_stealth.csv`): AUC 0.897 (High) / 0.961 (Medium) / 0.981 (Low). The injected share of the feature on positive rows is 52% / 69% / 79%.

### 2.4 Concern: simulator–feature coupling
The injection rule `I = max(1, ceil(m × P95))` is defined in units of the same 2010 P95 that E uses as its scale. The injected ΔE per attack day is therefore almost fixed by the stealth multiplier: median **0.5 / 1.0 / 1.5** for High / Medium / Low (`2_injected_dE_per_attack_day.csv`). The feature reads the simulator's own parameter back.

This is legitimate inside the simulation, but it makes detection structurally easier than with real attackers, who do not calibrate to the defender's P95. External validity is untested and should be stated as a limitation.

---

## 3. Label causality and latency

| Check (`3_label_checks.csv`) | Result |
|---|---|
| y_h(t)=1 without an attack day in [t−h+1, t] (or the reverse), all h | 0 rows |
| y_10 = 1 before an attacker's first injected day | 0 rows |
| Rows where a forward-looking 10-day label would differ from y_10 | 6,380 (the label is not forward-looking) |
| First-day latency anchor: rows with `wd_from_first_attack = 0` | 400, all injected days (y=1) |

### Why 96% of off-days are positive
- Of 1,670 forced off-days, **1,605 come after at least one attack day** in the same window. The gap to the last attack day is 1 workday (1,388), 2 (191), 3 (24) or 4 (2); the maximum is **4 workdays**.
- Because every such gap is below 5, they are positive under y_5, y_10 and y_20 alike (1,605 each). The positive label comes from the past attack day.
- The remaining **65 off-days (3.9%) precede the first injected day** (59 windows open with 1–2 off-days). They are negative under every horizon, which confirms that no future attack day labels an earlier row.

### Latency (workdays from the first injected day to the first in-window alarm; OOF, 3 repeats pooled)
| h | Model | User FPR | Detected (of 348) | P25 | Median | P75 | P90 | Alarm on day 0 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 10 | LightGBM | 1% | 303 | 1 | 3 | 6 | 9.2 | 15.5% |
| 10 | LightGBM | 5% | 320 | 0 | 2 | 4 | 7 | 28.1% |
| 10 | CatBoost | 1% | 308 | 1 | 3 | 5 | 8 | 22.6% |
| 10 | CatBoost | 5% | 327 | 0 | 1 | 4 | 7 | 35.3% |

All horizons: `3_latency_distribution.csv`. Alarms on a leading off-day (negative latency) are 0–4 per configuration.

**By attack duration** (LightGBM h=10, 1% user FPR, `3_latency_by_duration_lgbm_h10.csv`):

| Duration | 5 | 10 | 20 | 30 | 40 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| Detected | **48.6%** | 83.9% | 90.6% | 93.9% | 99.5% | 96.7% |

**Concern:** 5-day attacks are the clear weakness, with only about half detected at 1% user FPR.

---

## 4. Recall@1%FPR: definition and resolution

**Row level (fold metric):**
- Threshold = the score exceeded by at most 1% of **benign-user rows** in that validation fold (`threshold_at_fpr`: the (k+1)-th highest benign score, k = ⌊0.01·n⌋).
- Recall is the share of y_h = 1 rows above it.
- Each fold has 10,486–10,850 benign rows, so about 105 false-positive rows per fold. Resolution is fine.

**User level:**
- Per user, the peak = max OOF score over the whole period.
- Threshold = the score exceeded by at most 1% of **benign users' peaks**, pooled over the 5 folds of one repeat (511 benign users, k = 5).
- An attacker counts as detected if any row inside its attack window is above the threshold.
- Computed per repeat, then averaged.

**Actual FP counts (h=10, `1_user_points_fp_counts.csv`):**

| Target | FP users | Benign users | Achieved FPR | 95% exact CI of FPR |
|---|---:|---:|---:|---|
| 1% | 5 | 511 | 0.98% | 0.32%–2.27% |
| 5% | 25 | 511 | 4.89% | 3.19%–7.14% |

**Pooled vs per-fold threshold** (`4_per_fold_vs_pooled_threshold_h10.csv`):
- Each fold has only 101–103 benign users, so a per-fold 1% threshold allows exactly **1** FP per fold. That is the same 5 FP per repeat in total.
- With per-fold thresholds, recall is slightly higher: LightGBM 0.868–0.888, CatBoost 0.897–0.902.
- Pooling mixes scores from 5 fold models with slightly different scales. This is acceptable for comparison, but it is a source of noise.

**Test resolution (arithmetic only; no test scores computed):**
- 84 benign users: ⌊0.01·84⌋ = 0, so a test-derived "1% FPR" threshold equals the **maximum benign peak**. It means **0 FP**, not 1%.
- The smallest non-zero FPR step is 1/84 = 1.19%.
- Even 0/84 FP has a 95% upper bound of 4.3%; 1/84 → 0.03%–6.5%; 4/84 → 1.3%–11.8%.
- **Recommendation:** on test, do not derive thresholds from test scores. Apply thresholds **frozen from dev**: the fixed 85/60 risk bands, plus the dev 1%-user-FPR threshold expressed on the risk scale. Report raw FP counts (k/84) with exact CIs, and recall at 0, 1, 2 and 4 FP users as a curve.

---

## 5. Stage B readiness (proposals; nothing executed)

### 5.1 Optuna objective
**Handover objective (§3.4):** mean row PR-AUC on y_10 over 5 schedule-grouped inner folds of the dev users (seed offset +100), 60 trials, default configuration as trial 0. Adoption is already pre-registered: CV Recall@1%FPR ≥ default + 0.005 and PR-AUC ≥ default.

**Alignment concern:** row PR-AUC weights attackers by their number of positive rows. 50-day campaigns supply **29.3%** of y_10-positive rows and 5-day campaigns **5.3%** (`5_positive_rows_by_duration.csv`), yet 5-day attacks are the weakest operationally. MUBSIR's goal is per-attacker, early detection at a low benign-user alarm rate.

| Option | For | Against |
|---|---|---|
| (a) Row PR-AUC, unweighted (handover) | consistent with handover and with the pre-registered adoption metrics; smooth and stable | under-weights short attacks |
| (b) Row PR-AUC with evaluation weights so each attacker's positive rows sum to 1 | aligns with "each attacker counts once"; still smooth | new metric; must be fixed before running |
| (c) User Recall@1% user-FPR | the operational quantity | resolution of 1 FP per inner fold, too noisy to optimise |

My recommendation: (a) as the objective, plus a duration breakdown in the tuning report. Option (b) is reasonable if decided **before** running. Not (c).

### 5.2 Platt calibration with 3 repeated OOF sets
**Proposal:** fit one Platt model (`a·logit(p)+b`) on the **stacked** OOF scores of all 3 repeats. Each dev row appears 3 times, each time scored by a different fold model.
- Stacking matches the distribution of single-model scores, which is what the deployed model produces.
- It averages over fold-assignment randomness.
- Averaging scores across repeats before fitting would calibrate a smoother, ensemble-like score that is never deployed.
- `a, b` will also be reported per repeat to show stability.

**Known limitation (as in the handover):** the final model is refit on all dev users with the mean CV tree count. Its scores can be slightly sharper than fold-model scores, and the OOF-fitted calibration does not see this.

### 5.3 What the 0–100 risk score means
- Calibration is at **daily-row level** against **y_10**.
- `risk(t) = 100 · P(at least one injected attack day among the user's last 10 workdays | features at day t)`, as estimated under the **dev population mix**: 13.5% positive rows and 40.5% attackers among users.
- It is **not** a production probability. Real attacker prevalence is far lower, so the calibrated value overstates real-world risk; the 60/85 bands are fixed project decisions and do not change this.
- The per-user `risk_peak` is the max of daily risks over many days. It is **not** a calibrated per-user probability, because a benign user has about 104 chances to peak.
- With the single-model design there is no longer a max across horizons, only the max across days.

---

## 6. Concerns summary
1. **Rule A averages over horizons;** at the deployed h=10, CatBoost would pass. Documented, not changed. CatBoost's User AUC gain is significant; its User Recall gain is not.
2. **Simulator–feature coupling:** injection is sized in the same P95 units the deviation features use. Strong in-simulation signal; external validity untested.
3. **Residual population signal:** counterfactual AUC 0.60, since attackers' normal behaviour differs from benign users'. This inflates user-level metrics slightly beyond pure attack detection.
4. **5-day attacks:** 48.6% detected at 1% user FPR (dev OOF).
5. **Test resolution:** 84 benign users make "1% FPR" on test equal to 0 FP. Use dev-frozen thresholds and report FP counts with exact CIs.
6. **Bootstrap scope:** CIs are conditional on the trained models (user sampling only).
7. **Row-level Optuna objective under-weights short attacks:** decide (a) or (b) before Stage B.
