# Controlled channel rotation study — pre-registration (frozen protocol)

**Written:** 2026-11-26, after the Phase 0 audit (`AUDIT.md`) and the owner's conditional approval. **Before** any feature
re-implementation, data generation or model training. This file is hashed; the hash is stored in `PREREGISTRATION.sha256`
and checked by every script of the study. Any later change must be recorded as a dated amendment with a new hash.
**No rule below may be changed after results are seen. An inconclusive or null result is a valid outcome.**

---

## 1. Primary hypothesis
Adding explicit channel-transition features improves user-level detection of rotating exfiltration attacks, compared with a
model that uses channel and distributional-deviation features alone, at matched false-positive rates. The improvement is
**greater for rotating than for block-structured attacks**, which carry the same total and the same per-channel injected
deviation on the same days, in a different order.

**Primary estimand:**

    Δ = [Det(D, ROT) − Det(C+, ROT)] − [Det(D, BLK) − Det(C+, BLK)]

- `Det(arm, variant)` = share of counterfactual cases detected, at the cross-fitted iso-FPR 3% operating point (§6).
- Cases are pooled over the two attack lengths and averaged over the 3 OOF repeats.

## 2. Owner's conditions and how they are implemented
| # | Condition | Implementation |
|---|---|---|
| 1 | 253 employees primary, 39-employee original rule as sensitivity; document the changed assumption | Hosts are drawn from the 253 dev users with valid baselines on all 3 channels. Injection is allowed on any eligible workday, whether or not the employee naturally used that channel. **This changes an assumption of the original simulation** (injection only into naturally active channels). Exact per-channel matching (condition 2) is only possible for the **146** of the 253 with integer P50 and integer (P95 − P50) on every channel. The **primary contrast therefore uses these 146.** The full 253 (approximate matching) is secondary S7, and the original natural-activity rule is secondary S8. |
| 2 | Total AND per-channel deviation matched; ordering the only intended difference; report per-channel distribution | ROT and BLK use the same days, the same channels, the same number of days per channel (L/3) and the same injected deviation per channel-day (exactly 1 E unit). Matching is verified by recomputation (§4.4). Per-channel injected deviation and raw added counts are reported per variant. Any host whose recomputed per-channel deviation differs between ROT and BLK by more than 1e-9 is excluded and counted. |
| 3 | Frozen-model evaluation separate from controlled-data training; employee groups isolated | Arm A-frozen is scored once, as a descriptive reference only, and is part of no test. Regime 1 trains on the original dev simulation and scores each host only with fold models for which the host is in the validation fold. Regime 2 (secondary) trains on counterfactual data of training-fold hosts only and evaluates on validation-fold hosts. |
| 4 | Exact agreement with the team's 100-feature implementation before generating data; any mismatch stops the study | Gate 0 (§3). |
| 5 | Primary comparison, operating points, CIs, bootstrap unit and adoption criteria fixed in advance | §1 and §6–§8. |

## 3. Gate 0 — feature re-implementation (stop rule)
A standalone module, `features.py`, re-implements the full pipeline from daily counts plus the fixed 2010 baselines:
- NB12 §12.4 original features;
- §12.5.2 Clear-30;
- §12.6.1 rotation-27;
- `E_*_today` from `prepare_dataset.py`.

It must pass all of the following. **Any failure stops the study before counterfactual data is generated** (outcome: inconclusive).

1. From the injected daily counts (`user_day_injected_counts_r62.parquet`), it reproduces all 100 model features of
   `behavior_features_r62_ml_final_new.parquet` for the 995 eligible users: max |diff| ≤ 1e-9, identical NaN pattern,
   identical row keys.
2. From the pre-injection counts (`user_day_r62_v1.parquet`), it reproduces the same 100 features for every **benign** eligible
   user (whose rows are unaffected by injection).
3. Its injection helper reproduces NB10's `count_after` for every original ground-truth channel-day, using the 8 affected fields:
   - `external_email_count`, `email_send_count`, `email_event_count`;
   - `usb_copy_count`, `file_copy_count`, `file_event_count`;
   - `www_upload_count`, `http_event_count`.

## 4. Counterfactual data (non-destructive copy in `study_controlled_rotation/data/`)
### 4.1 Hosts
- **Primary cohort H146:** dev users with NB12-valid baselines on all three channels and integer P50_c and max(1, P95_c − P50_c)
  for c ∈ {Email, USB, HTTP}. This is 146 users (74 benign; 72 attackers, whose **pre-injection** history is used, so their
  original attacks are absent).
- Each host's unmodified history stays in the data only as the original dev user. It is never relabelled.
- **Minimum:** if fewer than 120 hosts survive §4.4 verification, the primary analysis is reported as inconclusive.

### 4.2 Attack window
- For each host, take its 2011 eligible workdays (weekday and logon > 0, as in training), in order, n of them.
- Draw a start index s uniformly from {30, …, n − 12 − 10} with `numpy.random.default_rng(42)`, iterating hosts in sorted order.
- Attack days are workdays s … s + L − 1, for L ∈ {6, 12}. Both lengths share the same start s.
- Hosts with n < 30 + 12 + 10 are ineligible and counted.

### 4.3 Variants (identical days; only the channel assignment differs)
- A channel permutation π of (Email, USB, HTTP) is drawn per host (same RNG, after s) and shared by all of that host's variants.
- Injected deviation per injected channel-day is **δ = 1 E unit**.

| Variant | Channel on attack day j (j = 0 … L−1) | Days per channel | Switches |
|---|---|---|---|
| ROT | π[j mod 3] | L/3 each | L − 1 |
| BLK | π[floor(3j / L)] | L/3 each | 2 |
| FIX-c (c = each channel) | c | L on c | 0 |
| SIM | all three channels, each with deviation 1/3 | L each, at δ/3 | 0 |

- ROT vs BLK is the **primary** contrast.
- FIX and SIM are secondary or descriptive. SIM cannot be matched exactly (δ/3·(P95 − P50) is usually not an integer); its achieved deviation is reported and it enters no test.

### 4.4 Injection rule (exact)
On channel c on attack day t, with natural count x = x_c(t), `den_c = max(1, P95_c − P50_c)` and target deviation δ:

    x' = max(x, P50_c) + δ · den_c        (an integer for every H146 host)

- Then E'_c(t) = E_c(t) + δ exactly when x ≥ P50_c, and E'_c(t) = δ when x < P50_c (where E_c(t) = 0). So the injected deviation is exactly δ.
- The added amount `x' − x` is added to the channel's count field and to its related event-total fields (the NB10 mapping in Gate 0.3).
- Nothing else is modified: no unique-file, byte or recipient field, consistent with how NB10 injected.
- **Verification:**
  - Per-channel injected deviation is recomputed with `features.py`.
  - ROT and BLK must have identical per-channel totals.
  - Every variant must have total injected deviation L·δ (SIM excepted).
  - Raw added counts per channel and variant are reported.

### 4.5 Features and labels
- All features are recomputed with `features.py` on the host's full modified 2011 history, using its fixed 2010 baselines.
- `y_10` comes from the injected days (trailing 10 workdays).
- Each case is a pseudo-user `host|variant|L`.

## 5. Arms (feature sets; LightGBM defaults, label y_10, same 15 outer folds, seeds 42 + repeat, early-stopping users as Stage A)
| Arm | Features | Count |
|---|---|---:|
| B | per-channel only: email 17, USB+file 16, HTTP 15, logon/logoff 2 (`study_rotation/results/feature_groups.csv`) | 50 |
| C | B + aggregated deviation: `cumulative_deviation_{5,10,20,30}wd`, `combined_channel_intensity_{5,10,20,30}wd`, `consecutive_elevated_days_{5,10,20,30}wd`, `nr_stealth_acc_{5,10,20}wd` | 65 |
| C+ | C + distributional deviation / diversity: `nr_m_*` (3), `nr_entropy_*` (3), `nr_excl_days_*` (3), `nr_overlap_days_*` (3), `cross_channel_deviation_days_*` (4), `channel_distribution_shift_*` (4), `active_channel_count`, `multi_channel_days_10wd` | 87 |
| A-cv | C+ + existing transition features: `nr_alt_*` (3), `nr_gap_alt_2_*` (3), `nr_handoff_*` (3), `dominant_channel_switches_*` (4). These are exactly the 100 MUBSIR features, refit per fold | 100 |
| **D** | A-cv + new sequence features (§5.1) + personal-norm switching (§5.2, subject to its gate) | 110 (or 108) |
| A-frozen | the frozen final model (descriptive reference only; in-sample for hosts' benign days; in no test) | 100 |

### 5.1 New sequence features (windows w ∈ {10, 20}; causal; computed from E with the NR rule)
Definitions:
- **Deviation day** t: Σ_c E_c(t) > 0.
- **Dominant deviation channel** dom(t) = argmax_c E_c(t), ties broken Email < USB < HTTP.
- Inside the window [t − w + 1, t], let d_1 < d_2 < … be its deviation days.
- **Linked pair:** (d_{i−1}, d_i) with at most 2 quiet workdays between them. A **change** is a linked pair with dom(d_i) ≠ dom(d_{i−1}).

Features:
1. `rot_changes_w` = number of changes. This is a count, unlike the ratio `nr_gap_alt_2`.
2. `rot_revisits_w` = number of deviation days d_i whose dom(d_i) occurred earlier in the window, but not at d_{i−1}.
   This captures returns such as A→B→A.
3. `rot_trans_entropy_w` = entropy of the counts over the 6 ordered change types (E→U, E→H, U→E, U→H, H→E, H→U),
   divided by ln 6. It is 0 if there are fewer than 2 changes.
4. `rot_cycle_w` = (number of i ≥ 2 such that dom(d_{i−2}), dom(d_{i−1}) and dom(d_i) are pairwise distinct and both pairs are
   linked) / max(1, rot_changes_w − 1).

### 5.2 Personal-norm switching (Experiment A) and its shortcut gate
- `base_rate(t)` = changes / linked pairs over workdays [t − 70, t − 11] (0 if there are no pairs).
- `cur_rate(t)` = the same over [t − 9, t].
- Features:
  - `sw_excess_rate_10 = cur_rate − base_rate`;
  - `sw_excess_count_10 = rot_changes_10 − base_rate × (linked pairs in [t − 9, t])`.
- **Gate, evaluated on the original dev data before D is trained:**
  - per user, take the mean of `base_rate` over rows before the first injected day (attackers) or over all rows (benign);
  - compute the user-level AUC for attacker vs benign, folded to ≥ 0.5;
  - if it is ≥ 0.60, both personal-norm features are dropped from D (D = 108 features) and the drop is reported.
  - No other use of the gate's result is allowed.

## 6. Evaluation
- **Regime 1 (primary):** train arms B, C, C+, A-cv and D on the original dev simulation. Each counterfactual case of host h is
  scored with the fold model (r, k) for which h is in validation fold k of repeat r.
- **Benign reference and thresholds:**
  - For each arm, repeat r and fold k, the threshold is `threshold_at_fpr(peaks, f)` (the score exceeded by at most a share f of the peaks).
  - The peaks are the whole-period peak OOF scores of the **511 original benign dev users in the other four folds**.
  - **f = 0.03 is primary, f = 0.01 secondary.** Counterfactual cases never set thresholds.
- **Detection of a case:** at least one day in [first attack day, last attack day + 9 workdays] (the y_10-positive span) has a
  score above the threshold.
- **Latency:** workdays from the first attack day to the first such day.
- **Aggregation:** Det = mean over cases (both L, each host contributing one case per L per variant), averaged over the 3 repeats.
- **Bootstrap:** 2,000 resamples of **hosts** (stratified benign/attacker host), with all of a host's variants and lengths resampled
  together and the estimand averaged over repeats inside each resample. 95% percentile CI.
  - Hosts are distinct users with independently drawn windows; no attack template is shared.
- **Regime 2 (secondary S4):** repeat 0, five folds. Arms C+ and D are trained on:
  - the original dev rows of the fold's fit and early-stopping users, plus
  - all counterfactual variant rows (both L) of H146 hosts **not** in validation fold k.

  They are evaluated on validation-fold hosts. Early stopping uses the original early-stopping users only.

## 7. Decision rules
**Outcome 1 — independent switching signal demonstrated.** All of:
1. Δ ≥ 0.02 (2 percentage points);
2. the 95% CI of Δ lies entirely above 0;
3. Det(D, ROT) − Det(C+, ROT) > 0, with its 95% CI above 0;
4. no ceiling or floor (below).

**Outcome 2 — distributed deviation matters, explicit switching does not.** The primary test fails, and secondary S5 is
positive after Holm correction. S5 is `[Det(C+, BLK) − Det(C, BLK)] − [Det(C+, FIX) − Det(C, FIX)]`, where FIX pools the three
single-channel variants.

**Outcome 3 — neither adds measurable value.** The primary test fails, and the 95% CIs of Δ and S5 both lie within (−0.05, +0.05).

**Outcome 4 — inconclusive.** Any of:
- Gate 0 fails;
- fewer than 120 H146 hosts survive verification;
- ceiling or floor: Det(C+, ROT) and Det(C+, BLK) both ≥ 0.97 or both ≤ 0.03;
- none of outcomes 1–3 applies, for example because CIs are too wide to exclude ±5-point effects.

**Adoption:** only Outcome 1 can justify proposing a new experimental MUBSIR version. Even then, it needs separate validation
on new data before any change to the frozen model. No result of this study modifies the current model.

## 8. Secondary analyses (Holm–Bonferroni across S1–S8, α = 0.05)
Two-sided bootstrap p-values are computed as 2 × min(share ≤ 0, share ≥ 0).

| ID | Estimand |
|---|---|
| S1 | Δ at iso-FPR 1% |
| S2 | Det(D, ROT) − Det(C+, ROT) at iso-FPR 3% |
| S3 | the Δ interaction with A-cv in place of D (existing transition features only) |
| S4 | Δ under Regime 2 (training on counterfactual data) |
| S5 | the distributed-deviation interaction (Outcome 2) |
| S6 | Δ separately for L = 6 and L = 12 (two tests within S6, both Holm-corrected) |
| S7 | Δ on the full relaxed cohort of 253 hosts. Non-H146 hosts use x' = max(x, ⌈P50⌉) + round(δ·den); the achieved mismatch is reported |
| S8 | Δ on H146 hosts with at least L days where all three channels are naturally active, with attack days restricted to such days (the original-simulation rule). Expected to be underpowered |

**Descriptive only (no tests):** SIM and per-channel FIX detection, latency distributions, the A-frozen reference, and
unique detections. Unique detections are ROT cases detected by D but not by C+ in at least 2 of 3 repeats, and the reverse.

## 9. Integrity
- No frozen artefact is modified: model, manifest, calibration, thresholds, allowlist, dataset, split, labels, OOF, reports.
- The test set and test scores are never read.
- SHA-256 of all pre-existing project files is recorded before (`hashes_before.json`) and after the study.
- All artefacts are written inside `study_controlled_rotation/`.
- `EXPERIMENT_LOG.md` records every command, seed and gate result in order.
