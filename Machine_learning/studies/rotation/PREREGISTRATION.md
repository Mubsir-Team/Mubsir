# Rotation-evasion experiment — pre-registration

Written 2026-11-26, **before any model of this experiment was trained**. Its SHA-256 is printed by `run_experiment.py` at start
and stored in `results/run_log.json`.

## Question
Does MUBSIR's cross-channel accumulation counter **evasion by channel rotation**? In other words, does switching channels stop
hurting detection once deviation is accumulated across channels, compared with a detector that monitors each channel separately?

The claim to test is **not** that "MUBSIR relies on rotation features". It is that rotation does not let an attacker evade MUBSIR,
while it does help evade a per-channel (siloed) detector.

## Data and protocol
- Dev users only (859: 348 attackers, 511 benign). Test users are dropped at load and never used.
- Same 15 outer folds (config/cv_folds.csv), seeds (42 + repeat), early-stopping users and LightGBM defaults as Stage A. Label y_10.
- **Arm M (MUBSIR):** the existing Stage A LightGBM y_10 out-of-fold scores (100 features). Not retrained.
- **Arm S (siloed per-channel detector):** three LightGBM models trained separately, each on ONE channel's own features only
  (email / USB+file / HTTP). Each also gets logon/logoff counts. No cross-channel feature is used.
  - Daily score = the maximum of the three channel probabilities, i.e. three independent monitors and an alarm if any fires.
- **Arm P (pooled per-channel):** one LightGBM on all per-channel features together (the trees may combine channels), but
  without the engineered cross-channel features (cumulative_deviation, combined_channel_intensity, channel_distribution_shift,
  consecutive_elevated_days, cross_channel_deviation_days, dominant_channel_switches, multi_channel_days_10wd,
  active_channel_count, all nr_*). It separates "having all channels" from "having explicit cross-channel accumulation".
- **User-level detection:**
  - The user score is the maximum daily score over the period.
  - An attacker counts as detected at a cross-fitted iso-FPR threshold: per repeat and validation fold, the threshold lets
    3% (primary) or 1% (secondary) of benign users of the OTHER folds alarm.
  - Every arm is therefore compared at the same benign false-positive rate. Results are averaged over the 3 repeats.
  - Attack-window detection (an in-window day above the threshold) is reported alongside.
- **Rotation measures (evaluation only):** scenario `switching_level` (Low / Medium / High), configured channel count (2 or 3),
  and the executed number of primary-channel switches from the ground truth.

## Decision rule (fixed now)
Let D(a, g) = detection rate of arm a for attackers in switching group g (iso-FPR 3%, user level, mean of 3 repeats).
Let Gap = [D(M, High) − D(S, High)] − [D(M, Low) − D(S, Low)].

The claim "cross-channel accumulation counters rotation evasion" is **supported** only if all three hold:
1. **Primary:** Gap > 0, with the 95% bootstrap CI (2,000 stratified user resamples, paired across arms) excluding 0.
2. **Siloed detector is hurt by rotation:** D(S, High) − D(S, Low) < 0, with the 95% CI excluding 0.
3. **MUBSIR is not hurt by rotation:** D(M, High) − D(M, Low) ≥ −0.03 (point estimate).

If 1 fails, the claim is **not supported**, whatever the other numbers show. If 1 holds but 2 or 3 fails, the result is reported
as partial. Robustness checks (reported, not part of the rule):
- the same at iso-FPR 1%;
- configured channel count 3 vs 2;
- a logistic regression of detection on arm × switches with stealth and duration as covariates;
- arm P in place of S.

No frozen artefact (model, manifest, calibration, thresholds, dataset, split, labels, OOF, test scores, reports) is modified.
