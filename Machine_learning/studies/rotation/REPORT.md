# Rotation-evasion experiment — results

Pre-registration: `PREREGISTRATION.md` (SHA-256 `b7ca05e5…`, written and hashed before training; the hash is re-checked by
`run_experiment.py`). Dev users only (859: 348 attackers, 511 benign); the test set was not used.
All detection rates are user-level, at a cross-fitted iso-FPR threshold (3% of benign users unless stated), averaged over 3 OOF repeats.

## Verdict under the pre-registered rule: **PARTIAL**

| Condition | Result | Holds |
|---|---|---|
| 1. Primary: Gap = (M − S at High switching) − (M − S at Low switching) > 0, 95% CI excludes 0 | **+0.081 (0.004 to 0.156)** | yes |
| 2. Siloed detector is hurt by rotation: S(High) − S(Low) < 0, CI excludes 0 | −0.025 (−0.112 to +0.061) | **no** |
| 3. MUBSIR is not hurt by rotation: M(High) − M(Low) ≥ −0.03 | +0.056 (−0.019 to +0.133) | yes |

## Arms
| Arm | Features | User AUC | Recall @ 3% FPR | Recall @ 1% FPR |
|---|---|---:|---:|---:|
| **M — MUBSIR** (Stage A OOF, not retrained) | all 100 (50 cross-channel) | 0.987 | **0.912** | **0.876** |
| **S — three siloed per-channel monitors** (alarm if any fires) | email 17 / USB+file 16 / HTTP 15, each + 2 logon | 0.981 | 0.883 | 0.823 |
| **P — one model, per-channel features only** | 50 per-channel, no engineered cross-channel | 0.988 | 0.908 | 0.873 |
| S, email monitor alone | 19 | 0.954 | 0.800 | 0.739 |
| S, USB+file monitor alone | 18 | 0.728 | 0.360 | 0.266 |
| S, HTTP monitor alone | 17 | 0.860 | 0.596 | 0.513 |

## Detection by rotation dimension (recall @ 3% FPR)
| Group | n | M | P | S | M − S |
|---|---:|---:|---:|---:|---:|
| Switching Low | 117 | 0.866 | 0.886 | 0.872 | −0.006 |
| Switching Medium | 116 | 0.948 | 0.943 | 0.931 | +0.017 |
| Switching High | 115 | 0.922 | 0.896 | 0.846 | **+0.076** |
| 2 configured channels | 229 | 0.886 | 0.892 | 0.885 | +0.001 |
| 3 configured channels | 119 | **0.961** | 0.938 | 0.880 | **+0.081** |
| Executed switches 0–1 | 36 | 0.676 | 0.685 | 0.713 | −0.037 |
| Executed switches 7+ | 170 | 0.980 | 0.967 | 0.953 | +0.027 |

**Robustness (gap = MUBSIR advantage over the comparator, high minus low group; 95% CI):**
- iso-FPR 1%, switching High vs Low: +0.047 (−0.030 to +0.119).
- M vs P (instead of S): +0.046 (−0.012 to +0.112).
- 3 vs 2 configured channels: **+0.080 (0.028 to 0.136).**
- Executed switches 7+ vs 0–1: +0.065 (−0.066 to +0.196).
  - Both detectors improve with more switches, because more switches mean longer attacks with more injected days.
- Logistic regression on attacker × arm, with stealth and duration as covariates:
  - siloed × switches interaction: **−0.58 (−1.35 to −0.04)**;
  - siloed arm main effect: −0.78 (−1.36 to −0.39).

## What the evidence supports
1. **MUBSIR's multi-channel view beats per-channel monitoring, and the advantage grows when an attacker spreads across channels.**
   - Overall, at equal FPR: +2.9 recall points at 3% FPR and +5.3 at 1% (M vs S).
   - The advantage is about 0 for low-switching or 2-channel attackers, and about +8 points for high-switching or 3-channel attackers.
   - The pre-registered primary test passed (CI 0.004 to 0.156), and the 3- vs 2-channel check is clearly significant.
2. **Rotation does not let attackers evade MUBSIR.** Its detection does not fall with switching. High-switching attackers are
   detected at 0.922 vs 0.866 for low switching, partly because, in this simulation, high-switching users also have a different
   natural profile (audit part 1 §8.4).
3. **Not shown:** that switching by itself makes per-channel monitoring fail. The siloed detector's drop from low to high
   switching (−2.5 points) is not significant. Most of its weakness comes from the weak USB and HTTP monitors
   (AUC 0.73 and 0.86), not from switching.
4. **The advantage comes from seeing all channels in one model, not from the explicit rotation/cross-channel features.**
   - Arm P, one model given only per-channel features, matches MUBSIR overall (0.908 vs 0.912).
   - Its gap against MUBSIR at high switching (+0.046) has a CI that includes 0.
   - This agrees with the earlier ablation (removing `nr_*` changed nothing) and with SHAP.

## Defensible wording
> MUBSIR combines all exfiltration channels in one per-employee model and accumulates deviation across them.
> In the simulation, this design is not weakened by channel rotation. It outperforms per-channel monitors at equal false-positive
> rates, and its advantage grows when attacks are spread over more channels (pre-registered test: +8 points for high-switching
> attackers, 95% CI 0.4–15.6). The evidence does not show that the dedicated rotation features drive this advantage, and the
> per-channel detector's own loss from rotation was not significant.

## Limitations
- Simulated attacks only. Injection is sized per channel in P95 units, and the switching level follows each user's natural switching.
- Subgroups have about 115 attackers each, so CIs are wide.
- The siloed arm combines monitors by the maximum of uncalibrated probabilities. Every arm is compared at the same cross-fitted benign FPR.

## Integrity
Frozen artefacts were hashed before and after (`hashes_before.json`, `hashes_after.json`, `hashes_comparison.json`).
This experiment only wrote files inside `study_rotation/`.
