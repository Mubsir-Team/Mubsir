# Experiment log — controlled channel rotation study

| Time (local) | Step | Command / action | Result |
|---|---|---|---|
| 2026-11-26 04:47 | Phase 0 audit | read-only inspection (AUDIT.md) | existing data cannot isolate switching; matched counterfactual feasible with relaxed day rule |
| 2026-11-26 04:47 | Eligibility check | read-only count of exact-matchable hosts | 146 of 253 (74 benign) |
| 2026-11-26 04:47 | Pre-registration frozen | sha256sum PREREGISTRATION.md | 954fe7d4bfadbea19e4f5b6ad7df0a18b6c95cd17c24d15c28ca3d67644e24fe |
| 2026-11-26 04:47 | Integrity baseline | python hash_artifacts.py before | all pre-existing files hashed (hashes_before.json) |
| 2026-11-26 04:53 | Gate 0 | python gate0.py | **PASS**: 0.1 all 100 features of 995 users from injected counts, max abs diff 0.0, NaN pattern identical; 0.2 595 benign users from pre-injection counts, max abs diff 0.0; 0.3 NB10 mapping exact on all 18 raw fields (results/gate0.json) |
| 2026-11-26 04:53 | Gate 0 negative control | perturb one email count (+3) for one user, recompute | 41 features change, only from the perturbed row onward, over 30 rows: comparison is sensitive and causal |
| 2026-11-26 05:09 | Data generation | python gen_data.py (seeds: H146 rng 42, R253 rng 43, S8 rng 44) | **PASS** (results/data_generation.json): 2,252 cases, 236,796 rows. H146: 145 of 146 hosts (1 excluded: 15 workdays). ROT vs BLK per-channel injected deviation identical (max diff 2.2e-16); total = L exactly for ROT/BLK/FIX; no deviation change outside attack days. R253: 104 of 107. S8: 36 hosts for L=6, 12 for L=12 |
| 2026-11-26 05:09 | Implementation note (before any model) | S8 eligibility per length (>= L all-3-active days), as written in PREREGISTRATION §8; first run had required >= 12 for both lengths and was regenerated. Non-H146 RNG (43) and S8 RNG (44) and SIM rounding max(1, round(den/3)) were not specified in the protocol and were fixed here before any training | SIM deviation not matched (mean 10.6 for L=6 vs 6): descriptive only, as registered |
| 2026-11-26 05:19 | New features unit check | toy sequences E-U-H-E-U-H / E-E-U-U-H-H / E x6 | ROT: changes 5, revisits 3, cycle 1.0; BLK: changes 2, revisits 0, cycle 0; FIX: all 0, as defined in §5.1 |
| 2026-11-26 05:19 | Shortcut gate (§5.2) | python train_arms.py | folded user-level AUC of base_rate = 0.572 (< 0.60) -> personal-norm features KEPT; D = 110 features. 834 dev users (25 attackers have no row before their first injected day) |
| 2026-11-26 05:19 | Training regime 1 + 2 | python train_arms.py (seeds 42+repeat; LightGBM defaults) | 15 folds x 5 arms + 5 folds x 2 arms; A-cv reproduces the Stage A OOF exactly; every case scored once per repeat by a model that never saw its host; 277 s; scores/ |
| 2026-11-26 05:19 | Integrity check | python hash_artifacts.py after | no pre-existing file changed |
| 2026-11-26 05:25 | Implementation note (before the analysis run) | analyze.py written; choices not fixed by the protocol, fixed before any estimand was seen: raw-probability scale for thresholds (monotone in risk); bootstrap rng default_rng(2026), one shared set of 2,000 stratified host draws per host set; 'S5 positive after Holm' = estimate > 0 and Holm p < 0.05; ceiling/floor checked before Outcomes 1-3; if Outcomes 2 and 3 both apply, 2 is reported and the overlap flagged; A-frozen at its frozen operating points (risk > 85, 92.632), descriptive only | - |
| 2026-11-26 05:34 | Registered analysis | python analyze.py | Primary Delta = -0.0080, 95% CI [-0.0276, +0.0115]; Outcome-1 conditions not met (no ceiling/floor: Det(C+,ROT) 0.836, Det(C+,BLK) 0.834). S1-S4, S6-S8 all n.s. after Holm; S5 = +0.0387 [+0.0226, +0.0571], Holm p < 0.005. **Outcome 2** by the frozen rules (Outcome 3 not met: S5 CI upper 0.057 > 0.05). Held-out benign FPR 3.0-3.5% (3%) and 1.0-1.3% (1%). results/analysis_* |
| 2026-11-26 05:34 | Post-hoc decomposition (descriptive, written after the analysis, no tests) | python posthoc_descriptive.py | S5 arises from C+ - C = -4.3 pts on FIX [-5.5, -3.2] vs -0.5 on BLK [-1.8, +1.0]: distributional features reduce single-channel detection, they do not improve multi-channel detection. results/posthoc_arm_differences.csv |
| 2026-11-26 05:34 | Figures, inputs manifest, README, run_all.ps1, REPORT.md | python figures.py; python inputs_manifest.py | 3 figures; 23 external inputs hashed, none changed; REPORT.md written |
| 2026-11-26 05:34 | Final integrity check | python hash_artifacts.py after | 350 pre-existing files unchanged, 0 removed, 0 added outside the study; no file outside study_controlled_rotation/ created since the pre-registration; PREREGISTRATION.md hash unchanged |
| 2026-11-26 05:48 | Documentation | REPORT.md Arabic summary replaced by an English executive summary; table of the new switching features added to REPORT §1 (no result changed) | all study files are English |

## 2026-11-26 — Folder move (no re-run, no result changed)
- The folder moved from `Model/study_controlled_rotation/` to `Model/studies/controlled_rotation/` for the MUBSIR Model v2 repository layout.
- Only path lines changed: `MODEL_DIR = HERE.parent.parent`, paths to sibling studies, `run_all.ps1` → `..\..\.venv`.
- No score, result, figure, report, pre-registration or hash file changed. Old folder names in the text above are kept as records.
- Details and the before/after hash check are in `Model/RESTRUCTURE.md`.
