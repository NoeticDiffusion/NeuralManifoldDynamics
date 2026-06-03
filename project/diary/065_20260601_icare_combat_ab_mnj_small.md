# 065 — 2026-06-01 — i-care ComBat A/B small-sample MNJ check

## Research question
Does ComBat reduce the extreme-tail behavior in MNJ-adjacent reachability metrics (especially `mnj`-relevant persistence terms) on a minimal i-care sample, and can this reduce winsorization pressure in downstream violin-style summaries?

## Scope
- Dataset family: PhysioNet i-care random100 longitudinal.
- A/B roots:
  - **A (pre-ComBat)**: `neuralmanifolddynamics_physionet_icare_2_1_20260523_173059`
  - **B (post-ComBat cleaned)**: `neuralmanifolddynamics_physionet_icare_2_1_20260527_101007_NeruroComBat_cleaned`
- Sample size: 6 matched run-level files (one run per subject):
  - `sub-0284`, `sub-0286`, `sub-0296`, `sub-0299`, `sub-0303`, `sub-0306`

## What was executed
1. Created matched pre/post subset folders with 6 `.h5` files each under:
   - `E:/Science_Datasets/physionet/processed/physionet_icare_2_1/ab_test_mnj_small/pre_h5`
   - `E:/Science_Datasets/physionet/processed/physionet_icare_2_1/ab_test_mnj_small/post_h5`
2. Added a temporary ndt-analysis test config with subject-anchor contract:
   - `project/summary/icare_ab_test_reachability_subject_anchor.yaml`
3. Ran `ndt-analysis` `reachability_cones` on both subsets.
4. Computed paired A/B comparisons for:
   - `tube_a_operator_norm_median`
   - `persistence_kappa_median`
   - `persistence_log_eig_var_median`
   - `tube_capture_gate_median`
   - `tube_d_eff_median`

## Key results (internal validated, small-N)
### 3D reachability metrics (forced `space=mnps_3d` for valid pairing)
- `persistence_kappa_median` tail compressed strongly:
  - max/median: **8.05 → 2.08**
  - std: **0.753 → 0.158**
  - Tukey outliers (1.5*IQR): **1/6 → 0/6**
- `persistence_log_eig_var_median` tail compressed strongly:
  - max/median: **6.95 → 2.12**
  - std: **7.51 → 1.80**
  - Tukey outliers: **1/6 → 0/6**
- `tube_a_operator_norm_median` mostly stable (no clear directional shift):
  - median relative change: **-0.42%**
  - abs decreases/increases: **3 down / 3 up**
- `tube_capture_gate_median`: slight upward drift on this tiny sample.
- `tube_d_eff_median`: near-stable central tendency.

Interpretation for reviewer-facing tails: on this small subset, ComBat mainly damped extreme spread in persistence-family metrics rather than shifting central tendency.

## Important diagnostic finding
Default `reachability_cones` (`space=v2` / 9D-first) failed post-ComBat for all 6 selected files:
- `pre9_finite_ok_count = 6/6`
- `post9_finite_ok_count = 0/6`

Root pattern in post-ComBat `coords_9d/values`:
- coordinate `e_m` is non-finite for all timepoints in all 6 tested files.

This makes 9D reachability summaries all-NaN unless forcing a 3D state-space.  
Status category: **internal validated diagnostic** (not yet root-caused in normalization code).

## Caveats
- Small sample (6 runs) by design, intended as a quick tail-sanity test.
- Results are run-level, not a full cohort inference.
- One subject (`sub-0299`) drives a large share of the observed tail collapse.

## Recommended next step
Run the same A/B calculation on 20-40 matched runs and include a dedicated QC check for `coords_9d` non-finite columns (`e_m` in particular) before plotting or winsorization.
