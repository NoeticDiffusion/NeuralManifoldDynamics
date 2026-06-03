## Session: Compare ComBat part1 vs older part1

Date: 2026-05-27

### Question
Compare the latest I-CARE part1 0-12h regional run with ComBat against an older part1 run from the same processed directory.

### Compared runs
- Baseline (older part1 regional):
  - `neuralmanifolddynamics_physionet_icare_2_1_20260525_141608`
- New (ComBat enabled):
  - `neuralmanifolddynamics_physionet_icare_2_1_20260526_164042`

### Run-manifest comparison
- Both:
  - `regional_outputs: true`
  - `mnps9d: true`
  - `run_status: completed_with_errors`
- Baseline:
  - `h5: 3396`
  - `run_errors: 88 / 3501`
  - no normalization block populated in manifest extra
- ComBat:
  - `h5: 3103`
  - `run_errors: 381 / 3501`
  - `extra.normalization.status: applied`
  - `rows_harmonized: 2846892/2846892`
  - `feature_columns_harmonized: 97/97`
  - `covariates_used: [age, sex]` (`group` had zero coverage)
  - `batch_counts: A=1308525, B=464567, D=431032, E=338237, F=304531`

### Error-profile comparison (`run_errors.json`)
- Baseline:
  - `RuntimeError: 83`
  - `ValueError: 5`
  - dominant message: `Stratified MNPS normalization failed: Duplicate coords_9d columns detected: {'m_e': 'm_a'}`
- ComBat:
  - `ValueError: 331`
  - `RuntimeError: 50`
  - dominant messages include:
    - `Duplicate coords_9d columns detected: {'m_e': 'm_a'}` (204)
    - `Duplicate coords_9d columns detected: {'m_e': 'm_a', 'd_n': 'm_a'}` (102)

### Output overlap sanity-check
- Run directories created per grouping are present in both runs (`3501` folder keys each).
- H5 success sets differ:
  - old-only successful H5 IDs: `330`
  - new-only successful H5 IDs: `37`
  - common successful H5 IDs: `3066`

### Spot-check artifact difference
- Example recording:
  - `sub-0284_post_cardiac_arrest_coma_continuous_eeg_run-001_acq-004`
- H5 hash changed:
  - old SHA256: `6eeda06e6a2adac92cf54e98108101a6fa4de45e4b374b02a27e6d8563b469fd`
  - new SHA256: `65c0b9611d934a18809fed0c79cf7f2d91cdfe635f866b04e23171197586a97d`
- H5 size changed:
  - old: `2,210,153 bytes`
  - new: `1,612,560 bytes`

### Takeaway
- ComBat was applied successfully at feature level, but this run produced substantially more stratified 9D normalization failures than the older part1 regional baseline.
- The bottleneck remains the known `coords_9d` duplicate-column failure mode, now appearing more frequently under the ComBat-transformed feature surface.

## Addendum: isolate regional vs ComBat on MNPS/MNJ

### Runs used
- `r0`: `20260523_173059` (older part1, no regional)
- `r1`: `20260525_141608` (regional enabled, no ComBat)
- `r2`: `20260526_164042` (regional enabled + ComBat)

### Global pairwise metric check (per-recording H5)
- Metrics computed from H5:
  - `mnps_rms` = RMS magnitude of `/mnps_3d`
  - `mnps_speed` = mean norm of `/mnps_3d_dot`
  - `mnj_fro` = mean Frobenius norm of `/jacobian/J_hat`
  - `mnjdot_fro` = mean Frobenius norm of `/jacobian/J_dot`

- `r0 -> r1` (`common=3396`):
  - all four metrics were numerically identical (`nonzero=0`, `max_abs_delta=0`) across all shared recordings.
  - Interpretation: enabling regional exports alone did not alter primary MNPS/MNJ tensors.

- `r1 -> r2` (`common=3066`):
  - `mnps_rms`: median pct `+1.786%`, p95 abs pct `33.122%`
  - `mnps_speed`: median pct `+1.206%`, p95 abs pct `30.281%`
  - `mnj_fro`: median pct `+2.505%`, p95 abs pct `54.943%`
  - `mnjdot_fro`: median pct `+2.560%`, p95 abs pct `56.575%`
  - Interpretation: ComBat materially shifts primary MNPS/MNJ values (especially tail behavior), while regional toggling does not.

### Selected-subject check (one per hospital, incl. prior focus subject)
- Subjects: `sub-0284` (A), `sub-0313` (B), `sub-0332` (D), `sub-0306` (E), `sub-0320` (F).
- Subject-level values are medians over each subject's common recordings across all three runs.

- `sub-0284` (n=15): `r2 vs r1`
  - `mnps_rms -9.39%`, `mnps_speed -6.31%`, `mnj_fro -5.69%`, `mnjdot_fro -8.02%`
- `sub-0313` (n=13): `r2 vs r1`
  - `mnps_rms +3.79%`, `mnps_speed +2.16%`, `mnj_fro +1.82%`, `mnjdot_fro +0.44%`
- `sub-0332` (n=101): `r2 vs r1`
  - `mnps_rms +6.48%`, `mnps_speed +4.02%`, `mnj_fro +24.68%`, `mnjdot_fro +23.90%`
- `sub-0306` (n=10): `r2 vs r1`
  - `mnps_rms -20.84%`, `mnps_speed -17.43%`, `mnj_fro +17.88%`, `mnjdot_fro +19.34%`
- `sub-0320` (n=1): `r2 vs r1`
  - `mnps_rms -0.05%`, `mnps_speed -0.23%`, `mnj_fro +2.00%`, `mnjdot_fro +0.53%`

### Note on outliers
- Some recordings show extreme MNJ inflation in `r2` (e.g. very large Jacobian Fro norms), consistent with already observed stratified/conditioning instability.
- For robust interpretation, median and p95 summaries are more reliable than max values.
