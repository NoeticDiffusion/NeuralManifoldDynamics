## Session: I-CARE part1 9D summarize rerun

Date: 2026-05-23

### Question
Can we re-run summarize for I-CARE part 1 with 9D enabled and regional MNPS disabled so pooled 9D analysis is possible?

### Config changes
- Updated `mndm/config/config_ingest_physionet_i-care_2_1.yaml`:
  - `regional_mnps.enabled: false`
  - `regional_mnps.stratified.enabled: false`
  - `regional_mnps.block_jacobians.enabled: false`
  - `mnps_9d.enabled: true`
  - `mnps_9d.block_jacobians.enabled: false` (kept lightweight)

### Run launched
- Command:
  - `python -m mndm.cli summarize --dataset physionet_icare_2_1 --config config/config_ingest_physionet_i-care_2_1.yaml --n-jobs 16`
- Output run directory:
  - `E:/Science_Datasets/physionet/processed/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260523_173059`

### Early validation (live sample H5)
- `coords_9d`: present
- `jacobian_9D`: present
- `regional_mnps`: absent

This indicates the rerun is producing 9D exports without regional outputs, as intended.

### Completion status
- Summarize run completed successfully (`exit_code: 0`).
- Runtime: ~3.07 hours (`elapsed_ms: 11,068,759`).
- Final output checks for run folder:
  - `h5_total`: 3396
  - `h5_with_coords_9d`: 3396
  - `h5_with_jacobian_9D`: 3396
  - `h5_with_regional_mnps`: 0
- Run-level diagnostics:
  - `run_manifest.json`: present (`extra.run_status = completed_with_errors`)
  - `run_errors.json`: present (`errors_total = 88`, `groupings_total = 3501`)
