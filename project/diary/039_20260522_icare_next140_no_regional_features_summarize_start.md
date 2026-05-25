# 039 — 2026-05-22 — I-CARE next140 no-regional features+summarize start

## Context
- User requested a full `features + summarize` pass for the newly completed I-CARE `next_140` 0-12h cohort.
- User explicitly requested no regional calculations and no block Jacobians for this run.

## Config updates
- Added run-specific overlay:
  - `mndm/config/config_ingest_physionet_i-care_2_1_next_140_0_12h_no_regional.yaml`
- Overlay details:
  - imports `config_ingest_physionet_i-care_2_1_mnps_only.yaml`
  - sets dataset root to:
    - `E:/Science_Datasets/physionet/received/i-care_2_1_next_140_longitudinal_0_12h/training`
  - disables:
    - `regional_mnps.enabled`
    - `regional_mnps.stratified.enabled`
    - `regional_mnps.block_jacobians.enabled`
    - `mnps_9d.block_jacobians.enabled`
    - `robustness.ensembles.enabled`

## Run launch
- Started sequential run with shared config and output dir:
  - `features` then `summarize` (second stage gated on first-stage success)
- Output root:
  - `E:/Science_Datasets/physionet/processed_next140_0_12h_no_regional`
- Workers:
  - `--n-jobs 16`

## Initial status
- Startup validated from terminal logs:
  - `Computing features for physionet_icare_2_1`
  - `No file index found ... building index from ... next_140_longitudinal_0_12h/training`
- New output dataset directory exists and began receiving files.

## Mid-run correction
- First launch command used `&&` and failed in this PowerShell environment.
  - Follow-up: relaunched with PowerShell-compatible gating (`if ($LASTEXITCODE -eq 0) { summarize }`).
- During early features execution, WFDB load errors appeared for:
  - `training/0447/0447_004_022_EEG.hea`
  - `training/0442/0442_013_052_EEG.hea`
- Root cause identified:
  - corresponding `.mat` files were truncated locally from earlier name-only skip mode.
  - local vs remote size mismatch confirmed:
    - `0447_004_022_EEG.mat`: `2,653,832` vs `36,864,024`
    - `0442_013_052_EEG.mat`: `4,439,688` vs `68,400,024`

## Repair actions
- Stopped the in-flight no-regional run processes.
- Redownloaded only the two affected `.mat` files in place.
- Re-verified repaired files now match remote content-length exactly.
- Restarted a clean sequential `features -> summarize` run to a fresh output root:
  - `E:/Science_Datasets/physionet/processed_next140_0_12h_no_regional_rerun_20260522`

## Extended repair pass and clean restart
- Additional WFDB read failures appeared in the restarted run for more records, indicating broader file truncation from earlier name-only checks.
- Stopped the run again and launched a full size-verified PhysioNet top-up:
  - `physionet_ingest.script.download_physionet --config-dataset physionet_ingest/config/config_i-care_2_1_next_140_longitudinal_0_12h.yml`
  - mode confirmed from logs: `Existing-file verification mode: size (checksum=False, size_check=True)`
  - completion: `3943/3943`, `errors=0`, `exit_code=0`
  - post-check: no missing planned files and no `.part` files
- Started a new clean sequential `features -> summarize` run to a new output root:
  - `E:/Science_Datasets/physionet/processed_next140_0_12h_no_regional_rerun_20260522_clean`
- Startup for the clean run validated:
  - `Computing features for physionet_icare_2_1`
  - `No file index found ... building index from ... next_140_longitudinal_0_12h/training`

## Clean run result (latest)
- Clean sequential run finished with `exit_code=0`.
- Output run dir:
  - `E:/Science_Datasets/physionet/processed_next140_0_12h_no_regional_rerun_20260522_clean/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260522_095945`
- Manifest summary:
  - `counts.h5 = 1837`
  - `counts.regional_csv = 0`
  - `counts.block_csv = 0`
- During run, one WFDB record failed (`0779_012_014_EEG.hea`) due truncated `.mat`.
- Follow-up applied immediately:
  - redownloaded `training/0779/0779_012_014_EEG.mat`
  - verified local size matches remote (`68400024`)
  - verified `wfdb.rdrecord(...)` now succeeds for that record.

## Config update + summarize rerun (9D enabled)
- User requested enabling `mnps_9d` while keeping block Jacobians disabled.
- Updated:
  - `mndm/config/config_ingest_physionet_i-care_2_1_next_140_0_12h_no_regional.yaml`
  - `mnps_9d.enabled: true`
  - `mnps_9d.block_jacobians.enabled: false`
- Launched summarize-only rerun with the updated config:
  - `mndm.cli summarize --dataset physionet_icare_2_1 --config ...next_140_0_12h_no_regional.yaml --out-dir E:/Science_Datasets/physionet/processed_next140_0_12h_no_regional_rerun_20260522_clean --n-jobs 16`
- Startup confirmed from logs:
  - `INFO: Summarizing physionet_icare_2_1`

## Manual part1 folder reconciliation
- User confirmed they had manually moved all recording subfolders from:
  - `..._20260519_153608_part2` -> `..._20260519_153608`
- Follow-up action performed for remaining top-level data files:
  - moved `features_snapshot.json` into the main run folder
  - merged row content from `_part2` into main-folder CSVs:
    - `regional_mnps_subjects_153608.csv`
    - `regional_block_jacobians_subjects_153608.csv`
    - `stratified_block_jacobians_subjects_153608.csv`
  - deduplicated rows during merge
  - created pre-merge backups for destination CSVs:
    - `*.bak_before_part2_merge`
- Post-check:
  - `_part2` contains no remaining files
  - merged CSVs and `features_snapshot.json` present in main run folder.
