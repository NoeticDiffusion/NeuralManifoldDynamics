# 030 - 2026-05-17 - Manifest fix and MNPS-only export

## Session goal

1. Remove stale historical error state in PhysioNet top-up manifest after manual retry success.
2. Run base (non-spindle) MNDM features + summarize to produce H5 outputs.

## What was done

- Re-ran targeted top-up downloader with:
  - `physionet_ingest/config/config_i-care_2_1_longitudinal_targeted_topup_cpc1_71.yml`
  - existing-file verification mode: `verify_checksum=false`, `verify_existing_size=true`
- Result:
  - `890/890` files processed
  - `errors=0`
  - stale manifest error row removed in regenerated `download_manifest.csv`.

- Ran base PhysioNet features:
  - Config: `mndm/config/config_ingest_physionet_i-care_2_1.yaml`
  - Command resolved with monorepo `PYTHONPATH`.
  - Existing cache detected; only 2 new/changed files required feature extraction.

- Initial base summarize run with same config failed late with known issue:
  - `Stratified MNPS normalization failed: Duplicate coords_9d columns detected`.

- Added a stable summarize overlay config:
  - `mndm/config/config_ingest_physionet_i-care_2_1_mnps_only.yaml`
  - Overrides:
    - `mnps_9d.enabled: false`
    - `mnps_3d.mode: direct_features`

- Re-ran summarize using the `mnps_only` config.

## Outcome

- Summarize completed successfully (`exit_code=0`).
- Output run:
  - `E:/Science_Datasets/physionet/processed/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260517_164032`
- Output counts:
  - `729` H5 files
  - `729` `summary.json` files
  - `run_manifest.json` present.

## Notes

- This run is explicitly non-spindle and focused on stable MNPS export completion.
- The `mnps_only` config can be reused for future full-cohort exports until the 9D duplicate-column path is fully hardened.
