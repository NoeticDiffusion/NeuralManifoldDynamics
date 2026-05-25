# 038 — 2026-05-22 — PhysioNet name-only top-up mode

## Context
- User requested one final i-care top-up pass for the `next_140` (0-12h) cohort.
- Before top-up, user requested faster existing-file checks: filename/path only (skip size/checksum verification for existing files).

## Code updates
- Updated `physionet_ingest/script/download_physionet.py`:
  - Added CLI flag:
    - `--name-only-existing-check`
  - Added runtime override in `run_download(...)`:
    - `name_only_existing_check_override`
  - When override is enabled:
    - forces `verify_checksum = false`
    - forces `verify_existing_size = false`
    - existing files are skipped via name/path presence only (`skipped_exists`)
  - Added run summary field:
    - `execution.existing_file_check_mode` (`checksum` / `size` / `name_only`)
- Updated tests in `physionet_ingest/tests/test_download_physionet.py`:
  - added explicit test that name-only mode does not call remote size checks
  - added integration-style run test for `name_only_existing_check_override=True`

## Validation
- Ran:
  - `python -m pytest physionet_ingest/tests/test_download_physionet.py`
- Result:
  - `22 passed`
- Verified CLI help contains:
  - `--name-only-existing-check`

## Execution
- Started final top-up pass with name-only mode enabled:
  - `python -m physionet_ingest.script.download_physionet --config-dataset physionet_ingest/config/config_i-care_2_1_next_140_longitudinal_0_12h.yml --name-only-existing-check`
- Early progress during run:
  - existing-file mode logged as `name_only`
  - progress reached `2000/3943`, `errors=0`

## Evidence category notes
- **Internal validated result**:
  - new CLI mode is implemented, tested, and actively used in current top-up run.
- **Plausible interpretation**:
  - top-up wall-clock should improve vs size-verified mode due removed remote HEAD checks on existing files.
