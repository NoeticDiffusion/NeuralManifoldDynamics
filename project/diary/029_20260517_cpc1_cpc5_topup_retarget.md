# 029 - 2026-05-17 - Retargeted top-up to CPC1/CPC5

## Session goal

Stop the in-flight broad top-up run and retarget selection to a clearer cohort objective: `CPC1=71` and `CPC5=71`, while reducing local verification overhead by avoiding SHA256 hashing on existing files.

## What was done

- Stopped active PhysioNet top-up download processes for the previous 105-ID plan.
- Re-counted currently downloaded cohort metadata under:
  - `E:/Science_Datasets/physionet/received/i-care_2_1_random100_longitudinal/training`
  - Current counts after stop: CPC1=46, CPC5=63 (129 patients with metadata).

- Updated planner behavior in `physionet_ingest/script/plan_icare_targeted_topup.py`:
  - Removed non-target tie-break drift toward CPC2-4.
  - Added early stop once all requested deficits are satisfied.
  - Effect: planner now stops at minimum required set for requested targets.

- Updated downloader behavior in `physionet_ingest/script/download_physionet.py`:
  - Added `download.verify_existing_size` mode for existing local files.
  - When `verify_checksum=false` and `verify_existing_size=true`, existing files are checked via remote `Content-Length` vs local size.
  - Added statuses:
    - `skipped_exists_size_match`
    - `skipped_exists_size_unverified`
    - `size_mismatch_redownload`
  - Added execution summary fields for verification mode flags.

- Updated docs in `physionet_ingest/README.md`:
  - Documented `download.verify_existing_size`.

- Added tests in `physionet_ingest/tests/test_download_physionet.py`:
  - Size-match skip path
  - Unknown-size graceful skip path
  - Full suite now passes.

## New targeted plan

Planner run:

- Command target: `CPC1=71`, `CPC5=71`
- Output config:
  - `physionet_ingest/config/config_i-care_2_1_longitudinal_targeted_topup_cpc1_71.yml`
- Output report:
  - `E:/Science_Datasets/physionet/metadata/i-care_2_1_random100_longitudinal/targeted_topup_plan_cpc1_71.json`
- Selected extra IDs: 33
- Projected counts after this top-up:
  - CPC1=71
  - CPC5=71

The generated top-up config was set to:

- `download.dry_run: false`
- `download.verify_checksum: false`
- `download.verify_existing_size: true`

## Runtime status

A new background download was started with:

- `physionet_ingest/config/config_i-care_2_1_longitudinal_targeted_topup_cpc1_71.yml`
- `--no-dry-run`

Startup logs confirm explicit 33-patient selection and normal initialization.
