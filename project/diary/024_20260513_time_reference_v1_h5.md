# 024 - 2026-05-13 - Time Reference v1 in H5

## Research question
Can we add a backward-compatible, YAML-driven time-reference layer to MNPS H5 outputs for WFDB-based I-CARE runs, without changing canonical `/time`, `/window_start`, `/window_end`?

## What was implemented
- Added new module: `mndm/src/mndm/pipeline/time_reference.py`.
- Implemented config resolution for:
  - `time_reference.enabled`
  - `time_reference.schema_version`
  - `time_reference.parser` (v1: `wfdb_header`)
  - `time_reference.anchor` (v1 default: `first_recording`)
  - `time_reference.bins_hours`
  - dataset overrides via `time_reference.datasets.<dataset_id>.*`
- Implemented WFDB header parsing:
  - reads `#Start time` and `#End time`
  - parses clock values into seconds
  - supports rollover notation including `24:00:00+`
  - emits parse status and warnings instead of hard-failing summarize
- Implemented run-level and window-level aligned outputs:
  - run-level clocks + elapsed offsets
  - `window_start/end_from_run_sec`
  - `window_start/end_from_anchor_sec`
  - `window_bin_id` and `window_bin_label`
- Integrated into summarize flow in `mndm/src/mndm/pipeline/summary.py`:
  - writes `payload.extensions["time_reference"]`
  - writes concise `payload.attrs["time_reference_*"]`
  - writes `manifest_extra["time_reference"]`
- Updated run-manifest support in `mndm/src/mndm/pipeline/run_manifest.py`:
  - field-guide entries for `/extensions/time_reference/run/*` and `/extensions/time_reference/windows/*`
  - capability probing for time-reference presence (`capabilities.time_reference`)
  - count metrics (`counts.h5_with_time_reference`)
  - added `time_reference` in config excerpt
- YAML wiring:
  - `mndm/config/config_ingest_physionet_i-care_2_1.yaml`
  - `mndm/config/config_ingest_physionet_i-care_2_1_sleep_spindles.yaml`

## Tests and validation
- Added parser + contract tests:
  - `mndm/tests/test_time_reference.py`
  - updated `mndm/tests/test_run_manifest.py`
  - updated `mndm/tests/test_writers.py`
  - updated `mndm/tests/test_dataset_subject_runner.py` with summary integration coverage
- Local checks:
  - `pytest mndm/tests/test_time_reference.py` -> pass
  - syntax compile checks for all touched Python files -> pass
  - some H5-dependent tests are skipped in this environment when optional deps are unavailable

## Runtime validation (requested configs)
- Ran `features + summarize` for:
  - base I-CARE config
  - spindle overlay config
- Verified generated H5 files include:
  - `/extensions/time_reference/run/*`
  - `/extensions/time_reference/windows/*`
- Verified run manifest includes:
  - `capabilities.time_reference = true`
  - `counts.h5_with_time_reference > 0`
  - field-guide entries for time-reference paths
- Verified canonical paths remain unchanged:
  - `/time`, `/window_start`, `/window_end`

## Notes
- One initial summarize run failed due object-dtype string arrays in extension payload; fixed by switching time-reference string vectors to Unicode arrays (`<U...`) before H5 write.
- Coverage policy still skipped one short run (`run-001/acq-022`) for subject 0332 under current thresholds; this is expected behavior and independent of time-reference v1.
