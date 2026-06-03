# Session Diary 048 - Normalization validation probes + family-wise ComBat

Date: 2026-05-27

## Goal

Implement the next small normalization increment for summarize:

- add a `normalization.validation` block in YAML configs/templates,
- run automatic pre/post probes in runtime,
- emit `normalization_report.json` and expose it in `run_manifest.json`,
- add family-wise ComBat grouping in runtime.

## What was implemented

### 1) Runtime: validation probes and family-wise ComBat

Updated `mndm/src/mndm/pipeline/summary.py`:

- Added normalization validation config resolution (`normalization.validation`) with sane defaults.
- Added deterministic row/feature probe sampling for large tables:
  - `max_rows`,
  - `max_features`,
  - seeded from run reproducibility seed.
- Added automatic pre/post probe computations:
  - batch-effect proxy via per-feature eta^2 (`batch_eta2`),
  - optional target-label eta^2 probes (`target_eta2`) for `target_keys`,
  - perturbation magnitude probe (`perturbation`) with robust shift summaries.
- Added family-wise ComBat grouping:
  - `combat.family_wise.enabled`,
  - `strategy` (`prefix` or `regex_map`),
  - `delimiter`,
  - `min_family_columns`,
  - optional `regex_map`.
- Extended normalization report payload (`self._normalization_report`) with:
  - `family_wise` summary (families/chunk stats/harmonized counts),
  - `validation` probe report (pre/post + deltas).

### 2) New sidecar: normalization_report.json

Added `_write_normalization_report_file()` and wired it into `DatasetSummaryRunner.run()`:

- always writes `normalization_report.json` in the run directory (once output dir exists),
- stores write status/path back into normalization report,
- adds manifest field `extra.normalization_report`.

### 3) YAML updates

Added the new schema block in config templates/common EEG:

- `mndm/config/config_template.yaml`
- `mndm/config/eeg_config_ingest_template.yaml`
- `mndm/config/config_ingest_common_eeg.yaml`
- `mndm/config/config_ingest_physionet_i-care_2_1.yaml`

Also enabled validation + family-wise ComBat in the active i-care ComBat run configs:

- `mndm/config/config_ingest_physionet_i-care_2_1_part1_0_12h_regional.yaml`
- `mndm/config/config_ingest_physionet_i-care_2_1_next_140_0_12h_no_regional.yaml`

### 4) Output docs update

Updated `mndm/Output_variables_guide.md` to include:

- `normalization_report.json` as a run-level sidecar,
- `run_manifest.json -> extra.normalization_report` field,
- expanded `extra.normalization` notes for `family_wise` and `validation` probe outputs.

## Tests and checks

Updated `mndm/tests/test_dataset_subject_runner.py`:

- expanded ComBat test to assert:
  - family-wise report fields,
  - computed validation probes.
- added test for writing `normalization_report.json`.
- extended run-failure smoke test to assert:
  - `normalization_report.json` is present,
  - `run_manifest.json` includes `extra.normalization_report.path`.

Executed targeted tests:

- `python -m pytest mndm/tests/test_dataset_subject_runner.py -k "combat_normalization or writes_normalization_report_file or writes_manifest_and_run_errors_on_group_failure"`
- Result: `2 passed, 1 skipped`.

Read lints on touched Python files: no linter errors.

## Notes

- Validation probes are designed to be lightweight and bounded for multi-million-row feature tables.
- Family-wise grouping defaults to disabled in shared templates/common config, but is enabled in the two current i-care ComBat configs for next summarize runs.
