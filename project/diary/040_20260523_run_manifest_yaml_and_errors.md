## Session: run-manifest robustness + config provenance

Date: 2026-05-23

### Research question
How can summarize runs keep provenance and diagnostics usable when a subset of recordings fail?

### Implemented changes
- Added run-level config provenance in `run_manifest`:
  - active YAML is copied into the run output directory,
  - copied filename/path/status is recorded under `run_manifest.json -> config.yaml_source`.
- Added resilient summarize error capture:
  - per-grouping failures are captured instead of aborting the entire run,
  - failures are written to `run_errors.json` with error type/message/traceback and grouping identifiers.
- Ensured `run_manifest.json` is still emitted when failures occur:
  - manifest now includes `extra.run_status` (`completed`, `completed_with_errors`, `failed`),
  - manifest links `extra.run_errors` metadata (count/path/status),
  - fatal dataset-level failures are summarized under `extra.fatal_error`.
- Wired CLI/orchestration so summarize receives the original config path for YAML copy provenance.

### Validation (smoke)
Executed targeted smoke tests:
- `tests/test_run_manifest.py::test_run_manifest_copies_yaml_and_records_filename`
- `tests/test_dataset_subject_runner.py::test_dataset_runner_writes_manifest_and_run_errors_on_group_failure`
- `tests/test_orchestrate_summarize.py::test_cmd_summarize_attaches_config_path_to_context`

Result: `3 passed` (targeted smoke run).

### Evidence categories
- Internal validated result:
  - targeted unit/smoke tests passed for YAML copy provenance and partial-failure manifest/error outputs.
- Plausible interpretation:
  - this should prevent the previous "no run_manifest when subset fails" workflow break in summarize runs.

### Known boundaries
- If summarize fails before a run directory is created, no run-level files can be emitted.
- Existing run-manifest schema version is retained (`mndm.run_manifest.v2`) with additive fields.

### Next step
- Optional: run one small real dataset summarize with an injected failure to confirm end-to-end behavior outside unit tests.
