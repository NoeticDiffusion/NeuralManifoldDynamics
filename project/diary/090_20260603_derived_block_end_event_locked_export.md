# 090 20260603 derived block end event locked export

## Research question

Can MNDM expose a generic sidecar-first event-locked workflow that derives
point-events from inferred stage-block ends, so post-block effects can be
analyzed without changing the primary HDF5/summarize contract?

## Completed work

- Added `event_table_from_stage_block_intervals(...)` in
  `mndm/src/mndm/pipeline/event_annotations.py`.
  - Emits one synthetic point-event per inferred block interval.
  - Carries `block_id`, `stage_code`, `block_parameter`, source event index,
    and support marker indices in `metadata_json`.
- Extended `mndm/src/mndm/pipeline/event_locked_config.py`.
  - Added `EventSourceConfig`.
  - Added `event_source_config_from_config(...)`.
  - Added `event_source_kind` to `EventLockedProfile` provenance.
  - Fixed custom event-locked bin reconstruction so profile-stored bins no
    longer fall back to default bins.
- Added `mndm/src/mndm/pipeline/event_locked_runner.py`.
  - Resolves either CSV-backed event tables or
    `event_source.kind: "derived_stage_block_end"`.
  - Reuses `sampling.stage_blocking` to infer intervals from raw events TSV/CSV.
  - Runs alignment, matched controls, flat table export, and manifest assembly.
- Generalized `mndm/src/mndm/pipeline/event_locked_export.py`.
  - Export rows now use generic `condition` labels (`event`,
    `matched_control`) rather than spindle-only naming.
  - Manifest now reports `n_event_rows` plus a compatibility alias
    `n_spindle_event_rows`.
- Updated `project/smoke_tests/batch_event_locked.py` to use the reusable
  runner instead of open-coding the alignment/control/export sequence.

## Tests

Focused coverage added/updated in:

- `mndm/tests/test_event_annotations.py`
  - `StageBlockInterval -> EventTable` synthesis
- `mndm/tests/test_event_alignment.py`
  - point-event alignment into pre/post bins around a block end
- `mndm/tests/test_event_locked_export.py`
  - generic `event` row condition and manifest counts
- `mndm/tests/test_event_locked_runner.py`
  - derived block-end resolution from stage blocking
  - end-to-end alignment + CSV sidecar export

Validation run:

```powershell
pytest mndm/tests/test_event_annotations.py mndm/tests/test_event_alignment.py mndm/tests/test_event_locked_export.py mndm/tests/test_event_locked_runner.py
```

Result: `63 passed, 1 skipped`

## Documentation and config surface

- Updated:
  - `README.md`
  - `mndm/README.md`
  - `mndm/Output_variables_guide.md`
  - `mndm/config/config_template.yaml`
  - `mndm/config/eeg_config_ingest_template.yaml`
  - `mndm/config/config_ingest_ds006036.yaml`
- `ds006036` now carries a first-pass `event_locked` example using
  `event_source.kind: "derived_stage_block_end"` with bins around inferred
  block ends.

## Contract notes

- v1 remains sidecar-only.
- No new HDF5 group was added.
- `/labels/stage`, `/events`, and `/event_windows` semantics are unchanged.
- Derived block-end provenance lives in the synthetic `EventTable` metadata and
  the exported Parquet/CSV rows.
