# 104 - 2026-06-05 - Noetic anchor task surface and validation

## Research question
Can the `ds003838` noetic-anchoring path be completed end-to-end so that task/load labels, reviewer-facing sidecars, additive anchor-coupling diagnostics, and downstream null-control helpers all exist without changing the canonical MNPS contract?

## What was implemented
- Added `ds003838` task/load labeling directly in `mndm/src/mndm/features/eeg.py` from BIDS event text.
- Promoted `task_state_label`, `task_load_label`, and `task_load_n` into summary-time label exports in `mndm/src/mndm/pipeline/summary.py`.
- Added `build_label_segment_event_table(...)` in `mndm/src/mndm/pipeline/state_labels.py` so within-run label segments can drive event-locked analyses without redefining the H5 core.
- Wired automatic `event_locked` sidecar export for `ds003838` from derived `task_state_label` segments.
- Extended both sidecar surfaces:
  - `mndm/src/mndm/pipeline/event_locked_export.py`
  - `mndm/src/mndm/pipeline/block_native_export.py`
  so they carry anchor-state, anchor-derivative, anchor-quality, and aligned task-label columns.
- Added optional additive anchor-coupling estimation in `mndm/src/mndm/jacobian.py` and policy gating in `mndm/src/mndm/pipeline/robustness_helpers.py`.
- Wired anchor-coupling export into `summary.py` as `/anchor_coupling/*` with manifest policy reporting.
- Added downstream-first validation helpers in `mndm/src/mndm/pipeline/anchor_validation.py`:
  - grouped load/anchor summaries
  - within-subject time-shift nulls
  - subject-shuffle nulls

## Config surface
`mndm/config/config_ingest_ds003838.yaml` now includes:
- `within_run_labels.datasets.ds003838`
- `event_locked.datasets.ds003838`
- `anchor_coupling`

This keeps the stronger reviewer-facing analyses config-driven and additive.

## Validation
Targeted tests run:
- `python -m pytest mndm/tests/test_event_locked_export.py mndm/tests/test_block_native_export.py mndm/tests/test_state_labels.py mndm/tests/test_jacobian.py mndm/tests/test_anchor_validation.py -q`
- `python -m pytest mndm/tests/test_writers.py mndm/tests/test_run_manifest.py mndm/tests/test_dataset_subject_runner.py mndm/tests/test_anchor_state.py -q`

Results:
- `49 passed`
- `32 passed`

## Notes
- Event-locked reviewer tables are now driven from derived task-state segments, not by redefining the canonical `stage` contract.
- Anchor coupling remains optional and QC-gated.
- Anchor-conditioned reachability is still intentionally downstream-only; this session added export and null-control support, not a new ingest-time reachability contract.
