# Minimal EEG H5 contract upgrade

Date: 2026-06-02

## Question

Can the EEG H5 export become more self-describing for event-locked and
between-subject analysis without breaking existing readers, paths, or JSON
sidecars?

## Implemented

Added a minimal, additive EEG H5 contract layer on top of the existing export.

New optional payload/H5 groups:

- `/event_windows/*`
  - explicit event-to-window alignment rows with `event_id`, `window_id`,
    `event_label`, exact onset/window bounds, relative-time bins, and
    `window_contains_event_onset`
- `/codebooks/stage/*`
  - explicit stage codebook arrays: `codes`, `labels`, `label_keys`
- `/participant/clinical_json`
  - richer additive participant/session metadata embedded directly into H5
- `/coverage/*`
  - explicit direct-axis coverage surface and layer-presence metadata
- `/provenance/*`
  - structured export-contract, anchoring, normalization, and event/stage
    provenance blocks
- `/qc/windows/*`
  - light-weight per-window QC flags aligned to `/time`

The implementation keeps existing paths unchanged:

- `/events/*` remains the source event table
- `/labels/*` remains valid
- `/mnps_3d`, `/coords_9d`, `coords_*_anchored`, Jacobians, feature surfaces,
  and JSON sidecars are preserved

## ds003490-specific outcome

The new additive contract now supports the Parkinson oddball use case more
directly:

- EO/EC stage codebooks are exported explicitly under `/codebooks/stage`
- concise helper labels such as `eyes_closed` and `eyes_open` are emitted in
  `/labels`
- oddball events like `Standard Tone`, `Novel Tone`, and `Target Tone` get an
  explicit row-per-join event-window table under `/event_windows`
- the contract exposes both exact event onset timing and matched window indices,
  avoiding downstream nearest-neighbor reconstruction from `/events` to `/time`

## Files changed

- `mndm/src/mndm/schema.py`
- `core/src/core/io/h5_writer.py`
- `mndm/src/mndm/pipeline/summary.py`
- `mndm/src/mndm/pipeline/run_manifest.py`
- `mndm/tests/test_schema.py`
- `mndm/tests/test_writers.py`
- `mndm/tests/test_sleep_stage_labels.py`
- `mndm/tests/test_epoch_selection_point_events.py`
- `mndm/Output_variables_guide.md`
- `mndm/README.md`

## Commands

```powershell
python -m pytest mndm/tests/test_schema.py mndm/tests/test_writers.py mndm/tests/test_sleep_stage_labels.py mndm/tests/test_epoch_selection_point_events.py
```

## Result

Targeted tests passed:

- `25 passed`

Observed warnings:

- `4` existing runtime warnings from `mndm/src/mndm/jacobian.py` about
  all-NaN slices in Jacobian diagnostics during synthetic test cases

## Evidence category

- Internal validated result:
  - additive EEG H5 groups now carry explicit event-window joins, codebooks,
    coverage, provenance, participant clinical metadata, and per-window QC
  - `ds003490` now has a dedicated regression test for EO/EC codebook export
    and explicit tone-event to window linkage
  - focused schema, writer, summarize, and epoch-selection tests pass
