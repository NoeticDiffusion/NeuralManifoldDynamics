# 053_20260527_ds006036_event_point_mapping_check

## Research question
- Verify that `ds006036` photomark events are handled correctly before launching full `mndm` (`features -> summarize`).

## What changed
- Updated `mndm/config/config_ingest_ds006036.yaml`:
  - Added `epoching.datasets.ds006036.sampling` override so stage extraction reads `*_events.tsv` column `value`.
  - Added event label -> code mapping for:
    - `PHOTO 5Hz/10Hz/15Hz/20Hz`
    - `open eyes`, `closed eyes`, `eye movement`
  - Extended `mnps.stage_codebook` with the same labels/codes.
- Fixed point-event handling (`duration=0`) so events map to containing epoch windows instead of strict midpoint equality:
  - `mndm/src/mndm/features/epoch_selection.py`
  - `mndm/src/mndm/pipeline/summary_events.py`
- Added regression tests:
  - `mndm/tests/test_epoch_selection_point_events.py`
  - `mndm/tests/test_sleep_stage_labels.py` (new point-event test)

## Validation
- Tests:
  - `pytest tests/test_epoch_selection_point_events.py tests/test_sleep_stage_labels.py -q`
  - Result: `4 passed`.
- Smoke run (`sub-001`) on isolated output root:
  - Features output contains `stage` column with non-empty event-derived labels.
  - Stage distribution observed: `-1, 51, 52, 53, 60, 61, 62` (not all `-1`).
  - Summarize output confirms:
    - `stage_source = "features_csv"`
    - `stage_column = "stage"`
    - `stage_frac_labeled = 0.37037`

## Notes / limitations
- Current representation is a single `stage` code per epoch; overlapping event semantics are reduced to one active code (last matching event in window).
- `events/*` arrays remain empty unless explicit `event_*` feature columns are produced and `event_mapping` is enabled.

## Next step recommendation
- Run full `ds006036` chain with updated config.
- If multi-label event traces are needed (parallel photic + eye-state labels), add explicit event-array extraction (`event_*`) as a follow-up.

