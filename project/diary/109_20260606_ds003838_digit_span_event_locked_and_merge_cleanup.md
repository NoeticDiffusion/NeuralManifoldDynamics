# ds003838 digit_span event_locked and merge cleanup

## Research question

Close the last smoke-run gaps after the noetic anchoring integration:

1. remove duplicate-column warnings from multimodal temporary/intermediate feature exports, and
2. verify that `sub-032` `digit_span` now emits its derived `event_locked` sidecar.

## What changed

- Hardened `mndm/src/mndm/parallel.py` multimodal frame merging:
  - duplicate aligned metadata columns such as `t_start` / `t_end` are now coalesced instead of recursively suffixing,
  - duplicate-name deduping now avoids collisions with already-suffixed names such as `__dup1`,
  - staged `embodied_arousal_proxy` columns are dropped before recomputing the final cross-modality resolver.
- Hardened `mndm/src/mndm/pipeline/state_labels.py`:
  - byte-valued labels from H5-backed payloads are now decoded before segment construction,
  - derived task-state event tables no longer emit labels like `b'listen'` / `b''`.
- Hardened `mndm/src/mndm/pipeline/control_matching.py`:
  - quartile matching now tolerates non-finite timeline bounds,
  - candidate control windows are restricted to finite timeline entries.

## Validation

- Added `mndm/tests/test_parallel_merge.py` to cover nested multimodal merges and ensure shared timing columns are not duplicated.
- Extended `mndm/tests/test_state_labels.py` with a bytes-label regression.
- Extended `mndm/tests/test_control_matching.py` with a non-finite timeline regression.
- Focused validation passed:
  - `python -m pytest mndm/tests/test_state_labels.py mndm/tests/test_control_matching.py mndm/tests/test_parallel_merge.py -q`
  - `python -m pytest mndm/tests/test_parallel_merge.py mndm/tests/test_bids_index_multimodal.py mndm/tests/test_preprocess_modalities.py -q`

## Smoke findings

- Direct runner repro against the existing `sub-032_digit_span` H5 now succeeds:
  - 96 derived task-state events,
  - 576 event rows,
  - 192 matched-control rows,
  - CSV and Parquet sidecars both write successfully in the targeted repro.
- Fresh full `mndm.cli all` rerun for `ds003838` / `sub-032` no longer emitted the earlier duplicate-column warnings.
- The fresh rerun also reached automatic `digit_span` event-locked export and logged:
  - `Control matching: 96 full, 0 partial, 0 no-match`
  - `Event-locked table: 576 event rows + 192 control rows = 768 total`
  - `Wrote event-locked CSV: .../sub-032_digit_span/event_locked.csv`

## Remaining caveat

The fresh rerun hit a new environmental blocker unrelated to the code fixes: the output volume ran out of space during later `sub-032_digit_span` H5 writing. Because of that, the rerun was not fully clean end-to-end and the output set under `.smoke_processed_clean2/` is partial. The code-path fixes themselves were validated before the disk-full failure:

- duplicate-column warnings were gone, and
- `digit_span` automatic `event_locked` emission occurred before the H5 write failure.
