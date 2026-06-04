# 091 20260603 derived block end provenance hardening

## Research question

How do we reduce the risk that a derived `stage_block_end` event is later read
as ground-truth block timing rather than as an operationalized endpoint defined
by the current `stage_blocking` policy?

## Changes made

- Hardened `StageBlockInterval` provenance in
  `mndm/src/mndm/pipeline/stage_blocking.py`.
  - Added `derived_from`
  - Added `end_reason`
  - Added `membership_mode`
  - Added `bridge_tail_sec`
  - Added `bridge_tail_cap_sec`
  - Added `is_inferred`
- Propagated these fields into synthetic derived-event metadata in
  `mndm/src/mndm/pipeline/event_annotations.py`.
- Added millisecond audit fields in the synthetic metadata:
  - `bridge_tail_ms`
  - `bridge_tail_cap_ms`
  - `block_start_ms`
  - `block_end_ms`
  - `block_duration_ms`
- Renamed example post-block bin labels away from fixed-width names:
  - `in_block_tail` -> `in_block_tail_ms`
  - `post_block_early` -> `post_block_early_ms`
  - `post_block_late` -> `post_block_late_ms`
- Updated docs, templates, and the `ds006036` example config accordingly.

## Validation

```powershell
pytest mndm/tests/test_event_annotations.py mndm/tests/test_event_alignment.py mndm/tests/test_event_locked_runner.py
```

Result: `35 passed, 1 skipped`

## Outcome

The derived block-end sidecar layer remains sidecar-only, but it is now more
audit-safe:

- the event explicitly says it is inferred
- the operational end mechanism is preserved
- the active membership/tail policy is preserved
- the example bin names no longer imply a fixed 8-second ontology
