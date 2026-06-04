# Generic interval membership core

## Research question

Can event-derived stage blocks be made framework-generic by separating:

1. interval inference from sparse event streams,
2. interval-to-window membership geometry,
3. dataset-specific config choices

so that `ds006036` is just one config instance rather than a special-case implementation?

## Work completed

1. Added a shared interval geometry layer in `mndm/src/mndm/pipeline/intervals.py`.
2. Added a shared stage-block inference/config-normalization layer in `mndm/src/mndm/pipeline/stage_blocking.py`.
3. Refactored both runtime consumers onto the shared helpers:
   - `mndm/src/mndm/features/epoch_selection.py`
   - `mndm/src/mndm/pipeline/summary_events.py`
4. Refactored `mndm/src/mndm/pipeline/event_alignment.py` to reuse the shared overlap helpers.
5. Added tests for:
   - fully-contained block membership,
   - overlap-threshold block membership,
   - half-open overlap boundary semantics.
6. Updated docs/templates so the new API is described as generic framework behavior.

## Implementation notes

### Shared interval geometry

`mndm/src/mndm/pipeline/intervals.py` now owns the core window/interval semantics:

- `TimeInterval`
- `WindowMembershipSpec`
- `overlap_sec(...)`
- `overlap_frac(...)`
- `window_membership_mask(...)`
- `event_window_mask(...)`

Default behavior preserves the existing contract:

- point events map to the window containing onset,
- duration intervals use midpoint-in-interval membership unless config says otherwise.

### Shared stage-blocking config + inference

`mndm/src/mndm/pipeline/stage_blocking.py` now owns:

- canonical parsing of `stage_blocking` config,
- backward-compatible alias handling,
- regex capture of numeric block parameters,
- inference of absolute block intervals from carrier + bridge events.

Canonical config now supports:

- `bridge_tail_sec` / `bridge_tail_cap_sec`
- `window_membership.mode`
- `window_membership.min_overlap_fraction`

Legacy aliases remain accepted:

- `photic_regex`
- `hv_mark_labels`
- `use_hv_marks`
- `hv_tail_sec`
- `hv_tail_cap_sec`
- `preserve_photic_blocks`
- `expected_frequencies_hz`

### Important semantic fix

During test expansion, one boundary leak was identified:

- block-start carrier events could still label the onset window directly,
  even when stricter block membership (`fully_contained` / overlap threshold)
  intended to exclude that boundary window.

This was fixed by treating carrier rows as block-definition events once
`stage_blocking` is active, rather than letting them re-enter through the
point-event fallback path.

## Internal validated results

### Unit tests

Passed:

- `pytest mndm/tests/test_event_alignment.py mndm/tests/test_epoch_selection_point_events.py mndm/tests/test_sleep_stage_labels.py`

Observed result in this environment:

- `19 passed, 1 skipped`

The skip is expected here because `test_sleep_stage_labels.py` now uses
`pytest.importorskip("scipy")`, and the current local environment does not
have `scipy` installed. The file remains runnable in full environments where
that dependency is available.

### New test coverage added

- strict `fully_contained` block membership drops boundary epochs/windows
- `overlap_frac_ge` block membership allows threshold-based inclusion
- overlap helpers preserve half-open interval semantics at touching boundaries

## Documentation updates

Updated:

- `README.md`
- `mndm/README.md`
- `mndm/Output_variables_guide.md`
- `mndm/config/config_ingest_common_eeg.yaml`
- `mndm/config/config_template.yaml`
- `mndm/config/eeg_config_ingest_template.yaml`

Key clarification added:

- `mnps.overlap` controls MNPS stride overlap during window construction
- `stage_blocking.window_membership.min_overlap_fraction` controls interval
  geometry when assigning already-constructed windows to inferred blocks

## Outcome

- The reusable seam is now explicit in code.
- `ds006036` no longer requires a dataset-specific code path to support
  stricter “pure block-internal windows” behavior.
- Enabling stricter photic membership is now a config decision rather than a
  new implementation branch.
