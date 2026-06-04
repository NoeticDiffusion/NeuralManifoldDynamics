# ds006036 fully-contained photic test

## Research question

What happens in practice if `ds006036` moves from midpoint-based inferred photic
block membership to strict `fully_contained` window membership?

## Config change

Updated `mndm/config/config_ingest_ds006036.yaml`:

- switched `stage_blocking.hv_tail_sec` to canonical `bridge_tail_sec`
- added `bridge_tail_cap_sec: 1.0`
- added:

```yaml
stage_blocking:
  window_membership:
    mode: "fully_contained"
```

This means an 8 s MNPS window must lie completely inside the inferred photic
block to receive that photic stage label.

## Validation before rerun

Ran targeted unit tests:

- `pytest mndm/tests/test_epoch_selection_point_events.py mndm/tests/test_event_alignment.py`

Result:

- `19 passed`

## Fresh rerun

Output root:

- `E:/Science_Datasets/openneuro/processed_ds006036_strict_fully_contained_v1`

Command:

1. `python -m mndm.cli features --dataset ds006036 --config H:/SourceRepo2/NeuralManifoldDynamics/mndm/config/config_ingest_ds006036.yaml --out-dir E:/Science_Datasets/openneuro/processed_ds006036_strict_fully_contained_v1 --n-jobs 6`
2. `python -m mndm.cli summarize --dataset ds006036 --config H:/SourceRepo2/NeuralManifoldDynamics/mndm/config/config_ingest_ds006036.yaml --out-dir E:/Science_Datasets/openneuro/processed_ds006036_strict_fully_contained_v1 --n-jobs 6`

Run produced:

- `E:/Science_Datasets/openneuro/processed_ds006036_strict_fully_contained_v1/ds006036/neuralmanifolddynamics_ds006036_20260603_114851`

Status:

- exit code `0`
- `h5_with_stage: 88`
- `labels_stage: true`

## Comparison against prior subject-anchor baseline

Baseline:

- `E:/Science_Datasets/openneuro/processed_ds006036_subject_then_cohort_anchor_v2/ds006036/neuralmanifolddynamics_ds006036_20260528_093207`

### Aggregate stage coverage

- old mean `stage_frac_labeled`: `0.584536`
- new mean `stage_frac_labeled`: `0.533997`
- absolute delta: `-0.050539`
- relative delta: `-8.65%`

### Total labeled windows

- old labeled windows: `3422`
- new labeled windows: `3178`
- delta: `-244` (`-7.13%`)

### Total photic windows only

Photic codes counted:

- `57, 50, 58, 51, 52, 53, 55, 56`

Result:

- old photic windows: `1856`
- new photic windows: `979`
- delta: `-877` (`-47.25%`)

### Per-code photic window counts

- `PHOTO 3Hz` (`57`): `6 -> 3`
- `PHOTO 5Hz` (`50`): `574 -> 269`
- `PHOTO 7Hz` (`58`): `2 -> 0`
- `PHOTO 10Hz` (`51`): `454 -> 261`
- `PHOTO 15Hz` (`52`): `435 -> 252`
- `PHOTO 20Hz` (`53`): `357 -> 184`
- `PHOTO 25Hz` (`55`): `16 -> 4`
- `PHOTO 30Hz` (`56`): `12 -> 6`

## Interpretation

- The strict rule works as intended: photic labeling becomes much more selective.
- The reduction is large for photic windows specifically (~47%), which is
  consistent with boundary windows being removed rather than a general pipeline failure.
- Overall stage coverage drops only moderately (~8.7%), so the stricter rule is
  meaningfully cleaner without collapsing the usable dataset.
- Frequency detection/QC remains intact:
  - detected raw frequencies still cover `[3, 5, 7, 10, 15, 20, 25, 30]`
  - missing expected frequencies remain `[]`
  - subjects with raw 25 Hz remain `6`
  - subjects with raw 30 Hz remain `2`

## Conclusion

`fully_contained` appears to be a viable “clean photic epochs” policy for
`ds006036` when the goal is to suppress boundary contamination. The main tradeoff
is a substantial drop in photic-labeled window count, especially for rare
frequencies such as 7, 25, and 30 Hz.
