# 099 20260604 block_native_threshold_tuning_ds003506_ds003509

## Context

After full dataset runs for `ds003506` and `ds003509`, we tuned
`block_native.datasets.<ds>.source` thresholds to better match observed
block-duration distributions.

Goal: keep phase-like blocks, suppress obvious short fragments, and cap extreme
long tails without destabilizing per-subject exports.

## Data used for tuning

Used emitted `/blocks/duration_sec` from completed runs:

- `ds003506`: `neuralmanifolddynamics_ds003506_20260604_141227`
- `ds003509`: `neuralmanifolddynamics_ds003509_20260604_144631`

Observed emitted distributions:

- `ds003506` (`n=91` blocks): min ~20.4s, p25 ~352.1s, median ~375.8s,
  p90 ~449.6s, max ~518.0s.
- `ds003509` (`n=663` blocks): min ~34.4s, p25 ~99.5s, median ~210.3s,
  p90 ~281.0s, max ~461.6s.

## Config changes

Updated:

- `mndm/config/config_ingest_ds003506.yaml`
- `mndm/config/config_ingest_ds003509.yaml`

### ds003506 (`block_native.datasets.ds003506.source`)

- `gap_tolerance_sec`: `10.0 -> 8.0`
- `min_block_sec`: `20.0 -> 60.0`
- `max_block_sec`: `600.0 -> 480.0`

Rationale:

- keep typical long phase blocks (~350-450s),
- reject short fragments (<60s),
- trim rare extreme tails above ~8 minutes.

### ds003509 (`block_native.datasets.ds003509.source`)

- `gap_tolerance_sec`: `10.0 -> 8.0`
- `min_block_sec`: `20.0 -> 60.0`
- `max_block_sec`: `600.0 -> 360.0`

Rationale:

- keep typical training/test phase spans (~100-300s),
- remove short fragments (<60s),
- cap uncommon long-tail merges (>360s).

## Validation

YAML parse checks passed for both updated configs.

Smoke runs (single subject) completed successfully:

- `python -m mndm.cli summarize --dataset ds003506 --config mndm/config/config_ingest_ds003506.yaml --subject 001 --n-jobs 1`
  - run dir: `M:\datasets\processed\openneuro\ds003506\neuralmanifolddynamics_ds003506_20260604_175959`
  - block_native: `1 blocks -> 197 windows`
- `python -m mndm.cli summarize --dataset ds003509 --config mndm/config/config_ingest_ds003509.yaml --subject 001 --n-jobs 1`
  - run dir: `M:\datasets\processed\openneuro\ds003509\neuralmanifolddynamics_ds003509_20260604_180051`
  - block_native: `8 blocks -> 647 windows`

Spot check on emitted block durations after tuning:

- `ds003506/sub-001`: 1 block, duration ~397.9s
- `ds003509/sub-001`: 8 blocks, min ~89.9s, median ~158.7s, max ~243.2s

## Notes

- Cohort-anchored coordinate contract remains requested-but-not-realized for
  these configs because no active feature anchor is configured; this tuning only
  targets block-native phase segmentation thresholds.
