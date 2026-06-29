# sub-0435 derivative threshold calibration

Date: 2026-06-06

## Question

Can the new `mnps_mnj_sanity.derivative_self_consistency` thresholds be calibrated so the block remains useful for pathological runs without warning on every exported `sub-0435` run?

## Calibration setup

Used the refreshed live rerun:

`E:/Science_Datasets/physionet/processed/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260606_181425`

Defined a simple surrogate split:

- **pathological**: `geometry_contract.jacobian_9d.windows_retained == 0`
- **cleaner**: primary Jacobian retained and 9D Jacobian retained

This yielded:

- pathological runs: `2`
- cleaner runs: `79`

## Main result

The separation on `sub-0435` is very clean.

### Pathological runs

- `run-001_acq-017`
- `run-002_acq-018`

Observed ranges:

- `bad_interval_fraction`: `0.455 - 0.604`
- `vector_rel_error p95`: `10.35 - 3.91e5`
- `speed_ratio p95`: `11.16 - 7.62e5`

### Cleaner runs

Observed ranges:

- `bad_interval_fraction`: `0.189 - 0.313`
- `vector_rel_error p95`: `1.25 - 1.85`
- `speed_ratio p95`: `4.22 - 7.45`

So there is a visible gap in all three metrics between the pathological pair and the cleaner exported runs.

## Candidate warning rules

All of the following selected only the pathological pair and none of the cleaner runs:

- `bad_interval_fraction >= 0.35`
- `bad_interval_fraction >= 0.40`
- `vector_rel_error_p95 >= 3.0`
- `vector_rel_error_p95 >= 5.0`
- `speed_ratio_p95 >= 8.0`
- `speed_ratio_p95 >= 10.0`
- `vector_rel_error_p95 >= 3.0 OR bad_interval_fraction >= 0.35`

## Important implementation insight

The current helper uses the same values for two different purposes:

1. per-interval badness definition
2. top-level run warning trigger

That coupling is why the block currently warns on all `81 / 81` exported `sub-0435` runs.

## Recommended policy

Keep the current interval-level thresholds for the raw QA metrics:

- interval relative error threshold: `1.0`
- interval speed ratio threshold: `5.0`

But use separate run-level warning thresholds, for example:

- `vector_rel_error_p95 >= 3.0`, or
- `speed_ratio_p95 >= 8.0`, or
- `bad_interval_fraction >= 0.35`

## Interpretation

This would preserve sensitivity in the exported raw diagnostics while making top-level `status = warning` much more selective on the `sub-0435` slice.

It is still only an internal calibration result on one subject, but it is a much better starting point than the current coupled thresholds.
