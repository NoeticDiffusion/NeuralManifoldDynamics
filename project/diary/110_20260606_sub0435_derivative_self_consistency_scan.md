# sub-0435 derivative self-consistency scan

Date: 2026-06-06

## Question

After adding:

- `geometry_contract.time_grid`
- `mnps_mnj_sanity.derivative_self_consistency`

run a live summarize rerun on `sub-0435` and check how those new fields behave across the refreshed outputs.

## Live rerun

Command:

`python -m mndm.cli summarize --dataset physionet_icare_2_1 --config "mndm/config/config_ingest_physionet_i-care_2_1.yaml" --subject 0435 --n-jobs 1`

Run root:

`E:/Science_Datasets/physionet/processed/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260606_181425`

Outcome:

- summarize completed successfully
- `81` `summary.json` / `qc_summary.json` outputs were written for `sub-0435`
- `run-032_acq-047` was still skipped by normal coverage gating (`epochs=2`)
- `run-002_acq-018` exported in degraded mode with duplicate 9D provenance instead of crashing

## Main findings

### 1. Time-grid audit looks clean

Across all `81` exported runs:

- `geometry_contract.time_grid.status = "ok"` for all runs
- recovered `dt` median was uniformly `4.0 s`
- recovered `window_len` median was uniformly `8.0 s`
- no time-grid warnings were observed

Interpretation:

`sub-0435` does not look like a `dt/window_len` bookkeeping problem.

### 2. Derivative QA is active on real data, but broad

Across all `81` exported runs:

- `mnps_mnj_sanity.derivative_self_consistency.status = "warning"` for all runs

This means the current thresholds are sensitive enough to catch the pathological runs, but not selective enough to isolate only those.

### 3. Worst runs

Clear worst case:

- `run-002_acq-018`
  - `bad_interval_fraction = 0.604`
  - `vector_rel_error p95 = 3.91e5`
  - `speed_ratio p95 = 7.62e5`
  - `coords_9d.duplicate_pairs = {"m_e": "m_a"}`
  - primary Jacobian retained `0 / 897`
  - 9D Jacobian retained `0 / 897`

Second-worst:

- `run-001_acq-017`
  - `bad_interval_fraction = 0.455`
  - `vector_rel_error p95 = 10.35`
  - 9D Jacobian retained `0 / 21`

Moderate but still warning:

- many other runs clustered around `bad_interval_fraction ~ 0.23 - 0.31`
- many had `vector_rel_error p95 ~ 1.4 - 1.8`
- some of these runs still retained essentially all Jacobian windows

## Interpretation

The new fields are doing two useful things:

1. `time_grid` confirms the exported temporal grid is internally consistent.
2. `derivative_self_consistency` strongly surfaces the catastrophic cases.

But the derivative QA is currently calibrated broadly enough that it warns on every exported `sub-0435` run, including runs that do not show catastrophic Jacobian collapse.

## Practical next step

If this block is meant to be a sharper instability discriminator rather than a broad reviewer warning, the next step should be threshold calibration using:

- a small pathological set (`run-001`, `run-002`)
- a small cleaner set (for example `run-060` and similar retained-Jacobian runs)

and then comparing how `bad_interval_fraction`, `vector_rel_error p95`, and `speed_ratio p95` separate those regimes.
