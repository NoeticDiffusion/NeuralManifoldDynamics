# time-grid and derivative QA

Date: 2026-06-06

## Question

Implement two additional upstream sanity checks in `mndm`:

1. an always-on `dt/window_len` audit on the realized MNPS time grid
2. a reviewer-facing derivative self-consistency QA between `diff(mnps_3d)/dt` and exported `mnps_3d_dot`

## What changed

### 1. Added always-on `geometry_contract.time_grid`

The canonical `geometry_contract` now records a realized time-base audit:

- recovered inter-window `dt` from `window_start` (fallback: `time`)
- recovered window lengths from `window_end - window_start`
- match/mismatch booleans against:
  - runtime `dt` actually used for derivatives / Jacobians
  - config-derived `dt = window_sec * (1 - overlap)`
  - configured `window_sec`
- structural warnings for:
  - inconsistent array lengths
  - non-finite window bounds
  - non-positive `dt`
  - non-positive window lengths
  - runtime/config inconsistencies

This lives under:

- `summary.json.geometry_contract.time_grid`
- `qc_summary.json.geometry_contract.time_grid`
- HDF5 manifest / provenance export through the existing `geometry_contract` path

### 2. Added reviewer-facing derivative self-consistency QA

`mnps_mnj_sanity` now includes a `derivative_self_consistency` block that compares:

- finite-difference velocity from `diff(mnps_3d) / diff(time)`
- midpoint-averaged exported `mnps_3d_dot`

Reported diagnostics include:

- number/fraction of valid intervals compared
- interval `dt` summary
- finite-difference speed summary
- exported-derivative speed summary
- vector relative error summary
- symmetric speed-ratio summary
- cosine similarity summary
- edge-vs-interior relative-error summaries
- bad-interval fraction

The comparison is file-boundary aware, so cross-file intervals are excluded instead of being treated as false mismatches.

### 3. Wired through summarize export

The summarize pipeline now:

- attaches `time_grid` audit immediately after `window_start/window_end` are resolved
- threads `x_dot`, `time`, runtime `dt`, file labels, and derivative config into `compute_mnps_mnj_sanity`
- upgrades top-level `geometry_contract.status` to `adjusted` when the time-grid audit reports a warning

## Tests

Added/updated focused tests for:

- realized time-grid recovery
- derivative self-consistency mismatch detection
- `compute_mnps_mnj_sanity` with the new derivative/time arguments
- subject-runner export of `geometry_contract.time_grid`

Verification run:

- `python -m pytest mndm/tests/test_robustness.py mndm/tests/test_dataset_subject_runner.py -q`
- result: `38 passed`

## Interpretation

- `geometry_contract.time_grid` is part of the canonical export contract
- `mnps_mnj_sanity.derivative_self_consistency` is QA only, not a hard invalidity gate

This preserves the existing policy split:

- hard invalidity belongs in `geometry_contract`
- richer reviewer diagnostics belong in `mnps_mnj_sanity`
