# sub-0435 reachability adapter trace

Date: 2026-06-06

## Question

Why did `sub-0435` persistence remain effectively unchanged in the analysis repo after the new standard geometry invalidity policy was added to `mndm`?

Focus run:

- `sub-0435_post_cardiac_arrest_coma_continuous_eeg_run-001_acq-017`

Compared exports:

- Old run: `G:/Science_Datasets_longtime_storage/processed/physionet_part1_0_12h_regional_combat/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260601_131927`
- New run: `E:/Science_Datasets/physionet/processed/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260605_202236`

## Main result

The new policy **did change the exported 9D Jacobian**, but it **did not change the state-space that the reachability adapter uses**.

That explains why persistence stayed the same.

## Evidence

### 1. New export flags the run as geometrically adjusted

In the new `summary.json` / `qc_summary.json` for `run-001`:

- `geometry_contract.shared_time_grid.epochs_dropped = 0`
- `geometry_contract.coords_9d.degenerate_axes = ["d_n", "m_a", "m_e"]`
- `geometry_contract.jacobian_9d.invalid_windows = 21`
- `geometry_contract.jacobian_9d.windows_retained = 0`

So the new policy correctly detects that the 9D geometry is degenerate and invalidates all exported 9D Jacobian windows.

### 2. But the 9D coordinates themselves are unchanged

Old vs new H5 root attrs for `run-001`:

- `coords_9d_hash_saved` is identical
- `coords_9d_hash_knn_input` is identical
- `coords_9d_hash_jacobian_input` is identical
- `x_hash_saved` is identical
- `x_hash_knn_input` is identical
- `x_hash_jacobian_input` is identical

Interpretation: the coordinate trajectory used as state-space input is the same in old and new exports.

### 3. Only exported `jacobian_9D` changed

Old H5:

- `/jacobian_9D/J_hat` present with shape `[21, 9, 9]`

New H5:

- `/jacobian_9D/J_hat` absent
- `jacobian_9d_hash_saved` becomes the SHA256 of the empty payload

So the policy successfully removes the invalid exported 9D Jacobian, but leaves the coordinate time series unchanged.

### 4. The analysis repo reachability adapter uses coordinates, not exported Jacobians

In `J:/repos/NoeticDiffusion/ndt-analysis/ndt_analysis/reachability_cones.py`:

- `_resolve_x_from_h5()` calls `resolve_coords_9d()` / `resolve_coords_3d()`
- `summarize_reachability_h5()` passes the resolved coordinate matrix `x` into `compute_reachability_cones_from_mnps(...)`

In `reachability_core.py`:

- `compute_reachability_cones_from_mnps()` rebuilds nearest neighbours and local linear models directly from `x`
- persistence metrics are then computed from those re-estimated reachability covariances

Crucially, this path does **not** consume the exported `geometry_contract` and does **not** rely on the exported `/jacobian_9D/J_hat`.

### 5. The H5 contract reader does not surface geometry invalidity as a gate

In `J:/repos/NoeticDiffusion/ndt-analysis/ndt_analysis/h5_contract.py`:

- `read_run_contract()` exposes coordinate-contract / anchor / schema provenance
- `resolve_coords_9d()` returns the resolved coordinate matrix
- `resolve_jacobian_9d()` returns the exported Jacobian if present

But there is no geometry-contract-based refusal path here for degenerate `coords_9d`.

## Conclusion

This is now an internally validated explanation:

- The new `mndm` policy **does** catch `sub-0435 run-001` as 9D-invalid.
- It **does** suppress the exported 9D Jacobian.
- But the downstream reachability adapter recomputes its own local dynamics from the still-unchanged `coords_9d` trajectory.
- Therefore persistence remains effectively unchanged.

## Practical implication

If the goal is to make the persistence explosion disappear downstream, one of the following must happen:

1. `mndm` must also alter or gate the **state-space export** used by reachability, not only the exported 9D Jacobian.
2. The analysis repo must read and enforce `geometry_contract`, refusing or masking degenerate 9D coordinate runs before reachability is computed.
3. The reachability code itself must add an explicit degeneracy / singular-covariance policy.

## Most likely next step

Implement an analysis-side gate first:

- when `geometry_contract.coords_9d.degenerate_axes` is non-empty, or
- when `geometry_contract.jacobian_9d.windows_retained == 0`,

then mark 9D reachability persistence outputs as invalid / `NaN` for that run instead of recomputing persistence from the degenerate coordinate matrix.
