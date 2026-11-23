# MNPS 3.4 dynamical-family export

`v3_4_dynamics.enabled` opts into drift, diffusion and lag-1 one-step exports
under `/measurements/v3_4/dynamics`. The established `dynamical_families`
configuration and its 3D subject-anchored outputs keep their existing behavior.

The exporter creates separate `v3_4_raw` and `v3_4_anchored` realizations. Raw
coordinates remain computable when an anchor is absent. Their Euclidean
neighborhoods and cross-axis summaries use mixed source units, which the HDF5
provenance records. Anchored coordinates are used only when the declared
external anchor was applied and its ID, content hash and manifest hash are
available. One-step uses only that anchored realization. A run with no applied
anchor retains raw dynamics and reports the anchored realization as
`not_testable`.

The diffusion family records the local conditional increment-covariance rate
`a_hat` using its configured nominal `dt`, and also stores a separate
recording-level `increment_covariance_level0` without division by `dt`. These
are chart-dependent covariance measurements; neither is asserted to be a
latent or identified Itô diffusion tensor.

Comparisons between anchored recordings require a shared anchor ID and matching
anchor and manifest hashes. The exporter preserves those identities for the
downstream comparison policy to check.

The default neighborhood size is 32 in nine dimensions. The minimum support is
28. Drift removes a query's own transition after neighbor lookup, so its usable
support can be smaller than `k`. Drift defaults to uniform weights. Equal
distances are ordered by source-row index for the opt-in v3.4 estimator calls.
Incomplete coordinate windows and irregular time steps split transition
segments. The window-step tolerance is recorded in configuration and
provenance. No coordinate is filled to create an increment.

One-step is a recording-level affine lag-1 map. Its existing support and
blocked-holdout requirements remain active. The result records fitted-window
support and the blocked evaluation. When a 9D temporal training block falls
below the declared 90-pair support floor, it records
`below_declared_support_floor` with the available pair count and emits no
affine map for that block. Pair count alone does not establish design-matrix
rank. The fit itself is not a local Jacobian, and it does not emit a lag-2
map.

MNPS 3.4 uses eight-second windows on a four-second grid in the current EEG
surface. These families measure evolution of overlapping window features at
the declared window step. The outputs retain window-level validity and support
and do not establish independence of those windows.

Each realization stores its coordinate layer, ordered names, unit note,
window centers, segment IDs, complete-window mask, input hashes, effective
family configuration and hash, implementation-source hashes, and Python,
NumPy, SciPy and h5py versions. The bounded example is
`config_ingest_ds004511_v34_dynamics_pilot.yml`.

## Experimental MEG integration

The opt-in configuration `config_ingest_ds003645_meg_v34_dynamics_e2e_k.yml`
attaches derived exports inside each
`/measurements/v3_4_meg/sensor_families/{mag,grad}/variants/{variant}/`
branch. MAG and GRAD are processed independently, and no family or coordinate
variant is pooled with another. Its drift, diffusion and one-step settings
match the bounded ds004511 v3.4 dynamics pilot (`k=32`, minimum 30 samples,
minimum neighborhood 28, nominal step 4 seconds, 0.05 relative step tolerance).
MEG Jacobian activation uses the modality-specific `v3_4_meg_jacobian` block,
so an inherited EEG `v3_4_jacobian.enabled` setting does not activate it.

MEG coordinates currently have no applied MEG-specific anchor. Drift and
diffusion therefore run only on the raw coordinate realization, with
`mixed_source_units` recorded. This verifies estimator computability on that
declared chart; it does not qualify the metric or make MAG and GRAD comparable.
One-step remains `not_testable` because it requires anchored coordinates. The
Jacobian runs on each supported 9D variant with whitening disabled. The 7D
`removal_only` variant is retained with `not_testable` status for Jacobian; no
coordinate is padded. Diffusion accepts that 7D realization, but the current
finite-lag drift estimator accepts only a 9D v3.4 chart or a 3D
subject-anchored chart, so drift for `removal_only` is explicitly
`not_testable`. Incomplete windows stay masked and split temporal segments.

This MEG path is opt-in and leaves the existing legacy Jacobian/dynamics
surfaces and historical MEG feature path unchanged. Its smoke configuration
uses the already-created ds003645 `sub-002`, FacePerception run-1 feature table
and writes the summarize result to a separate K: output root.

The 2026-11-19 ordinary summarize smoke completed for 121 windows. The output
window axis and all raw coordinate values/support masks in all three variants
matched the cached feature table exactly (absolute tolerance `1e-12`); MAG and
GRAD remained separate with 102 and 204 sensors. Jacobians computed for both
9D variants (119 centers for `existing_coordinate_set`, 114 for
`alpha_replacement`); the 7D Jacobian was `not_testable`. For each family,
drift and diffusion computed for both 9D variants. On the complete baseline,
realized velocity was valid for 120/121 windows, pooled conditional mean rate
for 121/121, blocked crossfit for 112/121, and diffusion for 121/121. On the
alpha replacement, the corresponding supports were 115/121, 116/121, 107/121,
and 116/121. The 7D removal-only diffusion computed for 121/121 windows while
its drift retained `not_testable` status. Its 9D replacement had five expected
edge exclusions (epochs 0–2 and 119–120). The anchored dynamics and one-step
operator explicitly report `not_testable` because no MEG anchor was applied.

After separating MEG Jacobian activation from the inherited EEG setting, a
final-config adapter-to-HDF5-writer smoke re-used the same cached feature table
and confirmed all six MAG/GRAD × variant combinations retained those estimator
statuses. Its sidecar is
`K:/ExternalReceivedDatasets/openneuro/processed_ds003645_meg_v34_dynamics_smoke_20261119/adapter_writer_smoke/sub-002_meeg_FacePerception_run-1/mnps_v3_4_meg_measurements.h5`.
The full ordinary summarize run above predates that opt-in-key refinement; the
final key was verified through the production adapter and HDF5 writer.

This run left `NUMBA_DISABLE_JIT` unset. No parity claim is made against the
previous JIT-disabled MEG pilot. The HDF5 and run metadata are under
`K:/ExternalReceivedDatasets/openneuro/processed_ds003645_meg_v34_dynamics_smoke_20261119/ds003645/neuralmanifolddynamics_ds003645_20261119_134328/`.
