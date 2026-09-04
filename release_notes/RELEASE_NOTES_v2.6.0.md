# NeuralManifoldDynamics v2.6.0 — Local Jacobian metrics and finite-time response

v2.6.0 extends the stable `mnps_3d`, `coords_9d`, anchored-coordinate, and
Jacobian-estimator contracts with versioned interpretation layers derived from
existing `J_hat`. The estimator itself, coordinate definitions, and anchor
semantics are unchanged. This release adds per-recording mathematical
measurements and explicit validity/provenance; it does not add cohort
statistics or biological claims.

The companion measurement-contract manuscript and its supplements remain in
`project/articles/NeuralManifoldDynamics/`.

## Highlights

**Jacobian Metrics v1**
- Added `mndm.jacobian_metrics.v1` for primary 3D and directly estimated
  stratified 9D Jacobians.
- New HDF5 paths: `/jacobian/derived_metrics/v1/` and
  `/jacobian_9D/derived_metrics/v1/`.
- Exports semantic stability, reactivity, symmetric-rate, magnitude,
  deformation, rotation, and support-count measurements with explicit
  coordinate-contract provenance.

**Finite-Time Response v1**
- Added opt-in `mndm.finite_time_response.v1` at
  `/finite_time_response/v1/`.
- `time_ordered_expm` is primary; `frozen_j_expm` is a comparator.
- Original Jacobian-center gaps and observed time gaps are hard boundaries.
  Results record requested/actual horizons, support, propagation semantics,
  `computation_status`, and `validation_level`.

**Residual covariance and stochastic reachability contracts**
- Added `mndm.residual_covariance_proxy.v1` and
  `mndm.stochastic_reachability.v1` library/export contracts.
- Derivative residual covariance is not an admissible stochastic-reachability
  input without an explicit continuous-to-discrete conversion.
- The summarize pipeline currently has no exported one-step transition
  residual, so it does not emit computed `W_Q` reachability results.

**Run provenance**
- Run manifests can discover the local-dynamics paths and capabilities.
- Local-dynamics outputs preserve their primary coordinate contract and do not
  merge subject- and cohort-anchored series.

## Validation and claim boundaries

| Dataset / stream | Evidence in v2.6.0 | Claim ceiling |
|---|---|---|
| Local-dynamics tests | Analytic Jacobian, continuity, writer, and Q-gate checks | Mathematical contract only |
| ds003944 / ds003947 | New per-recording summarize outputs | Internal descriptive output; not a group-level clinical claim |
| Finite-time response | `model_derived` by default | Not predictive or perturbational validation |
| Derivative-residual Q proxy | Explicitly gated | Not a `W_Q` input without a conversion model |

No claim in this release extends beyond these boundaries. Exploratory and
dataset-specific outputs remain distinct from validated biological inference.

## Upgrading

- Existing MNPS, 9D, anchor, and Jacobian paths remain compatible.
- Jacobian Metrics v1 is an additive derived surface when `J_hat` is available.
- Enable finite-time response with `local_dynamics.finite_time_response.enabled:
  true`; see `mndm/CONFIG_GUIDE.md`.
- No config permits silent derivative-residual to stochastic-noise conversion.

## Full changelog

See [`CHANGELOG.md`](CHANGELOG.md) for cumulative history and
`project/articles/NeuralManifoldDynamics/S6_Changelog.typ` for the
manuscript-integrated changelog.
