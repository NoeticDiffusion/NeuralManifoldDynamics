# Dynamical families

`mndm.dynamical_families` is separate from MNPS coordinates and
Jacobian-derived `mndm.dynamics` measurements.

Canonical write paths:

```text
/dynamical_families/diffusion/v1
/dynamical_families/destination/v1
/dynamical_families/resilience/v1
```

Legacy read-only paths (pre-v3 files):

```text
/orthogonal_dynamics/diffusion_geometry/v1
/orthogonal_dynamics/committor/v1
/orthogonal_dynamics/finite_amplitude_resilience/v1
```

Schema IDs are unchanged. Spread as a family YAML key
(`mndm.stochastic_reachability.v1`) remains Gate F `gate_closed` and is not
written under `/dynamical_families`. Opt-in discrete `W_Q` is
`/stochastic_reachability/v1` via `local_dynamics.stochastic_reachability`.

Each family v1 group writes three sibling certificate fields:
`computation_status`, `measurement_validity`, and `claim_status`.
`provenance.validation_level` (`mndm_translation_validated` /
`simulator_validated`) is a method-validation tag, not
`measurement_validity`. New writes set `claim_status` to
`no_biological_claim` and never emit `ndt_licensed`.
`measurement_validity=translation_qualified` requires a computed **destination
or resilience** object plus an already recorded TQ id and contract hash.
Diffusion writes `not_assessed` when computed; OD-TQ1 tags live in
`provenance` and do not gate computation. Legacy readers must use
`not_recorded` for missing fields and must not infer regime validity or NDT
licensing. Missing `claim_status` may be copied from `provenance/claim_status`
when that dataset exists; `certificate_origin` records whether the three
siblings were already present.

Round 3 adds nested `grain/` on each family v1 group. Diffusion and destination
are `native=window`; resilience is `native=event`. `biological_unit` is
`subject`; `direct_between_subject_inference` is `forbidden`. Grain is schema
metadata and is written even when `computation_status` is not `computed`.
Legacy readers must use `not_recorded` for missing grain fields and must not
infer `window` from series.

## Diffusion geometry

The diffusion tensor estimates local covariance of raw state increments
(`a_semantics=raw_increment_covariance`). It is not the
transition-residual covariance proxy. Jacobian derivative residuals are not
accepted as diffusion \(a\) (`jacobian_derivative_residual_as_diffusion`).
MNPS \(\dot x\) is not an SDE drift (`mnps_xdot_as_sde_drift`). Jacobian
intercepts and same-sample increment means are also refused as \(b\).

`contract_status=standard` is the schema-contract class on provenance. It is
not an "experimental vs licensed" scientific tag and is not
`measurement_validity`.

Ingest always calls the estimator with C1 defaults (`drift=None`,
`residualize_increments=False`). A `computed` object is increment-covariance
geometry, not a testable \(A_{bD}\) / \(R_{b/a}\) object. Those series remain
NaN and are labelled `summary.A_bD_computation_status=not_testable` and
`summary.R_b_over_a_computation_status=not_testable` with
`drift_alignment_failure_reason=independent_drift_not_supplied`. Do not
treat NaN as zero alignment. Library C1 may align an externally qualified
chart-space velocity to that increment covariance without changing `a_hat`.
Ingest does not estimate \(b\); nmd-analysis owns identification (SL-005).
Under C1, `ratio_semantics=chart_velocity_to_increment_spread`; this is not
an Itô drift-to-diffusion ratio. C2 residualization is not authorized on
ingest. An empirical C1 overlay is not authorized.

All diffusion estimates remain chart-dependent. When the family is enabled,
ingest computes the increment-covariance object if estimator support holds
and writes `measurement_validity=not_assessed`. Truth-known OD-TQ1 tags in
`provenance` are method-validation, not an empirical-interpretation license
and not a YAML on/off switch. Common profiles do not enable the family.

## Destination (committor)

The committor module requires explicit, disjoint A/B regime sets and multiple
independent segments containing both regimes. Stage labels alone are not
committor truth. Production ingest uses the 1-D O2b adapter
(`local_law_dense_grid_o2b`) only. That adapter uses an internal 1-D
potential for quadrature; it does **not** serialize \(V_{1/2}\). The
first-hit estimator (`local_first_hit_outcome_average`) serializes
`q_A_to_B` from observed hits and also does not emit \(V_{1/2}\).
Neither estimator emits \(\lvert\nabla q\rvert\). Do not treat a computed
`q` as a 2-D/3-D/9D MNPS committor. The first-hit estimator is not a
substitute for simulator Monte-Carlo or generator-based validation.

## Resilience (finite-amplitude)

Resilience accepts observed finite-perturbation outcomes. It does not infer
basin stability from spontaneous trajectories, a Jacobian, or finite-time
tangent gain. Standard retrospective ingest therefore reports it as not
testable with `no_perturbation_protocol` when requested without explicit
perturbation data. A computed result additionally requires an explicit
outcome protocol and a valid family qualification certificate.
