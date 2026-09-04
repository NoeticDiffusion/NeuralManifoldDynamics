# Gate F — admissible one-step Q for \(W_Q\)

Status: **frozen 2026-09-02**. Opt-in ingest emission is authorized only under
this freeze. Not an NDT license. Not coma `tube_d_eff_median`.

---

## Decision

```text
Admissible Q for computed W_Q
  = Gate E recording-level transition-residual covariance
    with conversion_model = not_applicable

Not admissible
  = derivative-residual covariance
  = any Q that requires a conversion model
  = diffusion a_hat
  = Gate E Q treated as Itô process noise
```

Gate E Q is a **discrete one-step innovation** in chart-state units. It is
not process noise, not diffusion \(a(x)\), and not a silent \(dt^2\)
rescaling of \(\dot x\) residuals.

The registry forbid `gate_e_proxy_as_process_noise` **stays**. Using the
proxy as \(W_Q\) input does not authorize calling it process noise.

---

## Admissibility contract (already in `compute_stochastic_reachability`)

All of:

```text
computation_status     = computed
q_time_semantics       = one_step_transition_covariance
q_units                = state_squared
conversion_model       = not_applicable
q_dt_sec               finite and positive
```

`make_derivative_residual_covariance_proxy` sets
`conversion_model=required_before_stochastic_reachability` and
`q_time_semantics=derivative_residual_covariance_proxy`. It remains refused.

Irregular observed steps make the Gate E Q proxy `unavailable` /
`materially_irregular_dt`. Then \(W_Q\) is `unavailable`, not a converted
derivative object.

---

## Propagators

\(W \leftarrow \Phi W \Phi^\top + Q\) uses the **same** one-step linear maps
whose cross-fitted residuals pooled into \(Q\):

```text
Φ = expm(J_crossfit Δt)
```

`J_crossfit` is the Gate E leave-one-transition-out affine Jacobian, not
canonical `/jacobian/J_hat`. Mean intercept does not enter the covariance
recursion.

These Φ are stored on Gate E as `series/phi_one_step`. Missing Φ →
`unavailable` / `missing_one_step_propagators`.

Do not invent `discrete_transition_product` without independent transitions.
Do not require FTR to be enabled (FTR stays default off).

---

## Ingest path

```text
local_dynamics.stochastic_reachability.enabled: false   # default
```

Requires Gate E `transition_residuals.enabled`. Write:

```text
/stochastic_reachability/v1/{primary,stratified_9d}
schema mndm.stochastic_reachability.v1
```

Not under `/dynamical_families/spread`. Family YAML key `spread` remains
refused. Grain: `recording_horizon`. Certificate: `not_assessed` /
`no_biological_claim` when computed.

Common EEG/fMRI/ephys profiles do not enable this flag. Coverage failures
(e.g. ds003944) stay coverage failures.

---

## Claim ceiling

```text
Safe:     Chart-space discrete reachability ellipsoid from Gate E Q and Φ.
Not:      Latent K^h, licensed NDT reachability, I-CARE tube estimators,
          process noise, diffusion a(x), consciousness.
```
