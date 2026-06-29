Yes — I would absolutely add a **MNPS 3D/9D admissibility check**, but I would not solve this by brutally clipping the biology. The right distinction is:

**Biological extreme:** the trajectory becomes very flat, slow, stereotyped, or nearly one-dimensional.
**Mathematical impossible:** a ratio, inverse, condition number, recovery time, or persistence metric explodes because a denominator is effectively zero.

Your intuition is probably right: if persistence is reaching ~10^15, the immediate failure is likely downstream in reachability / kappa / recovery logic, but MNPS 3D and especially 9D should still be audited because all later Jacobian and reachability metrics inherit their numerical support from those trajectories. NeuralManifoldDynamics explicitly treats `mnps_3d`, `coords_9d`, and Jacobian exports as release-bound measurement objects, not direct biological truth, so this kind of measurement-contract audit is appropriate rather than cosmetic. 

## First: clarify the time base

There are two time bases in your documents:

The older/general EEG contract says **8 s windows with 50% overlap**, so step = **4 s**. 
But the spindle/event-locked run explicitly used **6 s MNPS windows with 2 s step**, and it warns that actual `/window_start` and `/window_end` should be trusted over stale H5 attributes. 

So for I-CARE I would not rely on memory or metadata alone. I would compute:

```text
dt = median(diff(window_start))
window_len = median(window_end - window_start)
```

and store both in the audit. If it is truly **8 s window / 2 s step**, that is 75% overlap, not the older default 50%.

## Core recommendation

Add three audit layers:

1. **Coordinate admissibility:** are MNPS 3D / 9D values finite, bounded, and temporally plausible?
2. **Derivative admissibility:** are `mnps_3d_dot` and 9D derivatives compatible with actual window step?
3. **Operator admissibility:** are Jacobian / reachability / persistence denominators numerically safe?

The most important rule:

> Do not convert near-singular biological collapse into enormous numeric values.
> Preserve it as a categorical or censored state: `near_singular = true`, `kappa >= cap`, `persistence_unresolved = true`.

## 1. MNPS 3D / 9D coordinate checks

Because the contract uses robust normalization and, in some versions, logistic squashing to `[0,1]`, coordinates should usually be finite and bounded after projection. The v1.2 mapping says features are robust-normalized, then squashed to `[0,1]`, with fixed projections to 3D and 9D. 

For each H5 run:

```text
x3 = /mnps_3d              shape [T, 3]
x9 = /coords_9d/values     shape [T, 9]
```

Check:

```text
finite_frac_3d
finite_frac_9d
min/max per axis
n_all_nan_windows
n_zero_variance_axes
per_axis_mad
per_axis_iqr
```

If bounded mode is active, enforce:

```text
-1e-6 <= x <= 1 + 1e-6
```

But do **not** reject a patient because one axis is nearly constant. In coma, flatness can be biologically meaningful. Reject or flag only when:

```text
non-finite
out-of-contract range
all axes constant due to preprocessing failure
insufficient feature support
manifest / actual window time mismatch
```

This matches the NMD design principle that missing weighted features and all-non-finite stratified coordinates should be surfaced explicitly rather than silently propagated. 

## 2. Step-size plausibility check

This is the simplest “mathematical impossibility” filter.

If coordinates are bounded in `[0,1]`, then the largest possible adjacent jump is:

```text
max_step_norm_3d = sqrt(3)
max_step_norm_9d = sqrt(9) = 3
```

Speed bound:

```text
max_speed_3d = sqrt(3) / dt
max_speed_9d = 3 / dt
```

So if `dt = 2 s`:

```text
3D max speed ≈ 0.866 units/s
9D max speed = 1.5 units/s
```

If `dt = 4 s`:

```text
3D max speed ≈ 0.433 units/s
9D max speed = 0.75 units/s
```

Anything far beyond this means one of:

```text
wrong dt
wrong derivative scaling
Savitzky-Golay edge artifact
trajectory discontinuity not segmented
unbounded coordinate mode
corrupt H5 / manifest mismatch
```

The derivative contract matters here because `mnps_3d_dot` is estimated by Savitzky-Golay derivatives, with fallback to central differences, and boundary derivatives are explicitly lower-confidence near short or split segments. 

I would therefore add:

```text
speed_fd = norm(diff(x), axis=1) / dt
speed_dot = norm(mnps_3d_dot, axis=1)

speed_ratio = speed_dot / (speed_fd_interpolated + eps)
```

Flag if:

```text
speed_dot > 2 * theoretical_bound
speed_ratio > 5
edge_window == true
segment_boundary == true
```

Do not necessarily remove those windows; mark them as low-confidence for Jacobian/reachability.

## 3. 9D-specific check

The 9D layer is especially likely to produce downstream instability because local regression in 9D is harder. Your own stress-test text already notes that 9D MNJ is vulnerable to curse-of-dimensionality effects and should require overdetermined neighborhoods and condition diagnostics. 

For 9D, I would require:

```text
k_neighbors >= 4 * dim
```

So:

```text
3D: k >= 12 minimum, preferably 20+
9D: k >= 36 minimum, preferably 50+
```

For each local fit, store:

```text
design_rank
design_condition_number
min_singular_value
residual_norm
local_variance_min
local_variance_max
```

Flag:

```text
rank < dim
s_min < eps_abs
condition_number > 1e8
condition_number > 1e12 = invalid for ratio metrics
```

Very important: if 9D explodes but 3D remains sane, that is probably not a biological contradiction. It means the 9D local operator is under-supported or nearly singular. Then the 9D result should be treated as supportive/sensitivity only, exactly as the coma manuscript already does for repaired 9D Jacobian evidence. 

## 4. Persistence / kappa fix

The likely source of `10^15` is something like:

```text
kappa = lambda_max / (lambda_min + eps)
```

or:

```text
condition_number = sigma_max / sigma_min
```

Your compact summary defines reachability anisotropy / condition as:

```text
kappa = lambda_1 / (lambda_d + epsilon)
```

and persistence/recovery via a tube metric over time. 

That is exactly where near-zero denominators create astronomical values.

I would replace raw persistence/kappa reporting with this policy:

```text
eps_abs = 1e-10
eps_rel = 1e-6 * lambda_max
lambda_floor = max(eps_abs, eps_rel)

if lambda_min < lambda_floor:
    kappa_raw = lambda_max / max(lambda_min, eps_abs)
    kappa_report = kappa_cap
    kappa_censored_high = true
    near_singular = true
else:
    kappa_report = lambda_max / lambda_min
    kappa_censored_high = false
```

Use:

```text
kappa_cap = 1e8 or 1e10
```

For paper-facing summaries, report:

```text
log10_kappa_capped
near_singular_fraction
kappa_censored_fraction
```

rather than raw kappa.

A biologically locked coma state can therefore become:

```text
near_singular_fraction high
log10_kappa_capped high
```

not:

```text
kappa = 1e15
```

That preserves the biological extreme without pretending the numeric magnitude is meaningful.

## 5. Recovery / persistence-time bound

If a recovery metric asks:

```text
tau_rec = first Δt where V(t + Δt) >= V_rec
```

then failure to recover should not become huge. It should be:

```text
tau_rec = NaN
tau_rec_censored = true
tau_rec_max_observed = analysis_horizon
```

or, if you need a numeric model feature:

```text
tau_rec_report = horizon_cap
tau_rec_censored = true
```

For a 0–12 h coma window, the maximum meaningful persistence time is bounded by the analyzed support. A value like `1e15` seconds is not coma physiology; it is “not recovered within observed horizon” encoded incorrectly.

## 6. Suggested audit table

For each run, export something like:

```text
subject_id
run_id
n_windows
window_len_median
dt_median
dt_iqr

x3_finite_frac
x9_finite_frac
x3_min
x3_max
x9_min
x9_max

x3_max_step_norm
x9_max_step_norm
x3_max_speed_fd
x9_max_speed_fd
x3_speed_bound_violation_frac
x9_speed_bound_violation_frac

dot3_max_norm
dot3_speed_ratio_p99
dot_boundary_flag_frac

x3_axis_min_mad
x9_axis_min_mad
x3_zero_var_axis_count
x9_zero_var_axis_count

J3_condition_p95
J3_condition_p99
J3_near_singular_frac
J9_condition_p95
J9_condition_p99
J9_near_singular_frac

reach_lambda_min_p01
reach_lambda_max_p99
kappa_raw_max
kappa_log10_capped_median
kappa_censored_frac
tau_rec_censored_frac
```

Then use a simple classification:

```text
PASS:
  coordinates finite
  dt valid
  no impossible speed
  enough local support

BIOLOGICAL_EXTREME:
  coordinates valid
  speed valid
  local covariance/Jacobian near-singular across many neighboring windows

NUMERICAL_UNSTABLE:
  derivative exceeds theoretical bound
  isolated one-window explosion
  rank deficient local fit
  denominator collapse without stable neighboring support

INVALID:
  non-finite coordinates
  out-of-contract coordinate range
  missing/incorrect time base
  insufficient windows
```

## My practical proposal for I-CARE

For the coma pipeline, I would add a pre-summary script named something like:

```text
audit_mnps_admissibility.py
```

and run it before any reachability or persistence aggregation.

Then rerun the persistence metrics in three versions:

1. **raw legacy** — only for debugging, never paper-facing
2. **capped log version** — `log10(kappa)` with cap and censor flag
3. **categorical singularity version** — `near_singular_fraction`, `unrecovered_fraction`

The paper-facing claim should move from:

```text
persistence_kappa is extremely large
```

to:

```text
a subset of windows/runs enters a near-singular local-capacity regime, reported by capped log-kappa and near-singularity fraction
```

That is much more defensible.

## Bottom line

Yes, check both **MNPS 3D and 9D**, but the main guardrail should not be “clip away extremes.” It should be:

```text
bounded coordinates
verified dt
finite derivative
valid local support
safe denominator policy
censored near-singularity reporting
```

If MNPS coordinates are valid but persistence explodes, the foundation is probably okay and the persistence layer needs denominator/censoring repair. If MNPS step-size or derivative bounds fail, then the issue is deeper: time base, segmentation, derivative estimation, or coordinate export.
