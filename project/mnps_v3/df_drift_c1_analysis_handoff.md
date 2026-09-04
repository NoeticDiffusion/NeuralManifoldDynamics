# DF-DRIFT-C1 analysis handoff

Status: **RATIFIED** after Science Lead amendments (2026-09-03).
Authority: `project/mnps_v3/science_lead/sl_mnps3_005.md` (2026-09-02).
SL-005 **overrides** `sl_mnps3_004.md` on the next gate: M1–M2 remains
**PASS / RATIFIED**, but M3 is **not authorized in ingest**.

Companion in nmd-analysis:
`NeuralManifoldDynamics/dynamical_families/analysis_handoff.md`.

Ingest endpoint: diary `288_20260902_df_drift_c1_m1_m2.md`.
This note is the fork, not a new ingest estimator. After this ratification,
stop changing the Type C ingest implementation. Next DF-DRIFT work is
**M3 design in nmd-analysis**.

```text
INGEST  frozen C1 consume-external-b + a_hat identity
ANALYSIS  M3 identification, later M4/M5, and leftover science gates
FAR     parked BLOCKED_EXTERNAL_CUSTODY
```

---

## Boundary

Long-term repo-boundary rule (SL-005, preserve verbatim):

> If an operation requires choosing among scientifically plausible
> estimators, splitting data to establish independence, tuning a
> validation protocol, constructing null controls, or interpreting a
> condition contrast, it belongs outside ingest.

Complementary ingest-admissibility rule:

> An ingest primitive is admissible when the mathematical object and
> required inputs are already specified, and the operation deterministically
> transforms those inputs into a versioned measurement artifact without
> choosing the scientific interpretation.

```text
INGEST if:
specified inputs
+ deterministic transformation
+ versioned output
+ fail-closed validity checks

ANALYSIS if:
estimator choice
or identification
or data splitting
or comparator/null construction
or scientific model selection
or hypothesis inference
```

```text
INGEST
qualified inputs
      ↓
deterministic/versioned measurement primitive
      ↓
auditable artifact


ANALYSIS
data
 ↓
identification / estimator choice
 ↓
cross-fitting / controls / nulls
 ↓
scientific qualification
 ↓
possibly qualified input to ingest primitive
```

```text
INGEST / NeuralManifoldDynamics
Owns:
- MNPS 3D / 9D construction
- versioned schemas and HDF5
- deterministic measurement primitives
- family estimator APIs
- provenance / support / grain / validity metadata
- fail-closed gates
- "I can consume a chart-space velocity field supplied under
   an externally governed qualification"
- structural compatibility checks on that field
- refusal of x_dot / MNJ intercept / same-sample increment mean
- C1 alignment semantics
- no empirical drift estimation

Does NOT own:
- how b is estimated from a dataset
- cross-fitting strategy
- embargo selection
- estimator comparisons
- same-sample contamination experiments
- shuffled controls
- empirical qualification of b
- adjudicating scientific independence or M3 PASS
- condition contrasts or inferential statistics
- deciding whether C1 is scientifically useful
```

```text
ANALYSIS / nmd-analysis
Owns:
- candidate chart-drift estimators
- crossfit_local_chart_b (algorithm, not ingest YAML)
- train/evaluation partitioning
- embargo rules
- overlap / contamination audit
- contaminated comparator
- shuffled/time-displaced controls
- truth-known / synthetic benchmark harness
- dataset-specific feasibility
- M3 / M4 / M5
- scientific qualification certificate
- statistical inference
- eventual empirical overlay request back to ingest
- remaining science gates listed below
```

Distinction:

```text
analysis certifies
ingest records + validates structural compatibility
```

Ingest may validate shape, finite support, units/time semantics, time
alignment, allowed provenance token, and presence of qualification
fields. Ingest must not claim scientific independence, unbiased
estimation, or M3 PASS because those fields are present.

---

## Ingest handoff contract (available today)

```text
Available:
- MNPS chart trajectory
- timestamps / dt
- increment-covariance a_hat at /dynamical_families/diffusion/v1
  a_semantics = raw_increment_covariance
- C1 alignment API
- provenance fields
- forbidden-source rules

Analysis must supply (if C1 is used empirically / at M3):
- b_hat array
- b_hat content hash
- exact trajectory/timebase alignment
- source MNPS artifact hash
- source NMD schema/version
- NMD code/package revision used for C1 evaluation
- estimator id/version
- estimator provenance
- qualification id/hash
- train/evaluation partition provenance
- train/evaluation index hashes
- embargo specification
- overlap audit

Analysis owns:
- crossfit estimator
- embargo
- oracle/comparator/null experiments
- M3 qualification decision

Ingest does not:
- fit b
- choose embargo
- cross-validate
- infer NDT drift
- adjudicate scientific qualification
```

The contamination certificate must not be a declarative flag alone.
M3 should prove and serialize something auditable, for example:

```text
train_increment_ids ∩ evaluation_increment_ids = ∅
embargo violations = 0
```

with hashes or counts supporting those statements. A future analysis
must not be able to write `same_increment_contamination = false` without
an artifact that can be audited.

Library C1 (synthetic M1–M2, frozen):

```python
estimate_local_diffusion_geometry(
    ...,
    drift=external_field_or_none,
    residualize_increments=False,
)
```

Gate C1-A: truth-known chart-space \(b\) leaves `a_hat`, `D_total`,
`d_diff`, `c_diff`, and the valid mask byte-equal to `drift=None`.

Production ingest still calls this with `drift=None`. YAML
`diffusion.drift.enabled` remains false. Common EEG/fMRI/ephys profiles
do not import the family YAML. ds004100 OD-EPI-001 is an `a_hat`
translation overlay, not a drift overlay.

Provenance when an externally governed field is consumed:

```text
drift_mode = alignment_only
drift_residualization = none
a_semantics = raw_increment_covariance
ratio_semantics = chart_velocity_to_increment_spread
```

Those settings record structural mode, not scientific qualification.
\(R_{b/a}\) is secondary under C1. The denominator is increment
covariance, not residualized Itô \(a\).

Closed forbids (do not pass these as `b`):

```text
mnps_xdot
jacobian_intercept
local_increment_mean
jacobian_residual
committor_potential_gradient
```

C2 `residualize_increments=True` is `invalid` /
`c2_residualize_increments_not_authorized`. The token
`crossfit_local_chart_b` may later describe provenance of an
**externally** supplied field. The algorithm that produces it must not
live in ingest.

`chart_drift.py` validates structural compatibility of an externally
supplied object and refuses forbidden provenance. It is not a
drift-estimation module and does not certify scientific independence.

Overlay lifecycle (do not donate analysis estimator code back to ingest
just because a measurement becomes useful):

```text
1. INGEST
   exposes safe primitive

2. ANALYSIS
   identifies candidate b

3. ANALYSIS M3
   qualifies estimator synthetically

4. ANALYSIS M4
   tests empirical feasibility

5. SCIENCE LEAD
   decides whether this object has earned an external qualification

6. INGEST
   may later serialize it under an explicit overlay
   without owning the estimator
```

The input can remain externally produced.

---

## DF-DRIFT-C1 M3 (analysis)

Compare, on a truth-known or later empirical MNPS chart trajectory:

```text
truth-known b               ← positive control
cross-fit b                 ← candidate estimator
same-sample local mean b    ← contaminated comparator
shuffled/time-displaced b   ← negative control
```

Measure:

```text
alignment recovery
bias in A_bD
rank recovery
variance
sensitivity to embargo
sensitivity to dt
failure under insufficient support
train/eval overlap = empty
embargo violations = 0
```

M4 (later): one empirical C1 dataset-selection gate. **Not** ds004100
ictal contrast and **not** a flag on OD-EPI-001.

M5 (later): downstream utility only after M4.

C2 residualized \(a\) is **not authorized** until a new science-lead
gate. Analysis must not request ingest residualization.

---

## Other leftover science gates (nmd-analysis)

These are still open in `project/mnps_v3`. They are analysis (or parked
external custody), not ingest estimator work.

| Item | Analysis owns | Ingest does not |
|---|---|---|
| DF-DRIFT-C1 M3/M4/M5 | cross-fit, dataset scout, utility, overlap audit | implement `b` estimators |
| C2 residualized `a` | nothing until a new science-lead gate | enable residualization |
| ds004100 D-M4 | any later scientific contrast is a **new** prereg | retrofit OD-EPI-001 into drift/ictal biology (PASS is serialization/QC only; 319/319 `a_hat`; `claim_status=no_biological_claim`) |
| Gate F \(W_Q\) | whether the discrete chart-space predictive-spread object is scientifically useful | treat Gate E residual covariance as process noise, NDT reachability, I-CARE tubes, or consciousness |
| Committor RC / A–B | which RC, which A/B, new overlay prereg | 2-D/3-D/9D MNPS `q`; stage labels as truth; emit \(V_{1/2}\) or \(\lvert\nabla q\rvert\) |
| FAR \(R(\rho)\) / FAR-003+ | interpretability, recovery definition, empirical curve **after** E1+E2 XML | infer FAR from Jacobian/FTR/\(W_Q\); live 002E while `BLOCKED_EXTERNAL_CUSTODY` |
| Grain / inference | refuse \(N=\) window-count; keep recording/subject grain | change grain metadata to rescue power |
| Deferred charter families | do not invent \(I_{\mathrm{pred}}\) / TE / FDT / Fisher / \(M(\tau)\) as NMD exports | build them to “complete the chart” |
| Chart robustness | audit-only condition-sign stability | alternative MNPS charts |

### ds004100

```text
D-M4 PASS
    =
chart-space increment-covariance serialization feasibility

NOT
    ictal biology
    latent a(x)
    drift
    Type C
```

D-M4 (`d_m4_ds004100_diffusion_feasibility.md`) **PASS** on v3 replay
`20260902_050532`. That licenses chart-space increment-covariance
serialization under OD-TQ1 as a **method tag**, not ictal biology, not
latent \(a(x)\), not Type C drift.

### Gate F / \(W_Q\)

Frozen in `gate_f_admissible_q.md` (2026-09-02). That freeze authorizes
an **opt-in ingest primitive**, not an NDT license and not common-profile
enablement.

```text
Gate F / W_Q

The implemented opt-in primitive may propagate a qualified
recording-level discrete transition-residual covariance together
with a cross-fit transition operator to form a chart-space
predictive-spread object at /stochastic_reachability/v1.

input_semantics =
    discrete_transition_residual_covariance

NOT:
    process_noise_covariance
    continuous-time a(x)
    NDT SDE Q
    historical I-CARE reachability semantics
```

This does **not** identify the covariance as continuous-time process
noise \(a(x)\), does **not** identify an NDT SDE \(Q\), and does **not**
license historical I-CARE reachability (e.g. `tube_d_eff_median`).
Registry forbid `gate_e_proxy_as_process_noise` stays. Using the Gate E
proxy as \(W_Q\) input does not authorize calling it process noise.

The **spread family** remains `gate_closed` for production / common-profile
YAML. Opt-in `W_Q` is `local_dynamics.stochastic_reachability`, not
`/dynamical_families/spread`. Analysis may later ask whether that
discrete chart-space object is scientifically useful; that question is
not an ingest estimator change.

### Committor

Ingest primitive: MNPS chart trajectory + **explicit** 1-D RC + explicit
A/B + certificate → `q`. No production overlay without a new prereg.
BOAS G=17 pooled law failed local predictive criteria. ds004100
destination is NOT_TESTABLE. ds004511 closed.

### FAR

Ingest primitive: perturbation/outcome table carrying an external
protocol certificate → FAR summary. Observational ingest is
`not_testable` / `no_perturbation_protocol`. FAR-EXT-002E is
`BLOCKED_EXTERNAL_CUSTODY` until complete E1+E2 publisher/OA article XML.
No live 0.2, freeze, \(R(\rho)\), or FAR-003 until then. ds006036 photic
remains NOT_TESTABLE (frequency ≠ amplitude).

---

## Claim ceilings

```text
Safe:     MNPS is a versioned chart. MNJ is a chart Jacobian.
          C1 library alignment of an independent chart-space velocity
          with increment covariance leaves a_hat unchanged.
          Ingest does not supply a vector and does not adjudicate
          scientific qualification of b.
          ds004100 can emit chart-space a_hat without biology.
          Opt-in Gate F W_Q is a chart-space discrete predictive-spread
          object from discrete_transition_residual_covariance, not
          process noise.

Not:      latent SDE b(X) or a(x)
          MNPS x_dot or Jacobian intercept as SDE b
          empirical A_bD as NDT drift-diffusion alignment
          R_b/a under C1 as an Itô drift-to-diffusion ratio
          C2 residualized a_hat
          licensed empirical NDT α/ω/G_peak
          Gate E Q as process noise
          2-D/3-D committor or stage-label q
          observational FAR / R(ρ)
          I-CARE tubes = stochastic_reachability.v1
          I_pred, TE, FDT, Fisher, M(τ) as NMD exports
          ingest metadata = M3 PASS
```

---

## Evidence ledger

```text
ESTABLISHED / INTERNAL VALIDATED

DF-DRIFT-C1 M1–M2:
An externally supplied truth-known chart-space velocity can be used
for C1 alignment without modifying the frozen increment-covariance
diffusion measurement.


ARCHITECTURAL DECISION

Drift identification:
owned by nmd-analysis, not ingest.


PLAUSIBLE / UNTESTED

crossfit_local_chart_b:
candidate route to an independently estimated chart velocity.


NOT ESTABLISHED

empirical chart drift
Itô drift
C2 residualized diffusion
empirical NDT drift-diffusion alignment
scientific utility of C1
```

---

## Ingest next (not this handoff)

Control-plane / chart-contract work already in v3 R2–R4: validity,
grain, support/capability, later chart-robustness plumbing,
schema/docs/release. Do not extend drift science inside ingest.

```text
VERDICT:
HANDOFF RATIFIED

Contract impact:
None on serialized ingest a_hat. M3 is an analysis-repo experiment.
Type C ingest implementation is frozen.

Next gate (analysis):
DF-DRIFT-C1-M3 design in nmd-analysis, using the consume-external-b API
without requesting ingest residualization or OD-EPI-001 modification.
```
