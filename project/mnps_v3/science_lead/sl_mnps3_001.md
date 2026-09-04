Yes. I think this is a very strong direction, and I would make **one important conceptual move beyond the current plan**:

> **NMD v3 should not mainly be “NMD with more metrics.” It should become a measurement operating system with an explicit control plane for validity, evidence, dependency, and inferential grain.**

That fits the original purpose of NeuralManifoldDynamics unusually well. The current methods paper already defines NMD primarily as a fixed, auditable measurement contract rather than a latent-state discovery algorithm, and explicitly notes that the chart has not yet been validated for invariance under feature substitutions, weighting perturbations, or alternative projections. 

Your draft is already moving in exactly that direction.

## What I would keep from the plan

I would ratify almost all of the high-level architecture:

```text
chart          where is the system?
drift          what is the local deterministic tendency?
diffusion      how are local stochastic increments structured?
spread         what nearby futures are available?
destination    where is the process likely to end?
resilience     what survives finite perturbation?
```

And I strongly agree with **not** turning any of these into new MNPS axes. The current NMD contract explicitly treats 3D and 9D as release-fixed operational coordinates, with Jacobian-derived quantities layered above them rather than incorporated into the ontology of `[m,d,e]`. 

I also agree with **docs+gate v3 rather than a breaking namespace**. Renaming `/orthogonal_dynamics/` buys very little scientifically and creates migration risk. Make “dynamical families” the public language and leave the old path as a versioned historical artifact.

But I think v3 could become substantially stronger with five additions that are mostly **contract infrastructure rather than new estimators**.

| Proposed v3 addition                 | What it answers                                         |             Priority |
| ------------------------------------ | ------------------------------------------------------- | -------------------: |
| **Family registry + dependency DAG** | What does this quantity require?                        |            Very high |
| **Validity certificate**             | Is this estimate trustworthy here?                      |            Very high |
| **Inferential-grain contract**       | What is one independent observation?                    |            Very high |
| **Support / capability signature**   | What actually contributed to this value?                |                 High |
| **Chart robustness certificate**     | Is the frozen chart stable to reasonable perturbations? |                 High |
| **Trajectory/repertoire geometry**   | What temporal grammar was actually realized?            | Later / experimental |

### 1. Add a machine-readable **family registry and dependency DAG**

This may be the single most useful addition.

At present the scientific dependencies are clear to us, but they are distributed over protocols and documentation. v3 could make them executable metadata.

For example:

```text
diffusion_geometry
    requires:
        chart trajectory
        regular-enough Δt
        increment support
    forbids:
        derivative-residual-as-a

stochastic_spread
    requires:
        A_t
        admissible Q_t
        horizon h
    forbids:
        derivative residual Q
        transition-residual proxy unless Gate F authorizes it

committor
    requires:
        explicit RC
        explicit A
        explicit B
        first-hit certificate
    forbids:
        stage labels as destination truth

far
    requires:
        controlled perturbation amplitude
        outcome definition
        return/survival criterion
    forbids:
        inference from spontaneous trajectories
```

That turns your scientific discipline into software behavior.

I would make the dependency system capable of returning not just:

```text
computed = false
```

but:

```text
status = not_testable
reason_code = MISSING_ADMISSIBLE_Q
blocked_by = gate_f
forbidden_substitution = transition_residual_covariance_proxy
```

This is much stronger than NaN.

---

## 2. Introduce a **Measurement Validity Certificate**

This is the addition I would push hardest.

Your truth-known work has now shown something very important: diffusion and Jacobian quantities can genuinely recover truth under specified conditions, but their validity deteriorates with support, timestep, bandwidth and observation noise. It also explicitly keeps the distinction between simulator-side estimator validation and parity with the literal NMD ingest implementation. 

That should become part of v3's architecture.

Every local-dynamics/family export could carry something like:

```yaml
validity:
  estimator: local_increment_covariance.v1
  support_n: 742
  dt:
    median: 0.500
    irregularity: 0.012
    status: qualified
  neighborhood:
    n_effective: 61
    condition: qualified
  observation_noise:
    correction: none
    status: unknown
  truth_known_envelope:
    schema: od_o1.v1
    inside_declared_envelope: unknown
  literal_ingest_validation:
    status: not_established
  overall:
    status: measurement_valid_bounded
```

The important distinction is:

```text
value exists
≠ estimator succeeded
≠ estimator is valid under this observation regime
≠ NDT interpretation is licensed
```

You currently express this mostly through `claim_status`. I would separate it into at least three orthogonal fields:

```text
computation_status
measurement_validity
claim_status
```

For example:

```text
computation_status = computed
measurement_validity = translation_qualified
claim_status = no_biological_claim
```

That would be an unusually strong scientific software feature.

---

## 3. Make **inferential grain** part of the HDF5 contract

This is less glamorous but extremely important.

You already discovered this problem in local-dynamics integration: an H5/run is not automatically a participant, and windows/transitions/horizons are not independent biological replicates.

I would make grain first-class:

```text
recording
window
transition
recording_horizon
event
subject
```

Every exported schema should say something like:

```yaml
grain:
  native: window
  parent: recording
  biological_unit: subject
  repeated_measure: true
  direct_between_subject_inference: forbidden
```

Then analysis tooling can refuse something dangerous like treating 40,000 windows from 20 people as `N=40,000`.

This is exactly the kind of feature that strengthens NMD's identity as a **measurement contract** rather than merely a metric package.

---

## 4. Export a **support/capability signature**

The current NMD article already documents an important comparability problem: some coordinates can depend on modality-specific or fallback feature families, and higher-order capabilities differ between EEG and fMRI. 

At present, provenance tells us this happened. v3 could make it much easier to consume safely.

For every coordinate/family, export a compact support signature:

```text
m_a:
    direct_support = 1.00
    fallback = false

e_m:
    direct_support = 0.00
    fallback = eeg_highfreq_power_30_45
    semantic_equivalence = false

jacobian_9d:
    support_fraction = 0.87
    conditioning = qualified
```

At file level you could then expose a modality/capability matrix:

```text
                   EEG       iEEG      fMRI
chart_3d           yes       yes       yes
chart_9d           yes       yes       yes
MNJ_3d             yes       yes       limited
MNJ_9d             conditional yes     limited
diffusion          conditional ...
spread             gated
committor          no generic ingest
FAR                perturbational only
```

This would prevent two HDF5 files from looking semantically equivalent just because both contain `coords_9d`.

---

## 5. Add a **chart robustness certificate**, without changing the chart

This is the main gap I see relative to the existing NMD paper.

The current manuscript explicitly states that the chart has **not yet been validated as invariant under feature substitutions, weighting perturbations, or alternative projection families**. 

You do not need to solve chart uniqueness to improve this.

Instead, define a bounded chart audit:

```text
canonical P_fixed
      │
      ├── ±weight perturbation
      ├── feature leave-one-family-out
      ├── normalization sensitivity
      ├── fallback removal
      └── window sensitivity
             ↓
      chart robustness report
```

Possible outputs:

```text
trajectory rank correlation
pairwise-distance preservation
condition-contrast sign stability
9D→3D recomposition stability
local-neighbor retention
MNJ sign/order stability
```

Crucially, **none of these alternative charts becomes an export**.

They only answer:

> How fragile is the frozen operational chart under reasonable perturbations?

That would directly address one of the most important limitations of NMD 2.x without reopening `[m,d,e]`.

I would probably call this:

```text
mndm.chart_robustness_certificate.v1
```

and keep it analysis/audit-only initially.

---

# One genuinely new scientific layer I would consider later

There is one object missing from your current layer diagram.

You have:

```text
chart       where?
families    what can happen?
```

But there is a third question:

> **What sequence of transitions actually happened?**

This is neither occupancy nor reachability nor Jacobian.

Your compact NDT summary already contains the beginnings of this as **trajectory/repertoire geometry**, explicitly distinguishing it from MNJ and reachability:

```text
MNJ           local deformation of the flow
reachability  admissible nearby futures
trajectory    realized temporal grammar
```

and proposes objects such as repertoire entropy rate, irreversible flux, entropy production, return probability and transition sparsity. 

I think that idea is scientifically interesting.

For example, two systems could have approximately equal:

```text
MNPS position
local J
local W_Q
```

yet differ radically in realized traversal:

```text
A → B → C → A → B → C ...
```

versus

```text
A → A → A → B → A → A ...
```

The local dynamical potential may look similar while the **realized temporal grammar** differs.

I would **not put this in v3.0**, though.

It deserves its own synthetic falsification programme first because discretizing continuous MNPS into states introduces all the familiar partition/binning problems.

I would reserve something like:

```text
trajectory/
    repertoire_flow
    entropy_rate
    return_structure
    irreversible_flux
```

for **v3.x experimental**.

That gives us four conceptually clean questions:

```text
CHART
Where is the system?

LOCAL DYNAMICS / FAMILIES
What can happen from here?

TRAJECTORY
What actually happened over time?

CONTROL
What changes the dynamics?
```

I think that is cleaner than trying to squeeze predictive information, transfer entropy, etc. into the first v3 release.

---

# On your existing D–H families

I would retain the asymmetry rather than trying to make the release look complete.

**Diffusion** is currently the strongest candidate to become the first exemplary v3 family. The truth-known O1 chain has a positive bounded validation result, while the empirical ds004100 result is still only translation-qualified rather than biological validation. That is exactly the kind of mixed evidence state v3 should be designed to represent. 

**Committor** should remain extremely narrow. I agree with your `explicit RC + explicit A/B + first-hit` rule. I would rather have a tiny committor contract that is difficult to activate than a generic `q` output that later becomes uninterpretable.

**FAR** should remain perturbational. I strongly agree that spontaneous recovery, Jacobian contraction, FTR and reachability must not silently become resilience.

**Spread/W_Q** is scientifically attractive, but I would not rush Gate F merely because earlier analysis papers found useful “reachability” effects. For example, the depression analysis showed that its reachability layer carried stronger BDI-linked separation than coarse 3D summaries, but that is evidence that higher-order geometry can be informative—not evidence that the new `stochastic_reachability.v1` contract or a particular Q construction is therefore licensed. 

That historical separation should be explicit in v3.

---

# I would therefore define NMD v3.0 like this

Not:

> “NMD now computes lots of NDT dynamics.”

But:

> **NeuralManifoldDynamics v3 is a versioned chart-and-dynamical-family measurement contract with machine-readable validity, dependency, support, grain and claim semantics.**

And I would make the release scope:

```text
V3.0
────────────────────────────────────────────
Frozen MNPS 3D / 9D                       ✓
Existing local_dynamics                    ✓
Dynamical-family registry                  NEW
Dependency / forbidden-substitution DAG    NEW
Validity certificates                      NEW
Inferential-grain metadata                 NEW
Support/capability signatures              NEW
Claim/evidence status normalization        NEW
Legacy orthogonal_dynamics alias           NEW
Diffusion documentation                    promoted
Committor                                  gated
FAR                                        gated
W_Q / spread                               Gate F closed
New estimators                             none required
```

Then:

```text
V3.1 candidate
    diffusion empirical M4
    chart robustness certificate
    possibly Gate F

V3.2+
    trajectory/repertoire geometry
    only after synthetic qualification

M0
    predictive information
    TE
    FDT
    Fisher/observability
    scale consistency
```

That would make v3 a **much more important release than simply adding five families**. It would encode one of the strongest methodological lessons that has emerged from the whole NDT/NMD programme: a number without its provenance, support, estimator envelope, inferential grain, and claim ceiling is not yet a scientific measurement.

So my main verdict on your draft is **`APPROVE_DIRECTION_WITH_EXTENSION`**. I would ratify A–B, choose the non-breaking docs+gate path, and before touching Gate F add one new v3 work package—call it perhaps **A2: Family validity and dependency contract**—covering the dependency DAG, validity certificate, grain semantics, and capability/support signature. That is the piece I think would turn this from a naming/documentation refresh into a genuinely stronger next generation of NeuralManifoldDynamics.
