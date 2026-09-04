Yes — I think we have reached the point where **we are about to do too much in the ingest repo**.

M1–M2 was still defensible there because it was about **contract safety**: separating alignment from residualization, keeping `a_hat` byte-stable, refusing invalid pseudo-drifts, and making sure ingest cannot accidentally change semantics. But **M3 is qualitatively different**. Cross-fitting a drift field, comparing estimators, quantifying contamination bias, selecting embargoes, and asking whether a data-derived \(b\) is identifiable are analysis questions, not ingest questions.

I would draw the boundary like this:

```text
INGEST / NeuralManifoldDynamics
────────────────────────────────
Owns:
- MNPS 3D / 9D construction
- versioned schemas and HDF5
- deterministic measurement primitives
- family estimator APIs
- provenance / support / grain / validity metadata
- fail-closed gates
- "I can consume an externally supplied qualified b"
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
- condition contrasts or inferential statistics
- deciding whether C1 is scientifically useful
```

Then:

```text
ANALYSIS REPO
────────────────────────────────
Owns:
- candidate chart-drift estimators
- crossfit_local_chart_b
- train/evaluation partitioning
- embargo rules
- contaminated comparator
- shuffled/time-displaced controls
- truth-known / synthetic benchmark harness
- dataset-specific feasibility
- M3 / M4 / M5
- statistical inference
- eventual empirical overlay creation
```

So I would actually **revise my immediately previous recommendation**: I would **not authorize M3 inside NMD ingest**.

The right science-lead action now is:

```text
DF-DRIFT-C1 ingest line

M1–M2:
PASS / RATIFIED

Ingest implementation:
FROZEN at supplied-field C1 capability

M3 in ingest:
NOT AUTHORIZED

Reason:
crossfit drift identification is an analysis-layer estimator-selection
and validation problem, not an ingest measurement-contract responsibility.

Next:
handoff specification to analysis repo.
```

### What should remain from the current implementation?

Almost everything that landed in M1–M2 is useful.

Ingest should retain:

```python
estimate_local_diffusion_geometry(
    ...,
    drift=external_field_or_none,
    residualize_increments=False,
)
```

and the semantic safeguards around it.

Conceptually, NMD can say:

> “If you hand me a qualified chart-space velocity field with compatible time semantics, I can calculate C1 alignment without changing the raw-increment diffusion object.”

That is a perfectly reasonable ingest capability.

But NMD should **not answer the next question**:

> “How should I construct that velocity field from this dataset?”

That belongs downstream.

This also makes `chart_drift.py` clearer. I would probably keep it very small in ingest. It should validate an externally supplied object and refuse forbidden provenance, rather than grow into a drift-estimation module.

Something like:

```text
chart_drift.py

VALIDATE:
truth_known_chart_b
externally_qualified_chart_b   # later, if needed

REFUSE:
mnps_xdot
jacobian_intercept
local_increment_mean
unknown provenance
```

I would be cautious about making `crossfit_local_chart_b` itself an ingest-side source implementation. The token could eventually describe provenance, but **the algorithm producing it should live outside ingest**.

## This also clarifies the larger NMD v3 philosophy

I think this is actually an important architectural realization.

We have been discussing v3 as:

```text
chart + drift + dynamical families + control plane
```

But there are two meanings of “family” that must not be allowed to blur:

1. **Measurement primitive** — an estimator that can be deterministically applied when its required inputs are supplied.
2. **Scientific analysis programme** — how those inputs are identified, qualified, compared, and interpreted.

NMD ingest should own **1**, not **2**.

For diffusion:

```text
INGEST:
trajectory + dt
    → raw increment covariance
    → diffusion geometry

ANALYSIS:
Is this a useful empirical stochastic-process approximation?
Does it differ between condition A/B?
```

For C1:

```text
INGEST:
trajectory + qualified external b
    → A_bD

ANALYSIS:
How do we estimate b independently?
Does cross-fitting work?
```

For committor:

```text
INGEST/library primitive:
trajectory + explicit RC + A/B + certificate
    → q estimate

ANALYSIS:
Which RC?
Which A/B?
Does this scientific dataset justify them?
```

For FAR:

```text
INGEST/library primitive:
qualified perturbation/outcome table
    → FAR summary

ANALYSIS:
Is the perturbation interpretable?
What constitutes recovery?
Does R(rho) exist empirically?
```

That division is much healthier.

### There is even an argument that some existing work already sits slightly too deep in ingest

I would **not refactor it now**, but things like the growing family logic can easily become a scientific-analysis framework by accretion.

A useful rule for future v3 changes could be:

> **If an operation requires choosing among scientifically plausible estimators, splitting data to establish independence, tuning a validation protocol, constructing null controls, or interpreting a condition contrast, it belongs outside ingest.**

Conversely:

> **If it takes an explicitly qualified set of inputs and deterministically produces a versioned measurement object with fail-closed semantics, it can belong in ingest.**

That is a strong boundary.

## What I would do next

I would close Diary 288 as the endpoint of the ingest branch and create a handoff note, perhaps:

```text
project/mnps_v3/df_drift_c1_analysis_handoff.md
```

with:

```text
INGEST HANDOFF CONTRACT

Available:
- raw MNPS trajectory
- timestamps / dt
- raw-increment a_hat
- C1 alignment API
- provenance fields
- forbidden-source rules

Analysis must supply:
- b_hat array
- exact alignment to trajectory/timebase
- estimator provenance
- qualification id/hash
- train/evaluation partition provenance
- no same-increment contamination certificate

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
```

Then **M3 becomes an analysis-repo experiment**, while the ingest repo can return to the v3 contract work.

And this is where your intuition is important: I would now probably **continue the NMD ingest programme with the control-plane / chart-contract tasks**, rather than keep extending drift science inside it.

So the program forks cleanly:

```text
NMD INGEST v3
    → finish measurement contract
    → validity/support/grain
    → chart robustness plumbing
    → schema/docs/release

NMD ANALYSIS
    → DF-DRIFT-C1-M3
    → crossfit identification
    → later empirical feasibility

FAR
    → parked pending external custody
```

I think this is the better architecture. **M1–M2 found exactly the right ingest boundary: NMD can safely consume an independent drift field. M3 is where we should cross into analysis.**
