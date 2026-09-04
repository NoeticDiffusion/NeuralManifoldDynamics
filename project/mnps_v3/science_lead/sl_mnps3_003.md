Yes. I would **stop spending active research cycles on FAR-EXT-002E for now**. This is not a scientific FAIL; it is a clean **external-custody block**. You have already done the useful engineering work: the binder exists, it fails closed, E3/E4 are complete, and the branch has an unambiguous restart condition. There is little scientific value in repeatedly trying to defeat publisher access controls.

I would therefore park it as something like:

```text
FAR-EXT-002E
STATUS = BLOCKED_EXTERNAL_CUSTODY

Resume only when:
complete E1 + E2 publisher/OA article XML becomes available.

No further active work:
- no live 0.2
- no freeze
- no R(rho)
- no FAR-003
```

That is a successful stopping decision, not an unfinished analysis.

## I think Type C drift is the right next branch

After reading the Type C proposal, I like it considerably more than simply trying to “turn on drift.” The document has identified the key semantic trap correctly: **the existing estimator currently couples supplying `drift=` to residualizing the increments**, so naively supplying a drift vector to obtain \(A_{bD}\) would silently change the meaning of the existing \(a_{\hat{}}\). C1's proposed split between an alignment field and residualization is therefore not cosmetic; it is required to preserve the current diffusion contract. 

I would make the science-lead decision:

```text
TYPE-C DECISION

C1 alignment_only:
    AUTHORIZED for M1–M2 qualification

C2 residualize_increments:
    NOT AUTHORIZED

Empirical C1 overlay:
    NOT AUTHORIZED

Common-profile enablement:
    NOT AUTHORIZED

ds004100 OD-EPI-001 modification:
    NOT AUTHORIZED

MNPS x_dot as drift:
    FORBIDDEN

MNJ affine intercept as drift:
    FORBIDDEN
```

In other words, **we authorize the machinery required to ask whether C1 is well defined, not the empirical claim.**

### The first experiment should be very small

I would make the immediate target something like **DF-DRIFT-C1-M1/M2**.

The implementation should first split:

```python
alignment_drift = ...
residualize_increments = False
```

conceptually, rather than letting one `drift` argument do two jobs.

Then the synthetic qualification asks only three central questions:

```text
1. Does supplying truth-known chart b for alignment leave
   a_hat / D_total / d_diff / c_diff byte-identical
   to the no-drift path?

2. Does A_bD recover the expected anisotropy/alignment ordering
   when truth-known b is supplied?

3. Do forbidden pseudo-drifts fail closed?
   x_dot
   Jacobian intercept
   same-sample local increment mean
```

If #1 fails, C1 fails immediately. That is a very nice falsification gate.

I would actually make **#1 the hard Gate C1-A**. There is no reason to proceed to clever cross-fitting if the implementation cannot guarantee that the old diffusion measurement remains unchanged.

## One refinement to the Type C proposal

There is one place where I would be slightly more conservative than the current sketch.

For C1, I would regard \(A_{bD}\) as the primary object and \(R_{b/a}\) as secondary.

Why? Because under C1 the denominator is still based on **raw increment covariance**, not a residualized Itô diffusion estimate. Directional alignment is relatively easy to state safely:

> an independently estimated chart-space velocity field is aligned to the principal structure of local increment covariance.

But

$$
R_{b/a}
=
\frac{\Delta t\, |b|^2}{\mathrm{tr}(\hat a)}
$$

looks much more readily like a genuine drift-to-diffusion ratio. Under C1, that is not yet what it is.

So I would either give it explicit metadata such as:

```text
a_semantics = raw_increment_covariance
ratio_semantics = chart_velocity_to_increment_spread
```

or keep it secondary until C2.

This isn't a blocker. It is claim hygiene.

## Then comes the genuinely interesting experiment: M3

If synthetic C1 passes, **M3 becomes scientifically interesting**.

The real question becomes:

> Can we estimate a chart-space \(b\) independently enough that \(A_{bD}\) is not just a circular re-expression of the same increments used to estimate \(a\)?

That is exactly where `crossfit_local_chart_b` matters.

I would make M3 explicitly compare:

```text
truth-known b               ← positive control
cross-fit b                 ← candidate estimator
same-sample local mean b    ← contaminated comparator
shuffled/time-displaced b   ← negative control
```

Then measure:

```text
alignment recovery
bias in A_bD
rank recovery
variance
sensitivity to embargo
sensitivity to dt
failure under insufficient support
```

That gives us an actual **drift-identification experiment**, rather than simply adding another export.

And importantly, the same-sample estimator should not just be “forbidden because we said so.” M3 can demonstrate numerically **how much bias it introduces**. That would turn a design rule into internal evidence.

## I would not jump to an empirical dataset yet

The Type C note is right here as well. ds004100 should remain what it already is: an empirical translation overlay for current \(a_{\hat{}}\), not retroactively turned into a drift experiment. 

After M3, we can scout a dataset specifically for C1.

The ideal empirical C1 dataset will probably **not** simply be whichever NMD dataset is convenient. We want data where an independently estimable local velocity field is plausible and where temporal partitioning/cross-fitting can actually provide separation between estimation of \(b\) and estimation of increment covariance.

That deserves its own dataset-selection gate.

There is also no conflict with the pending diffusion M4. I would regard them as two different branches:

```text
DIFFUSION-D-M4
current raw-increment a_hat
→ empirical feasibility question

DF-DRIFT-C1
independent chart-space b
→ synthetic identification/qualification question
```

Neither licenses the other.

## So my programme-level next move would be

```text
FAR
│
└── PARKED — BLOCKED_EXTERNAL_CUSTODY
    restart only if E1/E2 arrive


DIFFUSION
│
├── existing a_hat branch
│   └── D-M4 remains separate/pending
│
└── NEW: DF-DRIFT-C1
    │
    ├── M1 estimator semantic split
    │     alignment_drift
    │     residualization separately gated
    │
    ├── M2 truth-known C1 qualification
    │     a_hat identity
    │     A_bD recovery
    │     forbidden-source refusal
    │
    └── STOP / science-lead
          ↓ only if PASS
        M3 cross-fit identification experiment

C2 residualized diffusion
└── CLOSED / NOT AUTHORIZED
```

I think this is a **better next scientific step than trying to force FAR forward**. It attacks a real unresolved issue in NDT/NMD: we have made meaningful progress on local stochastic geometry and Jacobian structure, but the empirical **drift side of the local law remains much less well identified**.

And C1 lets us approach that without making the dangerous jump from “MNPS trajectory has a derivative” to “we have measured the SDE drift.”

### What I would send to the repo agent

> **Science-lead decision: park FAR-EXT-002E as `BLOCKED_EXTERNAL_CUSTODY`; no further source-acquisition work is requested unless complete E1/E2 XML becomes available. Open Type-C drift as a separate branch. C1 `alignment_only` is authorized for M1–M2 synthetic qualification only. C2 `residualize_increments`, empirical C1 overlays, common-profile enablement, and modification of ds004100 OD-EPI-001 are not authorized. First implement the estimator semantic split so alignment drift cannot trigger residualization. Then test the hard C1 invariant that truth-known alignment drift leaves `a_hat`, `D_total`, `d_diff`, and `c_diff` byte-equivalent to the no-drift path, together with truth-known \(A_{bD}\) recovery and fail-closed refusal of `mnps_xdot`, Jacobian intercept, and same-sample increment-mean sources. Return the M1–M2 packet to science lead before cross-fit M3. Treat \(R_{b/a}\) as secondary under C1 and explicitly record that its denominator is raw increment covariance rather than residualized process diffusion.**

That is the branch I would open now.
