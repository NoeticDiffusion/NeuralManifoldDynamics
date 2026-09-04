Yes — I agree with you. If we already know that **`orthogonal_axis` / `orthogonal_dynamics` is conceptually wrong for the long term**, I would not build the v3 control plane on top of those names and then rename it afterward.

So among the three options in the screenshot, I would choose **C**, with one refinement:

> **Round 1 = canonical rename + minimal family registry/DAG.**
> Then validity, inferential grain, support/capability, and richer certificates get their own later gated rounds.

The reason is architectural. The registry will become the vocabulary authority for everything that follows. If we first create:

```text
orthogonal_dynamics registry
orthogonal_axis family IDs
orthogonal config
```

and subsequently rename everything, we will have to migrate the very control plane we just created. Worse, legacy terminology can leak into schema IDs, tests, reports, APIs, and documentation.

The conceptual hierarchy is now clear enough to freeze:

```text
NeuralManifoldDynamics
│
├── chart/
│   ├── mnps_3d
│   └── coords_9d
│
├── drift/
│   ├── jacobian
│   ├── spectral_abscissa
│   ├── numerical_abscissa
│   ├── reactivity_gap
│   └── finite_time_response
│
└── dynamical_families/
    ├── diffusion
    ├── spread
    ├── destination
    └── resilience
```

That distinction is consistent with NMD's existing role as an operational, versioned measurement contract rather than a claim that all exported quantities belong to the MNPS coordinate ontology. 

### I would therefore make Round 1 slightly larger than option C implies

Not a broad estimator change, but a **semantic foundation refactor**:

```text
ROUND 1 — V3 SEMANTIC KERNEL

1. Canonical vocabulary
   orthogonal_axis       → dynamical_family
   orthogonal_dynamics   → dynamical_families
   "orthogonal stack"    → family program

2. Code namespace
   mndm.orthogonal_dynamics
       → mndm.dynamical_families

3. Config namespace
   orthogonal_dynamics:
       → dynamical_families:

4. Analysis naming
   od_* generic names
       → family_* where OD does not denote a historical protocol ID

5. Documentation
   CONFIG_GUIDE
   Output_variables_guide
   schema_docs
   reporting docs
   changelog
   terminology/alias map

6. Minimal registry/DAG
   canonical family IDs
   schema IDs
   dependencies
   forbidden substitutions
   legacy aliases
   implementation status
```

I would **not rename historical experiment IDs**, though. Things such as:

```text
OD-TQ1
OD-TQ2b
OD-TQ3
OD-EPI-001
OD-SLP-004
```

are provenance. They happened under that programme name. Renaming them retrospectively would corrupt the historical record.

So:

```text
historical OD-* experiment IDs       KEEP
current program terminology          RENAME
current Python/config terminology    RENAME
future protocols                     FAMILY-* or explicit family name
```

## I would also now lean toward a real v3 HDF5 namespace change

This is the one place where I would revise my previous conservative recommendation.

If we were making a v2.6.1 maintenance release, I would leave:

```text
/orthogonal_dynamics/
```

alone.

But if this is genuinely **NeuralManifoldDynamics v3**, and we already know the name is both misspecified conceptually and partly misspelled elsewhere, then v3 is exactly the point at which a controlled breaking namespace change is defensible.

I would use:

```text
/dynamical_families/
    /diffusion/
    /destination/
    /resilience/
```

rather than carrying `/orthogonal_dynamics/` indefinitely.

But it needs a proper compatibility rule:

```text
NMD <= 2.x
    /orthogonal_dynamics/.../v1

NMD >= 3.0
    /dynamical_families/.../v1
```

with reader-side migration:

```text
if /dynamical_families/ exists:
    use canonical v3 path
elif /orthogonal_dynamics/ exists:
    read as legacy family namespace
    provenance.namespace_origin = "legacy_orthogonal_dynamics"
```

I would **not dual-write both paths into every v3 HDF5**, because then we have two apparently authoritative copies of the same object. Read legacy; write canonical.

Schema IDs themselves do not necessarily have to change:

```text
mndm.diffusion_geometry.v1
mndm.committor.v1
mndm.finite_amplitude_resilience.v1
```

Those are already semantically fine.

So we are breaking the **container namespace**, not gratuitously bumping every mathematical schema.

---

## The minimum registry should stay genuinely minimal

I would resist putting validity/grain/support into the first implementation round. Otherwise C secretly becomes A+B+half of v3.

Something like this is enough:

```yaml
families:

  diffusion:
    canonical_id: diffusion
    schema: mndm.diffusion_geometry.v1
    namespace: /dynamical_families/diffusion/v1
    legacy_namespace: /orthogonal_dynamics/diffusion_geometry/v1
    implementation: landed
    default_enabled: false

    requires:
      - chart_trajectory
      - transition_increments

    forbids:
      - jacobian_derivative_residual_as_diffusion

  spread:
    canonical_id: spread
    schema: mndm.stochastic_reachability.v1
    implementation: gate_closed
    gate: F

    requires:
      - local_transition_operator
      - admissible_process_covariance

    forbids:
      - derivative_residual_covariance
      - gate_e_proxy_as_process_noise

  destination:
    canonical_id: destination
    schema: mndm.committor.v1
    implementation: restricted

    requires:
      - explicit_reaction_coordinate
      - explicit_set_A
      - explicit_set_B
      - first_hit_certificate

  resilience:
    canonical_id: resilience
    schema: mndm.finite_amplitude_resilience.v1
    implementation: serializer_only

    requires:
      - perturbation_amplitude
      - outcome_definition
      - recovery_or_survival_criterion
```

That immediately gives later work a stable backbone.

Then the next rounds become beautifully small:

```text
R1  V3 semantic kernel
    rename + migration + registry/DAG

R2  validity contract
    computation_status
    measurement_validity
    claim_status

R3  inferential grain
    window / event / run / subject semantics

R4  support/capability
    actual feature support, fallback use,
    modality capability

R5  family-specific science gates
    diffusion M4 / Gate F / FAR / etc.
```

This sequencing also obeys the principle in your plan that **v3 should not become an excuse to implement every theoretically interesting object**.

### One further naming detail

I like your human-facing names:

```text
diffusion
spread
destination
resilience
```

but I would keep the mathematical family names immediately underneath them:

```text
spread       → stochastic reachability / predictive spread
destination  → committor
resilience   → finite-amplitude resilience
```

That gives us both conceptual readability and mathematical precision.

For example:

```text
family_id          = "destination"
family_object      = "committor"
schema_id          = "mndm.committor.v1"
```

That will be especially useful later if a family acquires several estimators.

---

So my answer to the Cursor question would effectively be:

> **Choose C.** Since `orthogonal_axis` / `orthogonal_dynamics` is now known to be the wrong long-term ontology, v3 should establish the canonical naming before the wider control plane is built. Round 1 should combine the rename with a minimal family registry/dependency DAG as the v3 semantic kernel. Historical `OD-*` protocol IDs remain unchanged as provenance. Because this is a major v3 release, prefer canonical HDF5 `/dynamical_families/` writes with explicit legacy `/orthogonal_dynamics/` reader compatibility and a migration note; do not dual-write both namespaces. Validity, inferential grain, and support/capability should then be separate subsequent rounds.

I think that gives us a much cleaner v3 foundation than either A or B.
