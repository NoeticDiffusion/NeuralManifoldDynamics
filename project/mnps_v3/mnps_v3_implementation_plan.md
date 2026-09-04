# MNPS v3 — implementation plan

Status: active programme (SL-005 fork, 2026-09-02). Not a chart change and
not a biological program. C1 ingest line frozen; M3 lives in nmd-analysis.

This document turns the local-dynamics + dynamical-family audit into a single to-do list. It does **not** redefine `[m, d, e]` or the 9D order. MNPS remains the observation chart. v3 is the contract generation that treats **dynamical families** as first-class, versioned, fail-closed exports beside that chart.

Invariant:

```text
MNPS answers: where is the system?
Families answer: what can happen from here?
Control (mostly analysis/theory) answers: what changes that?
```

Do not merge those questions into new MNPS axes.

---

## 1. Naming

`orthogonal axis` / `orthagonal_axis` is a bad program name.

- The NDT v2 article is explicit: the objects are **non-equivalent questions**, not linearly orthogonal axes.
- `axis` collides with MNPS coordinates (`m`, `d`, `e`).
- The folder is misspelled (`orthagonal`).
- The code already calls them **families** (`mndm.dynamical_families`: “standard non-MNPS dynamical measurement families”).

### Recommended vocabulary

| Do not use | Use | Why |
|---|---|---|
| orthogonal axis / orthagonal | **dynamical families** | Matches code and NDT table of non-equivalent questions |
| OD / “the orthogonal stack” | **family program** or named family (`diffusion`, `committor`, `far`) | OD is overloaded (orthogonal, original data, OD-TQ) |
| new MNPS axis | **family export** | Families are not coordinates |
| reachability cone | **predictive-spread ellipsoid** | Article and `stochastic_reachability.v1` are covariance tubes, not cones |
| Φ (unqualified) | **propagator `phi` / `g_max`**, NDT potential **`V`**, IIT **`Phi`** | Three different objects |
| FDT (unqualified) | **fluctuation–dissipation** vs EEGLAB **`.fdt`** | Repo collision |
| transfer (unqualified) | **family-energy transfer on `phi`** vs **transfer entropy** | FTR `transfer_*` is not TE |
| predictive information (as HDF5) | **StateQ** until a family schema exists | Charter object ≠ current program |

### Proposed layer names (human + config)

Keep existing HDF5 **schema IDs** for v1 (`mndm.jacobian_metrics.v1`, `mndm.diffusion_geometry.v1`, …). Round 1 (SL-002) **did** rename the HDF5 **container** to `/dynamical_families/<family_id>/v1`. Schema IDs, MNPS order, and the Jacobian estimator stay frozen. Write canonical only; read legacy `/orthogonal_dynamics/` when the canonical group is missing. Never dual-write.

```text
chart/            MNPS 3D + 9D                         (unchanged)
drift/            J, α, ω, Δ_react, FTR                (local_dynamics)
diffusion/        a(x), D_total, d_diff, c_diff        (family)
spread/           Q, W_Q, predictive-spread ellipsoid  (local_dynamics opt-in; family YAML still gate_closed)
destination/      committor q                          (family)
resilience/       FAR / basin-return / survivability   (family)
deferred/         I_pred, TE, FDT, Fisher, M(τ)        (M0 until triad M4 closes)
```

Config sketch (do not implement in this file):

```yaml
dynamical_families:
  enabled: false
  diffusion: { enabled: false }
  destination: { enabled: false }   # committor
  resilience: { enabled: false }    # FAR
```

`local_dynamics` can stay as the Jacobian-derived package name. It is already accurate.

**SL-002 HDF5 decision (Round 1, 2026-08-31):** rename the container to `/dynamical_families/{diffusion,destination,resilience}/v1`. Keep schema IDs. Read legacy `/orthogonal_dynamics/` only. Do not dual-write.

```text
/dynamical_families/  =  dynamical families v3 write path
/orthogonal_dynamics/ =  pre-v3 read-only alias
```

---

## 2. What v3 is and is not

**v3 is**

- a measurement-contract generation on top of frozen MNPS 3D/9D
- a place to finish, document, and sequence the families already in code
- a claim-ceiling map so HDF5 math is not read as licensed NDT

**v3 is not**

- a new `[m, d, e]` ontology
- a consciousness scalar
- a reason to enable families on common EEG/fMRI by default
- a reason to implement all eight charter families at once

NDT v2 owns \(X_t\), \(b\), \(a\), \(K^h\), \(q\), FAR. NMD owns charts, windows, gates, and particular estimators. v3 must keep that split.

---

## 3. Current inventory (do not re-implement)

### 3.1 Chart — landed

- `/mnps_3d`, `/coords_9d` with canonical 9-name order
- subject- vs cohort-anchored layers; do not merge
- MNJ = `/jacobian/J_hat` (chart Jacobian, not \(J(X_t)\))

### 3.2 Drift-side / local dynamics — landed in v2.6.0

| Contract | HDF5 | Default | Claim ceiling |
|---|---|---|---|
| `mndm.jacobian_metrics.v1` | `/jacobian/derived_metrics/v1/` (+ 9D) | on whenever `J_hat` exists | math measurement; S3 does **not** license empirical NDT α/ω |
| `mndm.finite_time_response.v1` | `/finite_time_response/v1/` | opt-in | `model_derived`; not FAR; not perturbational |
| `mndm.transition_residuals.v1` | `/transition_residuals/v1/` | opt-in (Gate E) | residual proxy, not process noise |
| `mndm.transition_residual_covariance_proxy.v1` | sibling Q proxy | opt-in | not `W_Q` |
| `mndm.residual_covariance_proxy.v1` | library | not summarize | derivative residual ≠ admissible Q |
| `mndm.stochastic_reachability.v1` | `/stochastic_reachability/v1/` | opt-in after Gate F freeze | chart-space discrete \(W_Q\); not process noise; not I-CARE tubes |

Name map:

| Theory / summary.md | Serialized name |
|---|---|
| α(J) | `spectral_abscissa` |
| ω(J) | `numerical_abscissa` |
| Δ_react | `reactivity_gap` |
| G_peak | FTR `g_peak_over_horizons` (not jacobian_metrics) |
| Ω | only scalar `rotation_norm`, not the matrix |

Reference ingest: ds003947, ds004100, ds003478, ds006036, ds004504 emitted metrics+FTR as descriptive output. ds003944 produced **zero** HDF5 (9D coverage collapse); do not loosen coverage to rescue it.

### 3.3 Dynamical families — landed as gated v1, not in v2.6.0 notes

HDF5: `/dynamical_families/{diffusion,destination,resilience}/v1`  
Legacy read: `/orthogonal_dynamics/{diffusion_geometry,committor,finite_amplitude_resilience}/v1`  
Default: **off**. Common EEG/fMRI/ephys must not import the family yaml.

| Family | Schema | Pipeline | Empirical `computed` |
|---|---|---|---|
| Diffusion | `mndm.diffusion_geometry.v1` | gated; `drift=None` so `A_bD` / `R_b_over_a` are explicit `not_testable` / `independent_drift_not_supplied` (NaN is not zero) | ds004100 OD-EPI-001: 319/319 3D `a_hat`; translation-qualified method tag, not biology |
| Committor | `mndm.committor.v1` | O2b 1-D adapter only; fail-closed without explicit A/B + RC + certificate | **no production overlay**. ds004100 NOT_TESTABLE. BOAS METHOD-LIMITED (SLP-003/004). ds004511 closed |
| FAR | `mndm.finite_amplitude_resilience.v1` | outcome-table summarizer only | observational ingest `not_testable`. Empirical R(ρ) **not started**. Furthest: ds003670 002D signal/timebase PASS; 002E custody BLOCKED |

Every computed family payload: `claim_status = no_biological_claim`.

### 3.4 Charter families with no code (M0)

Do not start these until the triad has an empirical M4 decision.

| Family | Object | Notes |
|---|---|---|
| Predictive information | \(I_{\mathrm{pred}}\), \(\tau_{\mathrm{pred}}\) | Not StateQ occupancy. StateQ is a separate pre-occupancy program (C3 METHOD-LIMITED) |
| FDT / susceptibility | \(D_{\mathrm{FDT}}\) | NDT v2 **does not assume** FDT. Code “FDT” is EEGLAB `.fdt` |
| Transfer entropy | \(TE_{e\to m}\) | Not FTR `transfer_*` energy fractions |
| Observability / Fisher | \(\mathcal{I}(x)\) | Circularity risk: MNPS is already from \(Y\) |
| Scale consistency | \(M(\tau)\) | Not ACF `tau_summary`, not FTR `horizon_steps` |

### 3.5 Documentation debt (from audit)

- Family schemas missed `CHANGELOG.md` v2.6.0; Round 1 recorded the container rename under Unreleased
- `Output_variables_guide.md` now lists `/dynamical_families/` plus legacy-read note
- NDT v2 S3 refuses licensed empirical α/ω/FTR; HDF5 still serializes them
- Coma “reachability” in the article is older analysis-repo estimators, **not** `stochastic_reachability.v1`

---

## 4. To-do list

Order is the recommended sequence. Do not skip a freeze box into a later family.

Checkboxes are work items, not claims of completion.

### A. Freeze v3 scope and names

- [x] Ratify: MNPS 3D/9D order and `P_fixed` **do not change** in v3
- [x] Ratify program name: **dynamical families** (not orthogonal axis)
- [x] Ratify layer names: `chart / drift / diffusion / spread / destination / resilience / deferred`
- [x] Write a one-page alias map: registry in `mndm/src/mndm/dynamical_families/registry.py`; human id vs math object vs schema vs write/legacy paths
- [x] Record claim ceiling: serialized math ≠ licensed NDT (S3 α/ω, FTR, coma reachability)
- [x] SL-002: v3 Round 1 is a **new export namespace** (breaking container rename). Schema IDs unchanged. Write canonical `/dynamical_families/`; read-only legacy `/orthogonal_dynamics/`

### B. Documentation catch-up (Type A, do first)

- [x] Add family schemas / container rename to changelog (Unreleased Round 1; they missed v2.6.0)
- [x] Document `/dynamical_families/*/v1` in `mndm/Output_variables_guide.md` (legacy-read note included)
- [x] Align `schema_docs.md` / `reporting/dynamical_families.md` with canonical write + legacy read
- [x] Fix experimental vs `contract_status=standard` wording
- [x] Note `G_peak` → `g_peak_over_horizons`; `A_bD` not computed on ingest (`drift=None`)
- [x] Note O2b does **not** emit \(V_{1/2}\); first-hit estimator does **not** emit it either (it serializes `q_A_to_B`); \(\lvert\nabla q\rvert\) absent
- [x] Optional NDT v2 sentence: v2.6 `jacobian_metrics` / FTR are provenance, not S3-licensed empirical NDT; I-CARE reachability ≠ `stochastic_reachability.v1`

### C. Drift-side remaining gates (local_dynamics)

- [x] **Gate F design freeze:** what one-step Q is admissible for `W_Q` (transition residual vs explicit conversion model)
  - Frozen 2026-09-02: `project/mnps_v3/gate_f_admissible_q.md`. Admissible Q = Gate E recording-level transition-residual covariance (`conversion_model=not_applicable`). Derivative residual still refused. Gate E Q is not process noise and not \(a(x)\).
- [x] Implement summarize emission of `stochastic_reachability.v1` **only after** Gate F freeze
  - Opt-in `local_dynamics.stochastic_reachability.enabled` (default false). Writes `/stochastic_reachability/v1`. Family YAML `spread` still refused.
- [x] Tests: derivative residual still refused; irregular dt still `unavailable`
- [x] Do **not** treat Gate E residual-Q as diffusion \(a(x)\)
- [x] Keep FTR default off; `validation_level=model_derived` unless a new evidence class is preregistered
- [x] Do not enable FTR/metrics as clinical endpoints; ds003944 coverage failure stays a coverage failure

### D. Diffusion family

Already: estimator, OD-TQ1, ds004100 overlay, ingest `drift=None`.

- [x] Document ingest `computed` ≠ testable \(A_{bD}\) / \(R_{b/a}\)
- [x] Decide whether a **qualified independent drift** will ever be supplied (if no: drop those scalars from the public ingest contract or mark `not_testable` explicitly rather than silent NaN)
  - Decision (2026-09-01): ingest does **not** supply independent drift and will not use MNPS \(\dot x\) or Jacobian residuals as drift. Keep the estimator API. On the ingest write path, mark `A_bD` / `R_b_over_a` `not_testable` / `independent_drift_not_supplied` rather than silent NaN. A future qualified drift would be a new Type C overlay. Implementation: `project/mnps_v3/type_c_independent_drift_overlay.md`.
  - SL-003 (2026-09-02): C1 `alignment_only` authorized for **M1–M2 synthetic qualification only**. C2, empirical overlay, common-profile enablement, and ds004100 OD-EPI-001 modification are **not authorized**.
- [x] C1 M1–M2: estimator split (`residualize_increments` default false) + Gate C1-A identity + forbidden-source refusal
  - SL-004 RATIFY; SL-005 **freezes ingest** at consume-external-`b`. Handoff: `project/mnps_v3/df_drift_c1_analysis_handoff.md`.
  - C1 M3 cross-fit identification is **not an ingest work item** (SL-005). Do not implement `crossfit_local_chart_b` here.
- [ ] C2 `residualize_increments` — **NOT AUTHORIZED**
- [x] Keep Jacobian-residual substitution forbidden
- [x] Diffusion **D-M4** (not DF-DRIFT-C1-M4): preregistered `a_hat` serialization feasibility on OD-EPI-001 (translation/QC, not biology, not a drift overlay)
  - Prereg: `project/mnps_v3/d_m4_ds004100_diffusion_feasibility.md`
  - v3 replay: `E:\Science_Datasets\openneuro\processed\ds004100\neuralmanifolddynamics_ds004100_20260902_050532` (319/319 PASS)
  - Pre-v3 `075726` remains historical `/orthogonal_dynamics/` only
- [x] No default-profile enablement

### E. Destination family (committor)

Already: 1-D O2b adapter, TQ2/TQ2b/R PASS; O2 coarse FAIL preserved; BOAS scalar branch METHOD-LIMITED; ds004100 NOT_TESTABLE; ds004511 closed.

- [x] Freeze v1 claim: **explicit 1-D reaction coordinate + explicit first-hit A/B only**
- [x] Do **not** license 2-D/3-D/9D MNPS committor as v3 product export (article: 2-D finite-sample recovery closed)
- [x] Decide whether to emit \(V_{1/2}\) on O2b (today only first-hit `estimate_committor`)
  - Decision (2026-09-01): **do not emit** \(V_{1/2}\) on O2b. The first-hit estimator also does not emit it; it serializes `q_A_to_B`. Keep the potential internal to O2b quadrature until a dedicated serialization gate exists.
- [x] Keep \(\lvert\nabla q\rvert\) out until a synthetic gate exists
- [ ] No production committor overlay without a new prereg (BOAS G=17 pooled law failed local predictive criteria)
- [x] Stage labels remain not committor truth

### F. Resilience family (FAR)

Already: outcome serializer (OD-TQ3), fail-closed observational path, FAR-000 inheritance, ds003670 chain through 002D.

- [ ] Do not infer FAR from Jacobian, FTR, reachability, or spontaneous trajectories
- [x] **Park FAR-EXT-002E as `BLOCKED_EXTERNAL_CUSTODY` (SL-003, 2026-09-02)**
  Resume only when complete E1+E2 publisher/OA article XML is available.
  No live 0.2, freeze, \(R(\rho)\), or FAR-003 until then.
  Binder exists and fails closed; E3/E4 already PASS.
- [ ] Not yet authorized: FAR-003 home/away, FAR-004 DEV manifest, FAR-005 curve, FAR-006/007 held-out
- [ ] Measuring \(R(\rho)\) is a separate prereg after protocol+timebase+interpretability pass
- [ ] Frequency ≠ amplitude (ds006036 photic remains NOT_TESTABLE)
- [ ] Do not shorten the NMD window to rescue FAR or StateQ

### G. Spread family (stochastic reachability)

- [x] Keep article definition: predictive-spread ellipsoid, not cones, not Gramian, not consciousness
- [x] Gate F (item C) is the only ingest path to computed `W_Q`
- [x] Separate analysis-repo coma reachability from `mndm.stochastic_reachability.v1` in docs
- [x] No silent Q from derivative residuals

### H. Deferred families (M0 — track, do not build)

Start none of these until D–F have an explicit M4 pass/fail/stop.

- [ ] **Predictive information:** keep distinct from NMD_StateQ. StateQ occupancy is not authorized (C3 \(N_{s,\mathrm{session},U,p}=1\)). If ever built: \(I(X_{\mathrm{past}};X_{\mathrm{future}})\) as its own schema, experimental namespace first
- [ ] **Transfer entropy:** nonlinear check on block-MNJ, not a biomarker; must not reuse FTR `transfer_*` names
- [ ] **FDT:** only with actual perturbation data; never assume FDT in NDT; never confuse with `.fdt`
- [ ] **Fisher / observability:** simulator or held-out observable only; circular if MNPS-from-Y is the sole \(Y\)
- [ ] **Scale consistency:** curves \(M(\tau)\) over windows; not a new scalar; not `tau_summary`

### I. Out of v3 ingest (theory / analysis / other programs)

- [ ] RVC \(\sigma(t)\), thalamic operators, HELS / HELS-A, \(M_b\), \(q^{\mathrm{met}}\): not ingest families
- [ ] NMD-StateQ: continue as its own program; do not fold into family HDF5
- [ ] NMD-QC-FLOAT: supporting FAR-EXT QC, not a manifold family
- [ ] Atlas/topology, RQA, path metrics: already owned elsewhere; do not rebrand

### J. Tests and release hygiene

- [ ] Keep `test_od_v1_integration_audit.py`: common profiles do not enable families; MNPS/J byte-equal with/without family export
- [ ] Keep fail-closed statuses (`not_testable`, `insufficient_support`, `invalid`) rather than zeros
- [ ] Subject- vs cohort-anchored layers never merged in family outputs
- [x] If a breaking namespace rename happens: version it, migrate, do not silently repurpose `/orthogonal_dynamics/` (Round 1: canonical write + legacy read; no dual-write)
- [ ] Diary + science-lead note after each gated step (existing `NNN_YYYYMMDD_*.md` convention)

---

## 5. Recommended sequence (smallest next steps)

```text
1. A + B     freeze names and catch up docs          (done)
2. D docs    make A_bD / R_b/a ingest semantics explicit (done)
3. C Gate F  W_Q freeze + opt-in summarize             (done 2026-09-02)
4. F custody FAR-EXT-002E PARKED BLOCKED_EXTERNAL_CUSTODY
5. D C1      DF-DRIFT-C1 M1–M2 FROZEN in ingest (SL-005)
             M3+ → nmd-analysis (df_drift_c1_analysis_handoff.md)
6. E         no new committor overlay without new prereg
7. H         deferred families stay M0
8. Ingest    continue v3 control-plane (validity/grain/support already
             landed as R2–R4; chart-robustness plumbing / docs/release)
```

Do not implement predictive information, TE, FDT, Fisher, or \(M(\tau)\) to “complete the chart.” Completeness here means **every nominated object has an explicit status** (computed / gated / not_testable / not in contract), not that every object has a number.

---

## 6. Claim ceilings (copy onto any v3 overlay)

```text
Safe:     MNPS is a versioned chart. MNJ is a chart Jacobian.
          α, ω, FTR are mathematical provenance when J_hat exists.
          Three dynamical families can serialize behind fail-closed gates.
          ds004100 can emit chart-space a_hat without biology.
          Opt-in Gate F W_Q is a chart-space discrete predictive-spread
          object from discrete_transition_residual_covariance, not
          process noise.
          C1 library alignment of truth-known chart b leaves a_hat unchanged.
          Identification of empirical b is an analysis-repo experiment.

Not:      latent SDE (b, a, K^h)
          licensed empirical NDT α/ω/G_peak
          Gate E Q as process noise or a(x)
          2-D/3-D committor or stage-label q
          observational FAR / R(ρ)
          I_pred, TE, FDT, Fisher, M(τ) as NMD exports
          coma analysis reachability = stochastic_reachability.v1
          MNPS x_dot or Jacobian intercept as SDE b
          Type C C2 residualized a_hat
          empirical A_bD as NDT alignment
```

---

## 7. Evidence status for this plan

```text
VERDICT:
ACTIVE PROGRAMME / SL-005 FORK

Contract impact:
v3 Rounds 1–4 and C1 M1–M2 have landed. MNPS [m,d,e] / 9D and J_hat
remain frozen. M3 drift identification is not an ingest change.

Validation:
Inventory plus executed gates: naming/namespace, Gate F W_Q, D-M4
ds004100 serialization PASS, C1 Gate C1-A PASS/RATIFY (SL-004),
ingest freeze + analysis handoff RATIFIED (SL-005; 2026-09-03 amendments).

Safe claim:
v3 is a family-contract generation on frozen MNPS. Ingest can consume
a chart-space velocity supplied under an externally governed
qualification without changing a_hat. It does not estimate b and does
not adjudicate scientific independence.

Not established:
That all charter families should ever enter ingest.
That empirical C1, FAR R(ρ), or a production committor overlay is next.

Next gate (ingest):
Control-plane / chart-contract hygiene (docs/release; chart-robustness
plumbing as proposed in SL-001). Do not implement cross-fit b here.

Next gate (analysis):
DF-DRIFT-C1-M3 design in nmd-analysis. Handoff RATIFIED:
df_drift_c1_analysis_handoff.md. Type C ingest implementation frozen.
```
