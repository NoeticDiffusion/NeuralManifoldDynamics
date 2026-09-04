# Type C overlay — qualified independent chart-space drift

Status: **C1 M1–M2 frozen in ingest (SL-005, 2026-09-02)**. SL-004
RATIFY of Gate C1-A still stands. M3–M5 are **analysis-repo** work
(`df_drift_c1_analysis_handoff.md`). C2 `residualize_increments` remains
**not authorized**. Empirical C1 overlay, common-profile enablement, and
ds004100 OD-EPI-001 modification remain **not authorized**.
`crossfit_local_chart_b` is a provenance token for an externally
supplied field, not an ingest algorithm.

Authority: `mnps_v3_implementation_plan.md` §4 D; SL-003; SL-005;
ingest still calls `estimate_local_diffusion_geometry(..., drift=None,
residualize_increments=False)` and never passes `payload.x_dot`.

---

## 1. Why Type C

Type A would be docs-only. Type B would be a numerical backend with
equivalent semantics. Supplying a drift vector changes measurement meaning:

```text
Δx  →  residual = Δx − b Δt  →  â = cov(residual)/Δt     (C2)
       A_bD  = bᵀ â b / (|b|² tr(â))                      (C1 and C2)
       R_{b/a} = Δt |b|² / tr(â)
```

Today’s production `a_hat` is **raw increment covariance**. Residualized
\(a\) is a different object. Unlocking \(A_{bD}\) without residualizing \(a\)
is still a semantic addition (those scalars are currently
`not_testable` / `independent_drift_not_supplied`).

OD-TQ1 licenses adapter translation of increment covariance. It does **not**
license \(b\).

---

## 2. Split the contract (do not ship both at once)

### C1 — `alignment_only` (first overlay, if any)

- Leave `a_hat` / `D_total` / `d_diff` / `c_diff` **byte-equal** to
  `drift=None` on the same trajectory.
- Use an independently identified chart-space \(b\) **only** for
  `A_bD` and `R_b_over_a`.
- Claim: alignment of this velocity field with increment-covariance \(a\).
- Not claimed: Itô residualization, latent \(b(X)\), NDT drift.

Acceptance: synthetic C1 must keep `series/D_total` equal to the no-drift
path (NaN-equal, same valid mask).

### C2 — `residualize_increments`

- `residuals = increments - drift * dt` before forming `a_hat`.
- Changes `D_total`, `d_diff`, `c_diff` versus current ingest files.
- Requires its own qualification id/hash, distinct from OD-TQ1.
- Must not silently overwrite today’s `a_hat`. Prefer an explicit series
  name (`a_hat_residualized`) **or** a `contract_version` bump plus
  `provenance.settings.drift_residualization = supplied_drift_times_dt`.

The current estimator **used to** do C2 whenever `drift` was not `None`.
SL-003 required the C1 split first. Production ingest still must not call
the C2 path. C2 remains closed.

**Do C1 first. Do not enable C2 on ds004100 or infant EEG.**

---

## 3. What “independent” means

\(b\) must not be a deterministic function of the **same increment sample**
that estimates \(a\).

Closed **write** vocab for `diffusion.drift.source`:

| Token | Allowed? | Notes |
|---|---|---|
| `not_supplied` | default | Current ingest. Alignment `not_testable`. |
| `truth_known_chart_b` | C1/C2 validation | Simulator / unit-test \(b\). Not biology. |
| `crossfit_local_chart_b` | provenance token only | Algorithm lives in nmd-analysis. Ingest may later *accept* an externally supplied field with this source tag; it must not *fit* it. |
| `protocol_imposed_chart_b` | rare candidate | Known imposed restoring force / designed protocol. |
| `mnps_xdot` | **refuse** | Registry `mnps_xdot_as_sde_drift`. Chart velocity ≠ SDE \(b\). |
| `jacobian_intercept` | **refuse** | MNJ affine intercept is not Itô \(b\). |
| `jacobian_residual` | **refuse** | Already `jacobian_derivative_residual_as_diffusion`. |
| `local_increment_mean` | **refuse as \(b\)** | Same-sample contamination of \(A_{bD}\). |
| `committor_potential_gradient` | **refuse** | Circular with destination; \(V_{1/2}\) is not serialized. |

Any other string → `not_testable` / `independent_drift_not_supplied`.
No YAML alias to `x_dot` or `/jacobian/affine_intercept`.

---

## 4. YAML sketch (overlay only)

Do **not** add this block to `config_ingest_common_eeg.yaml` /
`config_ingest_common_fmri.yaml` / ephys common. Optional keys may later
live in `config_ingest_common_dynamical_families.yaml` with
`enabled: false` and `source: not_supplied`.

```yaml
dynamical_families:
  enabled: true
  diffusion:
    enabled: true
    estimator: local_increment_covariance
    translation_qualification:
      # Unchanged role: method tag for a_hat, not a b license
      qualified: true   # ignored as compute gate
      qualification_id: "OD-TQ1-MNDM-DIFFUSION-V1"
      qualification_contract_hash: "<existing a_hat hash>"

    drift:
      enabled: false
      mode: alignment_only          # C1; residualize_increments is C2
      source: not_supplied
      qualification_id: null
      qualification_contract_hash: null
      time_semantics: chart_state_per_unit_time
      # Do not put embargo / cross-fit knobs in ingest YAML.
      # Analysis owns those protocols and may later hand back a field.
```

Refuse the overlay if:

- `enabled: true` and `source` in the forbid set;
- `enabled: true` and `mode: residualize_increments` without a C2
  qualification id **and** hash distinct from OD-TQ1;
- `enabled: true` and `mode: alignment_only` but no drift array can be
  built from `source`;
- common EEG/fMRI/ephys profile imported the family YAML (already forbidden).

---

## 5. Code touch list (when authorized)

Keep the change local. Do not mix Gate F, FAR, or committor into the same
commit.

| File | Change |
|---|---|
| `mndm/src/mndm/dynamical_families/registry.py` | Keep forbids. Optionally document `drift.source` vocab next to diffusion. |
| `mndm/src/mndm/pipeline/dynamical_families_export.py` | Build `drift` only from a validated source helper. Default remains `None`. Never pass `payload.x_dot`. |
| **new** `mndm/src/mndm/dynamical_families/chart_drift.py` | Closed vocab, refuse forbidden sources, return `(array \| None, failure_reason)`. |
| `mndm/src/mndm/dynamical_families/diffusion_geometry.py` | No semantic change if C1: pass `drift` for alignment only. **C2 must not reuse the current `drift is not None` residualization without an explicit mode flag.** Split residualization from alignment if C1 is implemented: today one `drift=` argument does both. |
| `mndm/config/config_ingest_common_dynamical_families.yaml` | `drift.enabled: false`, `source: not_supplied`. |
| Dataset overlay (new, not ds004100 OD-EPI-001) | Explicit C1 qualification. New id, e.g. `DF-DRIFT-001-C1-...`. |
| Tests | See §7. |
| Docs | CONFIG_GUIDE, Output_variables_guide, `reporting/dynamical_families.md`, CHANGELOG, this file’s status. |

### Required estimator split before C1

Current code:

```text
if drift is None:
    residuals = increments
else:
    residuals = increments - drift * dt
    # and A_bD from neighbourhood mean of drift
```

C1 needs a **mode** (or two arguments):

```text
residualization: none | supplied_drift_times_dt
alignment_drift: ndarray | None
```

Without that split, “pass drift for A_bD” silently changes `a_hat`.
That would be an accidental C2.

Suggested function signature (landed M1):

```python
estimate_local_diffusion_geometry(
    ...,
    drift=None,
    residualize_increments: bool = False,
    drift_source: str | None = None,
)
```

- C1: `drift=b_hat`, `residualize_increments=False`
- C2 library flag: `residualize_increments=True` is refused
  (`invalid` / `c2_residualize_increments_not_authorized`) until C2 is
  separately authorized.
- ingest default: `drift=None`, `residualize_increments=False`

Invariants:

- `residualize_increments=True` → `invalid` /
  `c2_residualize_increments_not_authorized` (C2 not authorized)
- forbidden `source` → do not call the estimator with a vector; write
  `not_testable` plus the closed forbid reason on alignment only;
  leave `a_hat` on the no-drift path

---

## 6. HDF5 / certificate write contract

When C1 alignment is `computed`:

```text
summary.A_bD_computation_status = computed
summary.R_b_over_a_computation_status = computed
summary.drift_alignment_failure_reason = null   # or omit
provenance.settings.drift_source = <vocab>
provenance.settings.drift_mode = alignment_only
provenance.settings.drift_residualization = none
provenance.drift_qualification_id = DF-DRIFT-001-...
provenance.drift_qualification_contract_hash = ...
claim_status = no_biological_claim
measurement_validity = not_assessed
```

When C1 is requested but source fails:

```text
computation_status of a_hat = computed   # if increment support holds
A_bD_computation_status = not_testable
drift_alignment_failure_reason = <closed reason>
```

Do not set diffusion `measurement_validity=translation_qualified` from a
drift overlay. Drift TQ is a **nested** provenance tag, analogous to
OD-TQ1-as-method-tag after 2026-09-01.

Legacy files without the three summary fields: readers use `not_recorded`.
They must not infer `computed` alignment from NaN `A_bD` or from
`computation_status=computed` on `a_hat`.

---

## 7. Tests (minimum)

1. **Default ingest unchanged.** Overlay without `drift.enabled` still
   writes `not_testable` / `independent_drift_not_supplied` and all-NaN
   `A_bD` / `R_b_over_a`.
2. **C1 does not change `a_hat`.** Same Brownian path, `drift=known`,
   `residualize_increments=False` → `D_total` matches `drift=None`.
3. **C2 does change `a_hat` when authorized.** Known \(b\) subtracted
   recovers \(\mathrm{tr}(a)\) closer to true diffusion than raw increments
   when \(|b|\) is large (truth-known SDE).
4. **Forbidden sources refuse.** YAML `source: mnps_xdot` / passing
   `payload.x_dot` / Jacobian intercept → alignment `not_testable`, no
   residualization.
5. **Anisotropy ranking preserved** on C1 with true \(b\) (existing
   `test_diffusion_geometry_recovers_anisotropy_and_drift_alignment`).
6. **Common profiles** still do not enable families
   (`test_od_v1_integration_audit.py`).
7. **MNPS / `J_hat` byte-equal** with vs without the drift overlay.

Do not add a test that “succeeds” by feeding \(\dot x\) into `drift=`.

---

## 8. Staging (M0–M6)

```text
M0  This file. chart-b ≠ x_dot ≠ MNJ intercept ≠ latent b(X)
M1  YAML + closed vocab + estimator mode split; common profiles off
M2  Synthetic: C1 a_hat equality; x_dot refusal; known A_bD ranking
    INGEST LINE FROZEN HERE (SL-005)
M3  ANALYSIS: cross-fit vs same-sample bias (nmd-analysis)
M4  ANALYSIS: one overlay, methods audit, not biology (not ds004100 ictal)
M5  ANALYSIS: downstream utility only after M4
M6  Ingest docs only if a qualified overlay ever returns
```

ds004100 OD-EPI-001 remains an **`a_hat` translation** overlay. It is not
the C1/C2 drift overlay.

---

## 9. Claim ceiling

```text
Safe:     Estimator API can consume a supplied chart-space b for C1
          alignment without changing a_hat (Gate C1-A). Ingest does not
          supply a vector.

Not:      MNPS x_dot is b.
          Jacobian intercept is b.
          Computed A_bD is NDT drift-diffusion alignment on empirical EEG.
          C2 a_hat is the same object as current ingest a_hat.
          R_b/a under C1 is an Itô drift-to-diffusion ratio.
```

---

## 10. Next gate

Ingest C1 line is frozen. M3–M5 live in nmd-analysis
(`df_drift_c1_analysis_handoff.md`). C2 remains closed. ds004100
OD-EPI-001 remains an **`a_hat` translation** overlay, not a drift
experiment. `crossfit_local_chart_b` is not an ingest estimator.
