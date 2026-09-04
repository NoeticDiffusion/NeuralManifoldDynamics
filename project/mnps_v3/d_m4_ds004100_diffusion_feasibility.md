# D-M4 — ds004100 qualified diffusion feasibility (v3 contract)

Status: **PASS** 2026-09-02 (v3 replay `20260902_050532`). Translation / dataset-QC only. **Not biology.**

This is plan item D “Empirical M4” in `mnps_v3_implementation_plan.md`.
It is **not** the Type C independent-drift overlay
(`type_c_independent_drift_overlay.md`). Ingest remains `drift=None`.

Historical pre-v3 gate: `project/orthagonal_axis/od_epi_001_preregistration.md`
and run
`E:\Science_Datasets\openneuro\processed\ds004100\neuralmanifolddynamics_ds004100_20260815_075726`.
That run **does not close this M4**: it writes `/orthogonal_dynamics/`, has
no sibling certificate/grain/`support_signature`, and has no
`A_bD_computation_status`.

---

## Measurement question

On the frozen overlay
`mndm/config/sources/openneuro/config_ingest_ds004100_od_epi_001_diffusion.yaml`,
does current ingest serialize **v3** diffusion objects when estimator
support holds, with fail-closed destination/FAR, explicit
`not_testable` alignment scalars, and no biological contrast?

---

## Frozen overlay / certificate

```text
qualification_id: OD-TQ1-MNDM-DIFFUSION-V1
qualification_contract_hash: d80dffd7b94e7f5260fae66b3fdd2dd505bca3ad29e3c074a6b3d7cbaffe1722
```

Role after 2026-09-01: **method tag in provenance**, not a compute gate,
not `measurement_validity=translation_qualified` for diffusion.

Coordinate layer: `subject_anchored`. Chart: 3D `[m, d, e]`.
Estimator: `local_increment_covariance`. `drift=None`.
`max_dt_relative_deviation: 0.05`. Do not widen dt to raise `computed` count.

---

## Non-targets (any computed value is a gate failure)

- Destination / committor (no explicit A/B + RC).
- Resilience / FAR (no perturbation protocol).
- Spread / `W_Q` / `/stochastic_reachability`.
- Type C drift (`A_bD` must stay `not_testable`).
- Ictal vs interictal diffusion contrast, \(D_\perp/D_{\mathrm{total}}\),
  NDT \(a(x)\), licensed \(\alpha/\omega\).

---

## Run contract

```text
python -m mndm.cli summarize --dataset ds004100
  --config mndm/config/sources/openneuro/config_ingest_ds004100_od_epi_001_diffusion.yaml
  --data-dir M:\datasets\received\openneuro
  --out-dir E:\Science_Datasets\openneuro\processed
  --fit-anchor --n-jobs 4 --mem-budget-gb 12
```

Reuse the existing feature table. Complete cohort (no subject filter with
`--fit-anchor`). New timestamped run directory. Do not overwrite `075726`.

---

## Acceptance criteria (every HDF5)

1. Canonical write `/dynamical_families/{diffusion,destination,resilience}/v1`.
   No new `/orthogonal_dynamics/` write. No `/stochastic_reachability`.
2. `/mnps_3d` and `/jacobian/J_hat` present.
3. `/support_signature/v1` present.
4. Diffusion `grain/biological_unit=subject`,
   `direct_between_subject_inference=forbidden`.
5. If increment support holds: `computation_status=computed`,
   `measurement_validity=not_assessed`, `claim_status=no_biological_claim`.
   Provenance may copy OD-TQ1 id/hash; that does **not** set
   `translation_qualified`.
6. `summary.A_bD_computation_status=not_testable`,
   `R_b_over_a_computation_status=not_testable`,
   `drift_alignment_failure_reason=independent_drift_not_supplied`.
   Series `A_bD` and `R_b_over_a` all-NaN (not zero).
7. If support fails: `not_testable` / `insufficient_support` / `invalid`
   with an explicit reason (including `materially_irregular_increment_timestep`).
   No zero tensors.
8. Destination: `not_testable` /
   `explicit_A_B_reaction_coordinate_contract_not_configured`.
9. Resilience: `not_testable` / `no_perturbation_protocol`.

No threshold or condition contrast is part of this gate.

---

## Evidence classes

| Object | This gate may show |
|---|---|
| v3 write path + certificate + grain + support_signature | internal validated serialization |
| `computed` 3D `a_hat` on ds004100 | feasibility of increment covariance under this overlay |
| OD-TQ1 id/hash in provenance | method tag copied |
| `A_bD` | remains not testable |

| Object | This gate must not claim |
|---|---|
| Latent \(a(x)\) / \(b\) | no |
| Ictal biology | no |
| Type C drift | no |
| Pre-v3 `075726` as v3 canonical write | no (legacy read only) |

---

## Legacy 075726

Use only as a **historical** adapter-translation census (319/319 `computed`
`a_hat` under `/orthogonal_dynamics/`). Readers of those files must fill
missing v3 fields as `not_recorded` and must not infer `chart_3d=yes` from
`/mnps_3d` or alignment validity from NaN `A_bD`.

---

## Result (2026-09-02)

v3 replay:

```text
E:\Science_Datasets\openneuro\processed\ds004100\neuralmanifolddynamics_ds004100_20260902_050532
```

Census: `project/mnps_v3/results/d_m4_v3_20260902_050532_census.json`

| Criterion | Result |
|---|---|
| HDF5 count | 319 |
| namespace | `canonical` 319; `/orthogonal_dynamics/` 0 |
| `/mnps_3d`, `/jacobian/J_hat`, `/support_signature/v1` | 319 |
| `/stochastic_reachability` | 0 |
| diffusion | 319 `computed` / `not_assessed` / `no_biological_claim` |
| OD-TQ1 provenance id | 319 `OD-TQ1-MNDM-DIFFUSION-V1`; `validation_level=mndm_translation_validated` (method tag, not regime validity) |
| `A_bD` / `R_b_over_a` | 319 `not_testable` / `independent_drift_not_supplied`; series all-NaN |
| `a_hat` | 319 tensors ending in `(3,3)` |
| grain | 319 `biological_unit=subject`, `direct_between_subject_inference=forbidden` |
| destination / resilience | 319 `not_testable` with expected reasons |

Incidental, out of M4 scope: one FTR `OverflowError` on
`sub-HUP080_ses-presurgery_interictal_run-01_acq-ecog` (`d_resp` energy
overflow). Summarize continued; that is not a diffusion-family failure.

**Not established:** ictal biology, latent \(a(x)\), Type C drift, licensed NDT.

