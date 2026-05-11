# Protocol: N2 Sleep Spindle Event-Locked MNPS — Version 2 (C3 Primary, Full Load)

**Status**: Frozen canonical — 2026-05-08  
**Supersedes**: `protocol_n2_event_locked_v1.md` (F3 pilot, sub-1 to sub-5)  
**Protocol ID**: `sleep_spindle_event_locked_v2`

---

## Scope change from v1

v1 used PSG_F3 as primary channel and covered sub-1 to sub-5.  
v2 establishes PSG_C3 as primary based on the robustness check (session 011), extends to all available subjects, and adds baseline correction as the primary readout.

---

## Dataset

| Field | Value |
|-------|-------|
| Dataset | `ds005555` (OpenNeuro PSG sleep study) |
| Subjects | All 128 available PSG subjects |
| Modality | PSG EEG, 256 Hz |
| Sleep stage | **N2 only** |
| MNDM config | `config_ingest_ds005555_sleep_spindles.yaml` (6s/2s windows) |

---

## Primary detector: PSG_C3

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Channel | **PSG_C3** | Central; 4/5 direction agreement in pilot (v1 robustness check) |
| Library | YASA 0.7.0 | |
| `freq_sp` | `(12, 15)` Hz | |
| `duration` | `(0.5, 3.0)` s | YASA default |
| `min_distance` | 500 ms | YASA default |
| `thresh` | `{'rel_pow': 0.20, 'corr': 0.65, 'rms': 1.5}` | YASA defaults |
| `remove_outliers` | `False` | YASA default |
| `include` | `(2,)` | N2 only |

## Secondary/sensitivity detector: PSG_F3

Same parameters, same subjects. Run after C3.

---

## Primary readout: baseline-corrected event delta

**Rationale (from sub-5 audit, session 012)**:  
Raw event-vs-control delta is susceptible to time-of-night MNPS gradients when spindle distribution is clustered in early recording. Sub-5 showed Δd = −1.20 (raw) vs +0.06 (baseline-corrected). The confound is driven by bimodal spindle timing × MNPS-d time trend.

**Primary metric**:
> Δdim_bc = (event bin mean) − (pre_far bin mean), within spindle events

This isolates the spindle-locked trajectory change relative to the pre-event baseline (t = −30 to −10 s), removing level shifts and time-of-night gradients.

**Secondary metric**: raw event bin mean − matched control mean (absolute level, v1 style). Reported alongside baseline-corrected for transparency; flagged when confound signature is present (all bins show similar |Δ| relative to controls).

---

## QC gate (unchanged from v1, applied per subject per channel)

| Criterion | Threshold | Action on failure |
|-----------|-----------|-------------------|
| N2 spindle rate | 0.3–5.0/min | Flag as `qc_rate_warn` |
| All 5 bins populated | `n > 0` each | Flag |
| Control match rate | ≥ 0.80 | Flag |
| Finite MNPS fraction | ≥ 0.99 | Flag |
| Stage-transition exclusion | ≤ 30 % | Flag |

Additional: **confound detection** — if all 5 event-bin Δdim_raw values are within 10% of each other (flat profile), flag as potential time-of-night confound.

---

## Anomaly ledger (per subject, per channel)

For each subject/channel that passes QC gate, also record:
- Whether raw delta is confounded (flat across bins)
- Whether baseline-corrected delta changes sign vs raw delta
- Per-subject r(window_center_sec, d) as time-of-night gradient indicator

---

## Promotion threshold (unchanged)

A pattern is promoted to IVR if:
- ≥ 4/5 (or 80%) of subjects agree on direction for at least one MNPS dimension
- **Using baseline-corrected deltas as the primary measure**

---

## Deliverables

### For noetic diffusion analysis repository
1. **Per-subject H5 files** — 6s/2s MNDM, all 128 subjects, canonical config
2. **Per-subject event-locked Parquets** — C3 primary, F3 sensitivity

### For this repository
3. QC gate JSON per channel (`batch_event_locked_qc_psg_c3.json`, `_psg_f3.json`)
4. Baseline-corrected delta CSV (`channel_comparison_deltas_bc.csv`)
5. Anomaly ledger (`anomaly_ledger_v2.csv`)
6. Claim ledger updates

---

## Claim boundary

**Valid after this protocol**:
> "Using YASA 0.7.0 default parameters (freq_sp=(12,15)) on ds005555 PSG_C3 N2, baseline-corrected event-locked MNPS measurements show [direction] in [N]/128 subjects for [dimension]."

**Not yet valid**:
> "Sleep spindles causally alter MNPS."  
> "Results generalise beyond ds005555."  
> "MNPS effect reflects detector-independent physiology."

---

## Sensitivity checks (to run after C3 full load)

1. PSG_F3 same subjects (same QC gate, same baseline correction)
2. `remove_outliers=True` sensitivity on C3 for sub-1 to sub-5
3. N3 investigation deferred (near-zero C3 detection on pilot; separate protocol)

---

## Note on MNDM `--subject` flag

`--subject N` zero-pads to `sub-00N`, incompatible with ds005555's `sub-N` IDs. Always run without `--subject` flag for this dataset. See `project/issues/issue_subject_filter_zero_padding.md`.
