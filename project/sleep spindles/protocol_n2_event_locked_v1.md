# Protocol: N2 Sleep Spindle Event-Locked MNPS — Version 1

**Status**: Frozen canonical baseline — 2026-05-08  
**Track**: Sleep spindle / event-locked MNPS  
**Protocol ID**: `sleep_spindle_event_locked_v1`

---

## Dataset

| Field | Value |
|-------|-------|
| Dataset | `ds005555` (OpenNeuro PSG sleep study) |
| Modality | PSG EEG, 256 Hz |
| Sleep stage | **N2 only** (`include=(2,)`) |
| Subjects (this run) | sub-1 … sub-5 |

---

## Spindle detector

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Library | YASA 0.7.0 | `detector:yasa-0.7.0` |
| Channel | **PSG_F3** (primary) | Frontal, available in all PSG runs |
| `freq_sp` | `(12, 15)` Hz | Narrow sigma band, AASM-aligned |
| `freq_broad` | `(1, 30)` Hz | YASA default |
| `duration` | `(0.5, 3.0)` s | YASA default |
| `min_distance` | `500` ms | YASA default |
| `thresh` | `{'rel_pow': 0.20, 'corr': 0.65, 'rms': 1.5}` | YASA defaults |
| `remove_outliers` | `False` | YASA default; add as sensitivity check |
| `include` | `(2,)` | N2 only |

**Expected N2 rate**: ~0.7–4.8 spindles/min per Purcell et al. 2017 (N=11 630; 1.88/min central mean).  
**Observed range** (sub-1 to sub-5, PSG_F3): 0.55–1.72/min N2 (mean 1.03/min). 4/5 subjects within reference interval.

> **Claim boundary**: Detector-derived events (YASA 0.7.0), **not ground truth**. No expert annotation comparison. N3 spindle detection not yet validated on this channel/dataset.

---

## MNPS feature windows

| Parameter | Value |
|-----------|-------|
| Profile | `sleep_spindle_event_locked_v1` |
| Window length | 6 s |
| Window step | 2 s |
| Config overlay | `config_ingest_ds005555_sleep_spindles.yaml` |
| Stage label mapping | 30 s BIDS `events.tsv` → interval labels on 6 s windows |

---

## Event alignment

| Parameter | Value |
|-----------|-------|
| Reference point | `peak` |
| Bins | `pre_far` [−30, −10 s], `pre_near` [−10, 0 s], `event` [0, 3 s], `post_near` [3, 10 s], `post_far` [10, 30 s] |
| Stage transition margin | 5 s |
| Stage filter | N2 only |

---

## Matched controls

| Parameter | Value |
|-----------|-------|
| Stage | N2 |
| Controls per event | 3 |
| Exclusion margin | 30 s |
| Seed | 42 |

---

## QC gate (per subject — must pass before descriptive analysis)

| Criterion | Threshold | Action on failure |
|-----------|-----------|-------------------|
| N2 spindle rate | 0.3–5.0/min | Flag; include in report but mark as `qc_rate_warn` |
| All 5 bins populated | `n > 0` in each | Flag |
| Control match rate | ≥ 0.80 | Flag; re-check exclusion margin |
| Finite MNPS fraction | ≥ 0.99 | Flag; check HDF5 integrity |
| Stage-transition exclusion | ≤ 30 % | Flag if high |

Subjects that fail ≥ 2 criteria are excluded from the descriptive summary but retained in the QC report.

---

## Export format

- **Primary**: Parquet (`*_event_locked.parquet`)
- **Columns**: see `event_locked_export.py` contract (54 columns)
- **Provenance**: SHA-256 hash of annotation CSV in every row
- **Source tag**: `detector:yasa-0.7.0` + parameter hash

---

## Claim boundary — what this protocol supports

**Valid after this protocol**:
> "Using YASA 0.7.0 default parameters (freq_sp=(12,15)) on ds005555 PSG_F3, event-locked MNPS measurements for N2 spindle candidates and matched N2 controls are produced for subjects 1–5, with QC-gated provenance."

**Not yet valid**:
> "Sleep spindles have a measurable MNPS effect in this dataset."  
> "The MNPS pattern generalises beyond ds005555."  
> "Results reflect true spindle biology rather than detector artifacts."

---

## Sensitivity checks (future work)

1. Central channel: re-run on `PSG_C3` (separate protocol, same QC gate)  
2. `remove_outliers=True` sensitivity: compare descriptive pattern under stricter detection  
3. N3: investigate why PSG_F3 gives ~0/min N3; try PSG_C3 for N3  
4. Scale-mode audit: after central-channel robustness confirmed
