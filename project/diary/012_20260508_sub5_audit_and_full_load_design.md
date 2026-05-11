# 012 — 2026-05-08 — Sub-5 C3 Audit + Full Load Design

## Session type
Methodological audit + full-load scoping

---

## Sub-5 C3 Δd audit: finding

### Root cause
`Δd = −1.20` for sub-5 C3 is **not a spindle effect**. It is a **time-of-night MNPS confound**:

- Sub-5 C3 spindles occur in **two early-recording clusters** (~2000–2400 s, i.e. 30–40 min in) plus a late-night cluster (~22 900–24 700 s, i.e. ~6.4 h in).
- MNPS-d has a **strong positive time trend** in this subject: r(t, d) = +0.55. Early-recording d ≈ −1.80; late-recording d ≈ −0.75 to +0.37.
- The **control matching fails for the early cluster**: Q1 (t < 2 400 s) has 1 976 event rows but only 14 control rows. The time-of-night quartile matching is overwhelmed by the bimodal spindle distribution — no N2 non-spindle windows exist near the early cluster.
- Consequently, controls are sampled from later time points (higher d), producing an artifactual negative Δd.

### Proof: baseline-corrected deltas
Subtracting the `pre_far` bin mean from each bin (within spindle events only):

| Sub-5 C3 bin | Δm_bc | Δd_bc | Δe_bc |
|---|---|---|---|
| pre_far | +0.00 | +0.00 | +0.00 |
| pre_near | −0.04 | +0.03 | −0.03 |
| **event** | **−0.06** | **+0.06** | **−0.04** |
| post_near | −0.09 | −0.02 | −0.03 |
| post_far | −0.08 | −0.07 | −0.02 |

**Δd_bc = +0.06** (positive, small). The anomalous −1.20 disappears entirely with baseline correction.

---

## Methodological consequence: two distinct comparisons

The existing pipeline reports **event condition vs matched controls** (absolute level). This conflates:
1. **Where in MNPS space spindles tend to occur** (time-of-night confound susceptible)
2. **What MNPS does at t = 0 relative to the spindle event** (event-specific signal)

For a clean event-locked MNPS signal, the primary readout for the full load should be:

> **Baseline-corrected event delta = event bin mean − pre_far bin mean (within spindle events)**

This removes between-subject and time-of-night level shifts, isolating the spindle-locked trajectory change. The matched control comparison remains as a secondary/consistency check.

### Baseline-corrected event-bin picture (all 5 subjects, both channels)

For **MNPS-d** (the most consistent dimension after baseline correction):

| Subject | F3 Δd_bc | C3 Δd_bc |
|---------|----------|----------|
| sub-1 | +0.012 | +0.090 |
| sub-2 | +0.239 | +0.280 |
| sub-3 | +0.036 | −0.075 |
| sub-4 | +0.043 | +0.021 |
| sub-5 | +0.110 | +0.061 |

F3: 5/5 positive. C3: 4/5 positive (sub-3 negative, but also the flagged-rate subject). Cross-channel agreement on d: 4/5.

This is a markedly cleaner pattern than the raw event-vs-control Δd.

---

## Full load design

### Goal (as stated by architect)
Produce ds005555 as a full H5 dataset. H5 files are consumed by the noetic diffusion analysis repository.

### Scope
- **All available ds005555 PSG subjects** (need to discover how many)
- **MNDM 6s/2s** for subjects without H5
- **C3 primary event detection** (YASA 0.7.0 canonical)
- **F3 sensitivity** run on same subjects
- **Per-subject QC gate** (existing batch_event_locked.py thresholds)
- **Per-subject baseline-corrected deltas** as primary readout
- **Anomaly ledger** for subjects failing QC or showing sub-5-type confounds

### Primary output (for noetic diffusion analysis)
- Per-subject H5 files (6s/2s MNDM) — **the core deliverable**
- Per-subject event-locked Parquet (C3 primary, F3 sensitivity) — analysis artifact
- QC gate JSON — reproducibility record

### What this session clarified
- Raw event-vs-control delta is susceptible to time-of-night confounds when spindle distribution is clustered
- Baseline-corrected delta (event bin − pre_far) is the appropriate primary MNPS readout
- Sub-5 C3 Δd anomaly is resolved; sub-5 is not a bad subject — just needs baseline correction

---

## Claims to file
- DQF-003 (proposed): raw event-vs-control Δd for sub-5 C3 is confounded by time-of-night gradient; not suitable as event-locked signal without baseline correction
- IVR-012 (proposed): baseline-corrected MNPS-d shows positive direction in 4–5/5 subjects on both F3 and C3
