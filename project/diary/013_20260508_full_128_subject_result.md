# 013 — 2026-05-08 — Full 128-Subject C3 Result

## Session type
Full-load execution + primary result

---

## What was produced

### H5 dataset
- **128/128 H5 files**, run `neuralmanifolddynamics_ds005555_20260508_164627`, 1.5 GB
- Config: 6s windows, 2s step, `config_ingest_ds005555_sleep_spindles.yaml`
- Deliverable for noetic diffusion analysis repository

### C3 event-locked batch
- 117/128 PASS, 11 WARN (all rate-related), 0 SKIP
- WARN breakdown: 7 subjects too low (<0.3/min), 4 too high (>5.0/min)
- Rate distribution: p25=0.71, p50=1.71, p75=2.94/min — well within literature range (Purcell: 1.88/min)

---

## Primary result: baseline-corrected MNPS-d, 98% consistency

**Event bin (t=0 to +3 s) baseline-corrected delta (event − pre_far), QC-passing subjects:**

| Dimension | Positive | Total | Agreement | Median Δbc |
|-----------|----------|-------|-----------|------------|
| **MNPS-d** | **119** | **121** | **98%** | **+0.215** |
| MNPS-m | 98 | 121 | 81% | +0.038 |
| MNPS-e | 94 | 121 | 78% | +0.024 |

**MNPS-d (diffusivity) is elevated at spindle time relative to the pre-event baseline in 119/121 QC-passing subjects.**

Including WARN subjects: MNPS-d 129/133 positive (97%), median +0.205.

---

## Interpretation (exploratory — no inferential statistics)

**Why this pattern?**  
Sleep spindles are bursts of synchronized 12–15 Hz activity originating in thalamocortical circuits. During a spindle, EEG shows increased sigma-band power and coherence. MNPS-diffusivity (d) captures how rapidly the MNPS trajectory explores phase space locally. A positive Δd at spindle time is consistent with increased neural variability or dynamic range during the spindle event, relative to the preceding N2 baseline.

**What this is:**
- A detector-derived, baseline-corrected, within-subject measurement pattern
- 98% sign consistency across 117 QC-passing subjects is well beyond the 4/5 (80%) promotion threshold
- F3 sensitivity comparison pending (batch running)

**What this is not:**
- Not ground-truth spindle events
- Not inferential evidence (no p-values, no effect size confidence intervals)
- Not a claim about mechanisms or NDT — spindle → MNPS-d is a descriptive correlation
- Not generalizable beyond ds005555 / YASA 0.7.0 / PSG_C3 until replicated

---

## WARN subjects (11)

| Subject | Rate | Type |
|---------|------|------|
| sub-3 | 0.26 | Low |
| sub-27 | 0.07 | Low |
| sub-91 | 0.09 | Low |
| sub-102 | 7.78 | High |
| sub-105 | 0.02 | Low |
| sub-109 | 0.05 | Low |
| sub-110 | 0.03 | Low |
| sub-111 | 0.09 | Low |
| sub-112 | 5.24 | High |
| sub-114 | 5.47 | High |
| sub-128 | 5.53 | High |

Low-rate subjects: likely short N2 duration or poor PSG_C3 signal. High-rate subjects: may have muscle artifact, broad sigma contamination, or genuine high spindle density. All pass other QC criteria (bins, match rate, finite MNPS).

---

## Next
- F3 sensitivity batch: running (started same session)
- Compare F3 vs C3 Δd_bc consistency across all 128 subjects
- Valid promoted claim after F3 confirmation:
  > "In ds005555, MNPS-d is consistently elevated during YASA-detected N2 spindle epochs relative to the pre-event baseline, across >95% of subjects on PSG_C3."
