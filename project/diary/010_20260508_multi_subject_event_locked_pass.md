# 010 — 2026-05-08 — Multi-Subject Event-Locked MNPS: First Real Pass

## Session type
Implementation + first exploratory results

## What was done

### 1. Protocol frozen
`project/sleep spindles/protocol_n2_event_locked_v1.md` written.
Specifies: `ds005555`, N2 only, YASA 0.7.0, PSG_F3, canonical params (`freq_sp=(12,15)`, defaults), 3 controls/event, seed 42, QC gate thresholds, claim boundary.

### 2. MNDM pipeline run for sub-2 to sub-5
Features already computed from earlier session; `summarize` run for all 5 subjects together (without `--subject` filter — this dataset uses non-zero-padded IDs `sub-1..sub-5` and `--subject N` zero-pads to `sub-00N`, causing silently empty output).

New run: `neuralmanifolddynamics_ds005555_20260508_124335` containing H5 for sub-1 to sub-5.

### 3. Batch event-locked pipeline
`project/smoke_tests/batch_event_locked.py` written and run.
Pipeline per subject:
1. Find most-recent H5
2. Detect N2 spindles (canonical YASA, PSG_F3); write CSV to `eeg/` directory
3. Load payload from H5
4. Align events + match controls
5. Build event-locked export table → Parquet
6. Apply QC gate

### 4. QC gate results — all 5 PASS

| Subject | Rate/min | Bins | Match | Finite | Excl | Rows |
|---------|----------|------|-------|--------|------|------|
| sub-1 | 0.93 | 5/5 | 1.00 | 1.0000 | 0.12 | 5 382 |
| sub-2 | 0.55 | 5/5 | 1.00 | 1.0000 | 0.05 | 3 912 |
| sub-3 | 1.19 | 5/5 | 1.00 | 1.0000 | 0.04 | 11 163 |
| sub-4 | 0.76 | 5/5 | 1.00 | 1.0000 | 0.04 | 7 917 |
| sub-5 | 1.73 | 5/5 | 1.00 | 1.0000 | 0.04 | 19 680 |

All subjects: 5/5 bins populated, match rate 1.00, finite MNPS = 100%, transition exclusion within bounds.

### 5. Descriptive multi-subject MNPS
`project/smoke_tests/descriptive_multi_subject.py` written and run.
Total rows: 48 054 across 5 subjects.

**Event-bin (t=0 to +3 s) vs matched controls — per-subject delta:**

| Subject | Δm | Δd | Δe |
|---------|-----|-----|-----|
| sub-1 | −0.16 | −0.05 | +0.03 |
| sub-2 | +0.01 | +0.19 | +0.15 |
| sub-3 | +0.04 | −0.03 | +0.22 |
| sub-4 | −0.10 | +0.06 | −0.06 |
| sub-5 | +0.28 | +0.27 | −0.18 |

**Direction agreement: 3/5 positive for each dimension (60%).**

Pooled event bin vs controls (all subjects): Δm=+0.09, Δd=+0.12, Δe=−0.01.

## Key observations (exploratory only)

1. **High between-subject variability.** MNPS absolute values differ substantially across subjects before pooling — this is expected for robust-z projections with per-subject normalisation.

2. **No stable direction for any dimension at 60% agreement.** This is consistent with either (a) genuine null, (b) small effect size hidden by inter-subject variability, or (c) PSG_F3 not being the optimal channel for spindle MNPS detection.

3. **Sub-2 rate (0.55/min) is marginally low** (below reference centroid 1.88 but within the 0.70–4.80 range boundary; rate_in_range=True with the 0.3–5.0 gate).

4. **Sub-1 exclusion rate 12%** — higher than others (0.04–0.05), but within 30% threshold.

## What this means for the claim ledger

- IVR-007 (proposed): "5/5 subjects pass QC gate with YASA 0.7.0 canonical params on PSG_F3 N2." → **IVR** (internal validated result)
- No effect claim: direction agreement 60% is insufficient for any MNPS effect claim. This is correctly classified as exploratory measurement.

## What this does NOT mean

- Does not validate any spindle → MNPS causal relationship.
- Does not confirm or deny NDT predictions.
- Does not generalise beyond ds005555 PSG_F3 N2.

## Technical finding: subject filter zero-padding bug

`mndm --subject N` zero-pads to `sub-00N`, incompatible with `sub-N` BIDS naming in ds005555. Running `summarize` without `--subject` filter processes all subjects correctly. This is a usability issue, not a data bug.

## Next steps (as discussed with architect)

1. **Central-channel robustness** (PSG_C3): repeat protocol with same subjects, compare direction consistency.
2. If C3 direction agrees with F3 on ≥ 4/5 subjects for any dimension → promote to IVR (channel-robust pattern).
3. Consider per-subject MNPS normalisation before pooling (subtract per-subject baseline mean from matched controls).
4. Scale-mode audit deferred until central-channel robustness is established.
