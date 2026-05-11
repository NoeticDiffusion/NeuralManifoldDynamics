# 014 — 2026-05-08 — Denominator Audit and Provisional Claim Promotion

## Session goal

Resolve denominator mismatch between diary (117/128 PASS), primary table (119/121), and WARN-inclusive (129/133), then tighten the promoted claim for the C3 MNPS-d finding.

## Root cause of denominator mismatch

Sub-1 through sub-5 had parquet files written in **two** processed-run directories:

- Pilot run: `neuralmanifolddynamics_ds005555_20260508_124335/`  (C3 pilot, sessions 011–012)
- Full-load run: `neuralmanifolddynamics_ds005555_20260508_164627/`  (all 128 subjects)

`_baseline_corrected_all.py` used `rglob("*.parquet")`, which matched both copies:

| Count | Source |
|-------|--------|
| 128   | unique subjects (full-load run) |
| +5    | sub-1 to sub-5 duplicates (pilot run) |
| = 133 | raw total (WARN-inclusive) |

Of those 5 duplicate subjects, 4 pass QC (sub-1, sub-2, sub-4, sub-5) → 117 + 4 = **121** appeared as "QC-pass event-bin rows."

## Fix

`_baseline_corrected_all.py` now deduplicates before analysis: for each subject, it keeps only the parquet with the most recent `st_mtime`.

### Verified denominator audit output

```
Unique subjects with parquet      : 128
QC PASS (from QC JSON)            : 117
QC WARN (from QC JSON)            : 11
Event-bin rows (deduplicated)     : 128  [== n_total_subs, OK]
Event-bin rows, QC-pass subjects  : 117  [== n_qc_pass, expected]
Denominators consistent: YES
```

### Corrected primary numbers (deduplicated, QC-pass only)

| Coordinate | Positive | Fraction | Median Δbc |
|------------|----------|----------|-----------|
| MNPS-m     | 97/117   | 83%      | +0.042    |
| **MNPS-d** | **115/117** | **98%** | **+0.216** |
| MNPS-e     | 92/117   | 79%      | +0.024    |

WARN-inclusive (128 subjects): d=125/128 (98%).

## Promoted claim (IVR-012, provisional)

> In ds005555, using YASA 0.7.0 detector-derived N2 spindle events on PSG_C3 with canonical `freq_sp=(12,15)` settings, baseline-corrected MNPS-d (operational MNPS-d coordinate; event bin − pre_far bin) was positive in **115/117 QC-passing subject runs** (98%), with median Δd_bc = +0.216. This is an internal validated measurement-layer result, pending F3 sensitivity confirmation and downstream statistical analysis.

**Claim category**: IVR (internal validated result), marked provisional.

**What "operational MNPS-d coordinate" means**: MNPS-d is the diffusivity axis of the 3D robust-z MNPS projection. It is not directly equated to neural variability or dynamic range; that interpretation is a separate plausible interpretation (PLI), not the result itself.

## Claims updated

- IVR-012: corrected to 115/117, denominator-audited, provisional flag.
- IVR-013: corrected (97/117 m, 92/117 e), provisional flag.
- IVR-014: unchanged.
- PLI-003 added: F3 sensitivity replication hypothesis (pending).

## F3 batch

The full F3 sensitivity batch (`batch_event_locked.py --channel PSG_F3`) for all 128 subjects was initiated in session 013 and is expected to complete. When finished, run `_baseline_corrected_all.py` (channel=PSG_F3), verify denominator, and compare F3 vs C3 sign agreement to promote PLI-003 → IVR-015 or revise provisional flags.

## Next steps for analysis repository

The following are **not for the `mndm` package**; they belong in a downstream statistics repository:

1. **Binomial sign test** for Δd_bc (null: 50% positive). n=117, observed=115/117. p will be extremely small; report exact p and effect-size CI.
2. **Subject-level effect size CI**: bootstrap 95% CI on median Δd_bc across subjects.
3. **Rate/artifact sensitivity**: correlate Δd_bc with N2 spindle rate per subject; check if WARN subjects cluster differently.
4. **F3 vs C3 sign agreement**: per-subject cross-channel direction table (m, d, e). Both channels use the same MNPS trajectory; agreement is a detector consistency check, not an independent MNPS replication.
5. **Baseline choice robustness**: repeat with `pre_near` (−10 to 0 s) as baseline and with matched-control mean as baseline. Compare direction agreement.

## Pending before removing provisional flags

- [ ] F3 full batch: denominator audit + sign agreement vs C3
- [ ] Denominator mismatch resolved (DONE this session)
- [ ] Analysis repo: binomial sign test run and p reported
