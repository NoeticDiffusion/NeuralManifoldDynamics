# 015 — 2026-05-08 — F3 Sensitivity Batch Complete, Cross-Channel Agreement

## Session goal

Run F3 full-batch sensitivity, compute baseline-corrected sign agreement, compare F3 vs C3, lift provisional flags on IVR-012/013.

## F3 batch result

- 128 subjects run, 111 PASS, 17 WARN (0 SKIP).
- More WARN than C3 (17 vs 11), all rate-related (low frontal spindle density in some subjects).

### Baseline-corrected F3 event-bin direction (QC-pass, 111 subjects)

| Coordinate | Positive | Fraction | Median Δbc |
|------------|----------|----------|-----------|
| MNPS-m     | 93/111   | 84%      | +0.050    |
| **MNPS-d** | **110/111** | **99%** | **+0.264** |
| MNPS-e     | 93/111   | 84%      | +0.050    |

## Cross-channel sign agreement (all 128 subjects, both channels)

| Coordinate | C3/F3 agree | % | C3 median | F3 median |
|------------|-------------|---|-----------|-----------|
| MNPS-d     | 126/128     | 98% | +0.206   | +0.269    |
| MNPS-m     | 115/128     | 90% | +0.036   | +0.049    |
| MNPS-e     | 106/128     | 83% | +0.021   | +0.050    |

MNPS-d agreement is essentially unanimous across detectors. MNPS-m and -e are directionally consistent but noisier.

## Claims updated

- **IVR-015 added**: F3 sensitivity confirmed (110/111, 99%), cross-channel d-agreement 98%.
- **IVR-012 provisional flag lifted**: MNPS-d C3 result (115/117) no longer pending.
- **IVR-013 provisional flag lifted**: MNPS-m/e C3 results no longer pending.

## Fixed scripts

- `_baseline_corrected_all.py`: added `--channel` argument; output filename now channel-parameterized.
- `_channel_agreement.py`: new script computing per-dimension sign agreement across channels.

## Important caveat (registered in IVR-015)

F3 and C3 share the same MNPS trajectory (same H5 file). Cross-channel agreement is a **detector consistency check**, not independent MNPS replication. The result shows that the MNPS-d positive shift is not an artifact of a single detection channel, but it does not constitute two independent measurements of the trajectory.

## Final promoted claim (IVR-012, now unconditional)

> In ds005555, using YASA 0.7.0 detector-derived N2 spindle events on PSG_C3 with canonical `freq_sp=(12,15)` settings, baseline-corrected MNPS-d (operational MNPS-d coordinate; event bin − pre_far bin) was positive in **115/117 QC-passing subject runs** (98%), with median Δd_bc = +0.216. F3 sensitivity: 110/111 (99%), median +0.264. Cross-channel sign agreement: 126/128 (98%). This is an internal validated measurement-layer result, pending downstream statistical analysis.

## Remaining next steps (for analysis repository)

1. Binomial sign test for Δd_bc (n=117, observed=115; p will be < 10⁻²⁵).
2. Bootstrap 95% CI on median Δd_bc.
3. Rate/artifact sensitivity: Δd_bc vs spindle rate correlation.
4. Baseline robustness: pre_near and matched-control mean as alternative baselines.
5. F3 vs C3 per-subject delta correlation (quantitative, not just sign).
