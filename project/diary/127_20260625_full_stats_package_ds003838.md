# 127 — 2026-06-25 — Full verified statistics package for ds003838

## Context

After the ECG correction (diary 126), re-ran `summarize` and then produced the
complete statistics package requested by the analysis repo.

## Run used

`J:/processed/openneuro/ds003838/neuralmanifolddynamics_ds003838_20260624_151505`
(26 460 windows, 62 subjects, stages: rest / listen / mem5 / mem9 / mem13)

## Script

`project/scripts/run_stage_contrasts.py`

Covers all three analyses in one run:
- Stage medians (all primary + coupling metrics)
- Friedman repeated-measures across all 5 stages
- Pairwise Wilcoxon + Cohen's d for all stage pairs

## Key results

### Friedman — primary metrics (N=62)

| Metric | χ² | p | sig |
|--------|-----|---|-----|
| traj_path_length | 226.2 | 8.8×10⁻⁴⁸ | *** |
| traj_efficiency | 221.8 | 7.8×10⁻⁴⁷ | *** |
| vagal_index | 88.0 | 3.4×10⁻¹⁸ | *** |
| sympathetic_index | 63.7 | 4.8×10⁻¹³ | *** |
| ecg_hrv_hr_mean_bpm | 62.6 | 8.3×10⁻¹³ | *** |
| ecg_hrv_sdnn_ms | 62.0 | 1.1×10⁻¹² | *** |
| ecg_hrv_rmssd_ms | 24.1 | 7.7×10⁻⁵ | *** |
| ecg_hrv_pnn50 | 22.6 | 1.5×10⁻⁴ | *** |
| anchor_index | 13.6 | 8.6×10⁻³ | ** |
| vascular_index | 13.2 | 1.0×10⁻² | * |
| traj_mean_curvature | 7.6 | 0.11 | ns |

### C3: listen vs mem* (vagal_index)

| Contrast | Cohen's d | Wilcoxon p |
|---------|-----------|------------|
| listen vs mem5 | +1.718 | 6.9×10⁻¹¹ |
| listen vs mem9 | +1.750 | 1.8×10⁻¹⁰ |
| listen vs mem13 | +1.995 | 1.2×10⁻¹⁰ |

### Friedman — coupling metrics (all 12 significant)

All 12 inter-network coupling terms are highly significant (χ² = 65–97,
p range 2.6×10⁻¹³ – 5.4×10⁻²⁰ ***). Strongest: central↔parietal,
temporal←central.

### Key summary (one-line per primary metric)

| Metric | Friedman p | C3 Cohen's d | C3 Wilcoxon p |
|--------|-----------|-------------|--------------|
| vagal_index | 3.4×10⁻¹⁸ | **+2.085** | 2.8×10⁻¹¹ |
| ecg_hrv_rmssd_ms | 7.7×10⁻⁵ | +0.055 | 2.9×10⁻⁵ |
| anchor_index | 8.6×10⁻³ | +0.525 | 5.7×10⁻⁴ |
| traj_path_length | 8.8×10⁻⁴⁸ | +0.767 | 3.8×10⁻⁷ |

## Outputs

All saved to `J:/processed/openneuro/ds003838/audit_ecg_ds003838/stats_package/`:

- `stage_medians.csv` (115 rows)
- `friedman_results.csv`
- `pairwise_contrasts.csv` (110 rows)
- `coupling_friedman.csv`
- `coupling_stage_medians.csv`
- `key_stats_summary.csv`

## Claim status

The non-monotonic listen-peak pattern is strongly validated on corrected data
across multiple metrics. vagal_index listen vs mem13 Cohen's d = 1.995
(very large effect). All coupling metrics are stage-dependent (p < 10⁻¹²).
The trajectory geometry metrics (path_length, efficiency) show the largest
Friedman statistics of all (χ² > 220).
