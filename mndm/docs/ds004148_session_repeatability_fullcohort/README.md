# ds004148 MNPS 3.4 session repeatability: full cohort

**Descriptive full-cohort analysis.** The metadata-only selection fixed all 60 participants, sessions 1–3, and resting EC/EO recordings before coordinate inspection. Every recording used the same frozen ds005385 EyesClosed anchor; target sessions were not used to refit it. The source-level QC rule and support floor (at least 30 eligible windows and 120 unique seconds) were fixed in advance.

## Processing and support

- Reused session-1 H5 and QC from the frozen baseline and re-extracted sessions 2–3 with the current MNPS 3.4 implementation. Session 1 was not reprocessed.
- Checked 360 H5s (60 participants × 3 sessions × EC/EO): frozen anchor ID/hash, per-window implementation identity, expected source units, and the expected 8-second windows at 4-second steps. QC matched all 26640 H5 windows.
- 2858/3240 participant × session × condition × coordinate summaries met the support floor. Complete 9D support met it in 306/360 participant-session-condition rows. Coordinate-specific support is preserved in the tables.
- Support-floor failures: 382 coordinate summaries across 54 participant-session-condition groups (session1 EC: 8; session1 EO: 11; session2 EC: 7; session2 EO: 10; session3 EC: 11; session3 EO: 7). Per-session/condition counts are groups with at least one coordinate below the support floor; detailed coordinate-level counts and exclusion reasons are in `repeatability_eligibility.csv` and `participant_session_levels.csv`.
- 100 realization-comparison rows have a channel-order or case-only label difference. These are reported as QC; comparisons remain eligible only when the normalized channel set, source surface, reference, preprocessing, estimator identity, time grid and units match.
- Included participant counts across the 81 repeatability rows range from 36 to 52 of 60.

## Session comparisons

Bias is later session minus earlier session. Median is the median absolute difference. Sample SD, 95% Bland–Altman limits and Spearman rank correlation are in `repeatability_summary.csv`; Spearman describes rank order, not agreement. Each row uses participants who pass the predeclared support and compatibility rules in both sessions.

| Measure | Coordinate | Sessions | n | ICC(A,1) | Bias | Median |Δ| |
|---|---|---|---:|---:|---:|---:|
| EC | delta_theta_level | session1 to session2 | 52/60 | 0.895 | 0.149 | 0.219 |
| EO | delta_theta_level | session1 to session2 | 45/60 | 0.480 | -0.039 | 0.290 |
| EC | alpha_level | session1 to session2 | 52/60 | 0.875 | 0.183 | 0.150 |
| EO | alpha_level | session1 to session2 | 45/60 | 0.846 | 0.063 | 0.160 |
| EC | beta_alpha_ratio | session1 to session2 | 52/60 | 0.835 | -0.071 | 0.065 |
| EO | beta_alpha_ratio | session1 to session2 | 45/60 | 0.844 | -0.064 | 0.292 |
| EC | gamma_level | session1 to session2 | 52/60 | 0.827 | 0.308 | 0.293 |
| EO | gamma_level | session1 to session2 | 45/60 | 0.686 | 0.090 | 0.267 |
| EC | hjorth_mobility | session1 to session2 | 52/60 | 0.806 | -0.040 | 0.208 |
| EO | hjorth_mobility | session1 to session2 | 45/60 | 0.407 | 0.058 | 0.306 |
| EC | alpha_theta_ratio | session1 to session2 | 52/60 | 0.709 | 0.580 | 0.637 |
| EO | alpha_theta_ratio | session1 to session2 | 45/60 | 0.891 | 0.131 | 0.135 |
| EC | permutation_entropy | session1 to session2 | 52/60 | 0.837 | -0.182 | 0.300 |
| EO | permutation_entropy | session1 to session2 | 45/60 | 0.621 | 0.005 | 0.267 |
| EC | alpha_centroid_hz | session1 to session2 | 52/60 | 0.954 | 0.010 | 0.199 |
| EO | alpha_centroid_hz | session1 to session2 | 45/60 | 0.924 | -0.172 | 0.220 |
| EC | alpha_envelope_cv | session1 to session2 | 47/60 | 0.823 | -0.146 | 0.264 |
| EO | alpha_envelope_cv | session1 to session2 | 43/60 | 0.674 | -0.147 | 0.234 |
| EC | delta_theta_level | session1 to session3 | 47/60 | 0.830 | 0.054 | 0.229 |
| EO | delta_theta_level | session1 to session3 | 47/60 | 0.120 | 0.006 | 0.465 |
| EC | alpha_level | session1 to session3 | 47/60 | 0.885 | -0.073 | 0.217 |
| EO | alpha_level | session1 to session3 | 47/60 | 0.688 | 0.048 | 0.260 |
| EC | beta_alpha_ratio | session1 to session3 | 47/60 | 0.751 | 0.084 | 0.131 |
| EO | beta_alpha_ratio | session1 to session3 | 47/60 | 0.676 | -0.059 | 0.352 |
| EC | gamma_level | session1 to session3 | 47/60 | 0.874 | -0.016 | 0.332 |
| EO | gamma_level | session1 to session3 | 47/60 | 0.598 | 0.075 | 0.466 |
| EC | hjorth_mobility | session1 to session3 | 47/60 | 0.752 | -0.018 | 0.260 |
| EO | hjorth_mobility | session1 to session3 | 47/60 | 0.041 | 0.006 | 0.754 |
| EC | alpha_theta_ratio | session1 to session3 | 47/60 | 0.675 | -0.307 | 0.631 |
| EO | alpha_theta_ratio | session1 to session3 | 47/60 | 0.363 | -0.003 | 0.225 |
| EC | permutation_entropy | session1 to session3 | 47/60 | 0.786 | -0.003 | 0.263 |
| EO | permutation_entropy | session1 to session3 | 47/60 | 0.285 | -0.039 | 0.410 |
| EC | alpha_centroid_hz | session1 to session3 | 47/60 | 0.792 | -0.136 | 0.438 |
| EO | alpha_centroid_hz | session1 to session3 | 47/60 | 0.898 | -0.130 | 0.269 |
| EC | alpha_envelope_cv | session1 to session3 | 45/60 | 0.741 | 0.169 | 0.340 |
| EO | alpha_envelope_cv | session1 to session3 | 45/60 | 0.600 | -0.094 | 0.255 |
| EC | delta_theta_level | session2 to session3 | 50/60 | 0.808 | -0.025 | 0.233 |
| EO | delta_theta_level | session2 to session3 | 48/60 | 0.796 | 0.064 | 0.360 |
| EC | alpha_level | session2 to session3 | 50/60 | 0.809 | -0.257 | 0.235 |
| EO | alpha_level | session2 to session3 | 48/60 | 0.732 | -0.011 | 0.176 |
| EC | beta_alpha_ratio | session2 to session3 | 50/60 | 0.802 | 0.167 | 0.110 |
| EO | beta_alpha_ratio | session2 to session3 | 48/60 | 0.710 | 0.010 | 0.332 |
| EC | gamma_level | session2 to session3 | 50/60 | 0.752 | -0.324 | 0.348 |
| EO | gamma_level | session2 to session3 | 48/60 | 0.759 | 0.006 | 0.475 |
| EC | hjorth_mobility | session2 to session3 | 50/60 | 0.720 | -0.031 | 0.328 |
| EO | hjorth_mobility | session2 to session3 | 48/60 | 0.668 | -0.116 | 0.397 |
| EC | alpha_theta_ratio | session2 to session3 | 50/60 | 0.608 | -0.907 | 0.623 |
| EO | alpha_theta_ratio | session2 to session3 | 48/60 | 0.291 | -0.138 | 0.149 |
| EC | permutation_entropy | session2 to session3 | 50/60 | 0.687 | 0.181 | 0.404 |
| EO | permutation_entropy | session2 to session3 | 48/60 | 0.583 | -0.126 | 0.328 |
| EC | alpha_centroid_hz | session2 to session3 | 50/60 | 0.762 | -0.185 | 0.497 |
| EO | alpha_centroid_hz | session2 to session3 | 48/60 | 0.924 | 0.015 | 0.230 |
| EC | alpha_envelope_cv | session2 to session3 | 44/60 | 0.727 | 0.247 | 0.483 |
| EO | alpha_envelope_cv | session2 to session3 | 46/60 | 0.710 | 0.083 | 0.263 |
| EO_minus_EC | delta_theta_level | session1 to session2 | 40/60 | 0.417 | -0.314 | 0.389 |
| EO_minus_EC | alpha_level | session1 to session2 | 40/60 | 0.695 | -0.156 | 0.233 |
| EO_minus_EC | beta_alpha_ratio | session1 to session2 | 40/60 | 0.756 | 0.059 | 0.232 |
| EO_minus_EC | gamma_level | session1 to session2 | 40/60 | 0.341 | -0.195 | 0.443 |
| EO_minus_EC | hjorth_mobility | session1 to session2 | 40/60 | 0.508 | 0.243 | 0.486 |
| EO_minus_EC | alpha_theta_ratio | session1 to session2 | 40/60 | 0.660 | -0.421 | 0.584 |
| EO_minus_EC | permutation_entropy | session1 to session2 | 40/60 | 0.651 | 0.226 | 0.355 |
| EO_minus_EC | alpha_centroid_hz | session1 to session2 | 40/60 | 0.814 | -0.177 | 0.307 |
| EO_minus_EC | alpha_envelope_cv | session1 to session2 | 36/60 | 0.864 | -0.050 | 0.357 |
| EO_minus_EC | delta_theta_level | session1 to session3 | 39/60 | 0.273 | -0.167 | 0.377 |
| EO_minus_EC | alpha_level | session1 to session3 | 39/60 | 0.655 | 0.132 | 0.354 |
| EO_minus_EC | beta_alpha_ratio | session1 to session3 | 39/60 | 0.630 | -0.132 | 0.400 |
| EO_minus_EC | gamma_level | session1 to session3 | 39/60 | 0.249 | 0.058 | 0.543 |
| EO_minus_EC | hjorth_mobility | session1 to session3 | 39/60 | 0.272 | 0.137 | 0.478 |
| EO_minus_EC | alpha_theta_ratio | session1 to session3 | 39/60 | 0.666 | 0.646 | 0.853 |
| EO_minus_EC | permutation_entropy | session1 to session3 | 39/60 | 0.465 | -0.067 | 0.596 |
| EO_minus_EC | alpha_centroid_hz | session1 to session3 | 39/60 | 0.530 | 0.040 | 0.431 |
| EO_minus_EC | alpha_envelope_cv | session1 to session3 | 36/60 | 0.793 | -0.212 | 0.603 |
| EO_minus_EC | delta_theta_level | session2 to session3 | 42/60 | 0.683 | 0.073 | 0.377 |
| EO_minus_EC | alpha_level | session2 to session3 | 42/60 | 0.630 | 0.243 | 0.262 |
| EO_minus_EC | beta_alpha_ratio | session2 to session3 | 42/60 | 0.672 | -0.177 | 0.344 |
| EO_minus_EC | gamma_level | session2 to session3 | 42/60 | 0.517 | 0.262 | 0.444 |
| EO_minus_EC | hjorth_mobility | session2 to session3 | 42/60 | 0.662 | -0.065 | 0.406 |
| EO_minus_EC | alpha_theta_ratio | session2 to session3 | 42/60 | 0.547 | 0.799 | 0.616 |
| EO_minus_EC | permutation_entropy | session2 to session3 | 42/60 | 0.588 | -0.292 | 0.463 |
| EO_minus_EC | alpha_centroid_hz | session2 to session3 | 42/60 | 0.527 | 0.232 | 0.455 |
| EO_minus_EC | alpha_envelope_cv | session2 to session3 | 37/60 | 0.813 | -0.235 | 0.367 |

All anchored coordinates are dimensionless (including `alpha_centroid_hz`). EO−EC is the within-participant/session difference between the EC and EO coordinate medians using the same anchor. The comparison is descriptive and does not use p-values.

## Interpretation and scope

This completes the frozen-anchor session-repeatability analysis for the selected 60-participant ds004148 cohort. It estimates repeatability of this measurement procedure in these recordings; it does not establish biological stability, artifact absence, cross-dataset equivalence or clinical qualification. QC eligibility is an operational screen, not evidence that a window is artifact-free.

The analysis uses only MNPS 3.4 coordinate values and their declared support. The run disabled v3.1 experimental measurements and the v3.4 candidate Jacobian. Standard summarize still emitted its primary 3D and 9D Jacobian datasets; they are not analyzed or used in these summaries. No anchor refitting, threshold search or inferential p-values were used.

A separate one-participant summarize smoke confirmed the legacy Jacobian identities and verified the v3.4-only suppression configuration. With both legacy switches and the v3.4 Jacobian candidate disabled, all four sub-01 session-2/session-3 EC/EO files retained their `(74, 9)` coordinate arrays and support masks (660/666 entries valid per recording), exactly matching the unsuppressed full-cohort values and masks. The verifier checked absence of all six legacy primary/layer paths and `/measurements/v3_4/jacobian`. The historical full-cohort files above were preserved unchanged. See `reproduction/config/config_ingest_ds004148_v34_legacy_jacobian_suppression_smoke.yml`, `artifacts/jacobian_suppression_smoke.json`, and `reproduction/scripts/verify_ds004148_v34_jacobian_suppression_smoke.py` for the config, verification manifest, and exact checks.

The frozen method is documented in `mndm/config/v.3.4/DS004148_SESSION_REPEATABILITY_V1.md`. The reproducibility package contains the selection, QC and full-cohort analysis manifests, scripts, and summary tables. EEG source files and generated H5 files remain in the local output tree; source EEG binaries were not hashed.
