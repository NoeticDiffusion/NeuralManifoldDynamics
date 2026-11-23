# ds004148 MNPS 3.4 session-repeatability pilot

**Descriptive three-participant pilot only.** One frozen ds005385 EyesClosed anchor was used throughout; no target-session refit. QC v1.1 and the support floor (at least 30 eligible windows and 120 unique seconds) were fixed before examining sessions 2-3 coordinates.

## Processing and support

- Reused session-1 H5/QC for sub-01-sub-03. Re-extracted sessions 2-3 from source with current MNPS 3.4 code; the old cache was refused by the feature-implementation identity gate and was not used.
- Checked 18 H5s (3 participants x 3 sessions x EC/EO): same anchor, consistent feature implementation, expected raw units and identical 8 s windows/4 s steps. QC matched all 1332 H5 windows.
- 153/162 participant x session x condition x coordinate summaries met the support floor. Complete 9D support met it in 17/18 session-condition rows. Per-coordinate counts and support are retained in the CSV outputs.
- The support loss is concentrated in: sub-02 session1 EC: 6 eligible windows, 40 unique seconds. This removes that participant from every EC level comparison involving session 1 and from session 1 EO-EC repeatability, leaving n=2 for those rows.
- Two participants (sub-02 and sub-03) have case-only channel-label change CPz/Cpz in session 3. Their ordered channel positions, normalized channel set, source surface and reference match. The v3.4 scalar estimators aggregate over channels and do not depend on channel position; the label difference is retained as QC metadata, and these comparisons are included.
- Included participant counts across the 81 repeatability rows range from 2 to 3 of 3. ICC(A,1), agreement differences and correlations are highly unstable at this sample size; they are descriptive pilot values, not dataset-level reliability estimates.

## Session comparisons

Bias is later session minus earlier session. Median is the median absolute difference. Full sample SD, 95% Bland-Altman limits and Spearman rank correlation are in artifacts/repeatability_summary.csv; Spearman is a rank-order description, not agreement.

| Measure | Coordinate | Sessions | n | ICC(A,1) | Bias | Median |
|---|---|---|---:|---:|---:|---:|
| EC | delta_theta_level | session1 to session2 | 2/3 | 0.926 | 0.418 | 0.418 |
| EO | delta_theta_level | session1 to session2 | 3/3 | 0.899 | 0.123 | 0.487 |
| EC | alpha_level | session1 to session2 | 2/3 | 0.975 | 0.086 | 0.099 |
| EO | alpha_level | session1 to session2 | 3/3 | 0.962 | 0.208 | 0.152 |
| EC | beta_alpha_ratio | session1 to session2 | 2/3 | 0.844 | 0.089 | 0.098 |
| EO | beta_alpha_ratio | session1 to session2 | 3/3 | 0.964 | 0.036 | 0.202 |
| EC | gamma_level | session1 to session2 | 2/3 | 0.951 | 0.210 | 0.210 |
| EO | gamma_level | session1 to session2 | 3/3 | 0.924 | 0.067 | 0.645 |
| EC | hjorth_mobility | session1 to session2 | 2/3 | 0.772 | -0.420 | 0.575 |
| EO | hjorth_mobility | session1 to session2 | 3/3 | 0.887 | -0.440 | 0.481 |
| EC | alpha_theta_ratio | session1 to session2 | 2/3 | 0.939 | -0.160 | 0.160 |
| EO | alpha_theta_ratio | session1 to session2 | 3/3 | 0.915 | 0.737 | 0.085 |
| EC | permutation_entropy | session1 to session2 | 2/3 | 0.983 | -0.001 | 0.094 |
| EO | permutation_entropy | session1 to session2 | 3/3 | 0.929 | -0.199 | 0.216 |
| EC | alpha_centroid_hz | session1 to session2 | 2/3 | 0.996 | 0.128 | 0.128 |
| EO | alpha_centroid_hz | session1 to session2 | 3/3 | 0.987 | -0.155 | 0.109 |
| EC | alpha_envelope_cv | session1 to session2 | 2/3 | 0.742 | 0.082 | 0.228 |
| EO | alpha_envelope_cv | session1 to session2 | 3/3 | 0.458 | -0.391 | 0.103 |
| EC | delta_theta_level | session1 to session3 | 2/3 | 0.957 | 0.230 | 0.230 |
| EO | delta_theta_level | session1 to session3 | 3/3 | 0.715 | -0.394 | 0.376 |
| EC | alpha_level | session1 to session3 | 2/3 | 0.898 | -0.261 | 0.261 |
| EO | alpha_level | session1 to session3 | 3/3 | 0.722 | -0.489 | 0.222 |
| EC | beta_alpha_ratio | session1 to session3 | 2/3 | 0.506 | 0.312 | 0.312 |
| EO | beta_alpha_ratio | session1 to session3 | 3/3 | 0.418 | 0.669 | 1.031 |
| EC | gamma_level | session1 to session3 | 2/3 | 0.879 | -0.068 | 0.300 |
| EO | gamma_level | session1 to session3 | 3/3 | 0.649 | -0.472 | 0.823 |
| EC | hjorth_mobility | session1 to session3 | 2/3 | 0.993 | -0.045 | 0.060 |
| EO | hjorth_mobility | session1 to session3 | 3/3 | 0.706 | 0.625 | 0.620 |
| EC | alpha_theta_ratio | session1 to session3 | 2/3 | -0.108 | -0.561 | 0.561 |
| EO | alpha_theta_ratio | session1 to session3 | 3/3 | 0.091 | -1.317 | 0.225 |
| EC | permutation_entropy | session1 to session3 | 2/3 | 0.775 | 0.335 | 0.335 |
| EO | permutation_entropy | session1 to session3 | 3/3 | 0.530 | 0.678 | 0.346 |
| EC | alpha_centroid_hz | session1 to session3 | 2/3 | 0.867 | -0.062 | 0.558 |
| EO | alpha_centroid_hz | session1 to session3 | 3/3 | 0.598 | -0.471 | 0.043 |
| EC | alpha_envelope_cv | session1 to session3 | 2/3 | 0.892 | -0.089 | 0.089 |
| EO | alpha_envelope_cv | session1 to session3 | 3/3 | 0.433 | -0.501 | 0.259 |
| EC | delta_theta_level | session2 to session3 | 3/3 | 0.718 | 0.541 | 0.670 |
| EO | delta_theta_level | session2 to session3 | 3/3 | 0.773 | -0.516 | 0.687 |
| EC | alpha_level | session2 to session3 | 3/3 | 0.616 | -0.754 | 0.385 |
| EO | alpha_level | session2 to session3 | 3/3 | 0.562 | -0.697 | 0.197 |
| EC | beta_alpha_ratio | session2 to session3 | 3/3 | 0.465 | 0.425 | 0.327 |
| EO | beta_alpha_ratio | session2 to session3 | 3/3 | 0.573 | 0.633 | 0.614 |
| EC | gamma_level | session2 to session3 | 3/3 | 0.904 | -0.478 | 0.462 |
| EO | gamma_level | session2 to session3 | 3/3 | 0.741 | -0.539 | 0.678 |
| EC | hjorth_mobility | session2 to session3 | 3/3 | 0.792 | -0.029 | 0.836 |
| EO | hjorth_mobility | session2 to session3 | 3/3 | 0.509 | 1.065 | 1.335 |
| EC | alpha_theta_ratio | session2 to session3 | 3/3 | -0.146 | -3.272 | 0.738 |
| EO | alpha_theta_ratio | session2 to session3 | 3/3 | 0.062 | -2.053 | 0.141 |
| EC | permutation_entropy | session2 to session3 | 3/3 | 0.401 | 1.051 | 0.439 |
| EO | permutation_entropy | session2 to session3 | 3/3 | 0.516 | 0.878 | 0.944 |
| EC | alpha_centroid_hz | session2 to session3 | 3/3 | 0.245 | -0.809 | 0.794 |
| EO | alpha_centroid_hz | session2 to session3 | 3/3 | 0.680 | -0.316 | 0.089 |
| EC | alpha_envelope_cv | session2 to session3 | 3/3 | -0.197 | 0.178 | 0.381 |
| EO | alpha_envelope_cv | session2 to session3 | 3/3 | 0.885 | -0.110 | 0.096 |
| EO_minus_EC | delta_theta_level | session1 to session2 | 2/3 | -1.947 | -0.473 | 0.846 |
| EO_minus_EC | alpha_level | session1 to session2 | 2/3 | 0.486 | -0.056 | 0.221 |
| EO_minus_EC | beta_alpha_ratio | session1 to session2 | 2/3 | -0.019 | 0.066 | 0.164 |
| EO_minus_EC | gamma_level | session1 to session2 | 2/3 | 0.770 | -0.233 | 0.551 |
| EO_minus_EC | hjorth_mobility | session1 to session2 | 2/3 | 0.336 | 0.118 | 0.397 |
| EO_minus_EC | alpha_theta_ratio | session1 to session2 | 2/3 | 0.917 | 0.159 | 0.159 |
| EO_minus_EC | permutation_entropy | session1 to session2 | 2/3 | 0.938 | 0.000 | 0.310 |
| EO_minus_EC | alpha_centroid_hz | session1 to session2 | 2/3 | 0.989 | -0.217 | 0.217 |
| EO_minus_EC | alpha_envelope_cv | session1 to session2 | 2/3 | 0.923 | -0.134 | 0.177 |
| EO_minus_EC | delta_theta_level | session1 to session3 | 2/3 | -0.240 | -0.684 | 0.684 |
| EO_minus_EC | alpha_level | session1 to session3 | 2/3 | 0.764 | 0.127 | 0.150 |
| EO_minus_EC | beta_alpha_ratio | session1 to session3 | 2/3 | -0.007 | 0.018 | 0.499 |
| EO_minus_EC | gamma_level | session1 to session3 | 2/3 | 0.391 | -0.229 | 0.772 |
| EO_minus_EC | hjorth_mobility | session1 to session3 | 2/3 | 0.226 | 0.673 | 0.673 |
| EO_minus_EC | alpha_theta_ratio | session1 to session3 | 2/3 | 0.230 | 0.566 | 0.566 |
| EO_minus_EC | permutation_entropy | session1 to session3 | 2/3 | 0.320 | 0.510 | 0.673 |
| EO_minus_EC | alpha_centroid_hz | session1 to session3 | 2/3 | 0.827 | 0.050 | 0.589 |
| EO_minus_EC | alpha_envelope_cv | session1 to session3 | 2/3 | 0.962 | -0.088 | 0.088 |
| EO_minus_EC | delta_theta_level | session2 to session3 | 3/3 | 0.157 | -1.058 | 0.405 |
| EO_minus_EC | alpha_level | session2 to session3 | 3/3 | -0.427 | 0.058 | 0.193 |
| EO_minus_EC | beta_alpha_ratio | session2 to session3 | 3/3 | 0.621 | 0.208 | 0.382 |
| EO_minus_EC | gamma_level | session2 to session3 | 3/3 | 0.928 | -0.061 | 0.217 |
| EO_minus_EC | hjorth_mobility | session2 to session3 | 3/3 | 0.014 | 1.094 | 0.601 |
| EO_minus_EC | alpha_theta_ratio | session2 to session3 | 3/3 | -0.420 | 1.219 | 0.607 |
| EO_minus_EC | permutation_entropy | session2 to session3 | 3/3 | 0.161 | -0.173 | 0.872 |
| EO_minus_EC | alpha_centroid_hz | session2 to session3 | 3/3 | 0.798 | 0.492 | 0.883 |
| EO_minus_EC | alpha_envelope_cv | session2 to session3 | 3/3 | 0.447 | -0.289 | 0.286 |

All anchored coordinates are dimensionless (including alpha_centroid_hz). EO-EC is the within-participant/session difference between condition medians using the same anchor.

## Interpretation and scope

This completes the bounded repeatability workflow technically and preserves both condition levels and EO-EC changes for the three preselected pilot participants. It does not establish population-level repeatability, biological stability, artifact absence or clinical qualification. No p-values, Jacobian work, threshold exploration or anchor refitting were performed. The full 60-participant sessions 1-3 cohort was not processed; the full feature run was stopped after runtime extrapolation.

The frozen method is documented in mndm/config/v.3.4/DS004148_SESSION_REPEATABILITY_V1.md. The analysis implementation is project/mnps2_audit/analyze_ds004148_session_repeatability_pilot.py and reuses schema/anchor checks from project/mnps2_audit/summarize_ds004148_baseline.py. H5 and raw EEG remain in local pilot outputs; raw EEG binaries were not hashed.
