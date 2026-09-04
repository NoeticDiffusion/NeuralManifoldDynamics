# DF-DRIFT-C1 M1–M2 return packet

Authority: `project/mnps_v3/science_lead/sl_mnps3_003.md` (2026-09-02).
This packet stops **before** M3. SL-005 froze the ingest line here
and moved M3 to nmd-analysis.

```text
VERDICT:
PASS

Contract impact:
Implementation-preserving on the ingest write path (still drift=None,
residualize_increments=False). Measurement-semantic only as a library
C1 API: an independently supplied chart-space b may compute A_bD
without changing a_hat. C2 residualization remains closed. MNPS
[m,d,e] / 9D, J_hat, family schema IDs, and ds004100 OD-EPI-001 are
unchanged. FAR-EXT-002E is parked BLOCKED_EXTERNAL_CUSTODY.

Validation:
Gate C1-A: spatially varying truth-known b leaves a_hat, D_total,
d_diff, c_diff, and the valid mask byte-equal (NaN-equal) to the
no-drift path. Truth-known A_bD recovers anisotropy ranking under
residualize_increments=False. Forbidden sources (mnps_xdot, Jacobian
intercept, local increment mean, Jacobian residual, committor
potential gradient) return no field and do not residualize. YAML
source mnps_xdot on ingest still writes independent-drift-not-testable
alignment with a_hat on the no-drift path. Common EEG/fMRI/ephys
profiles still do not enable families. residualize_increments=True
is invalid / c2_residualize_increments_not_authorized, with or without
a supplied field.

Safe claim:
The estimator can align a truth-known chart-space velocity with raw
increment covariance (a_semantics=raw_increment_covariance;
ratio_semantics=chart_velocity_to_increment_spread) without changing
the existing a_hat contract. Ingest does not supply a vector.

Not established:
Itô drift b(X). Residualized diffusion a. Empirical A_bD as NDT
alignment. Cross-fit identification of b (M3). C2. Any biological
claim. R_b/a as a genuine drift-to-diffusion ratio (secondary under
C1 because the denominator is still raw increment covariance).

Next gate:
Handoff RATIFIED (2026-09-03). Type C ingest implementation frozen.
`project/mnps_v3/df_drift_c1_analysis_handoff.md`.
M3 is nmd-analysis, not ingest. Do not modify ds004100 OD-EPI-001.
Do not reopen FAR-EXT-002E unless complete E1+E2 XML arrives.
```
