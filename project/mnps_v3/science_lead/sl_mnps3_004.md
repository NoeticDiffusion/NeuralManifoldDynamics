> **SL-005 supersession (2026-09-02):** M1–M2 **PASS / RATIFY** below still stands.
> The NEXT GATE line is **withdrawn**. M3 is **not authorized in ingest**.
> Cross-fit identification belongs in nmd-analysis
> (`project/mnps_v3/df_drift_c1_analysis_handoff.md`).

DF-DRIFT-C1 M1–M2

VERDICT:
PASS / RATIFY

Evidence class:
INTERNAL VALIDATED RESULT

Established:
- alignment drift and covariance residualization are now separate operations;
- C1 leaves current a_hat and derived diffusion summaries unchanged;
- truth-known b can recover the intended A_bD ordering;
- forbidden pseudo-drifts fail closed;
- C2 cannot currently be invoked through either ingest or direct estimator use;
- common profiles and OD-EPI-001 remain unchanged.

Not established:
- empirical chart-space b;
- Itô drift b(X);
- cross-fit identifiability;
- empirical NDT drift–diffusion alignment;
- residualized process diffusion;
- C2;
- R_b/a as a genuine drift/diffusion ratio.

NEXT GATE (historical; superseded by SL-005 — do not execute in ingest):
Authorize M3 DESIGN ONLY:
crossfit_local_chart_b identification and contamination audit.

M3 EXECUTION (historical; superseded by SL-005):
NOT YET AUTHORIZED pending frozen protocol.

SL-005 replacement next gate: analysis-repo M3 design in nmd-analysis.
Ingest does not design or execute M3.