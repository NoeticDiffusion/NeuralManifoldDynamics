# MNPS v3_2 experimental modules

Status: partial implementation, 2026-10-12. The v3.2 tree contains standalone
matrix-level estimator prototypes; they are not integrated into the parent
namespace, feature pipeline, candidate grid, or export schema. Check the API
reference pages for each implemented estimator and its validation record.
The ED, OI and NIE result types share the
`nmdm.mnps_v3_2.measurement_record.v1` envelope documented in
[`docs/api/mnps_v3_2_contracts.rst`](../../../../../docs/api/mnps_v3_2_contracts.rst).

Import the experimental ED, OI and NIE kernels directly from
`mndm.mnps.v3_2.ed`, `mndm.mnps.v3_2.oi` and `mndm.mnps.v3_2.nie`; none is
re-exported by the v3_2 parent namespace. Their results keep temporal/source
support unknown and scientific qualification false. Definitions and test
records are documented in [`docs/api/mnps_v3_2_ed.rst`](../../../../../docs/api/mnps_v3_2_ed.rst),
[`docs/api/mnps_v3_2_oi.rst`](../../../../../docs/api/mnps_v3_2_oi.rst) and
[`docs/api/mnps_v3_2_nie.rst`](../../../../../docs/api/mnps_v3_2_nie.rst). The
NIE edge-construction baseline is specified in
[`docs/api/mnps_v3_2_nie_edge_contract_001.rst`](../../../../../docs/api/mnps_v3_2_nie_edge_contract_001.rst);
it authorizes no empirical graph run. Candidate comparison readiness is
summarized in
[`docs/api/mnps_v3_2_vector_ablation_readiness_001.rst`](../../../../../docs/api/mnps_v3_2_vector_ablation_readiness_001.rst).
A separate ED/N diagnostic and isotropic-shrinkage challenger are documented in
[`docs/api/mnps_v3_2_ed_challengers.rst`](../../../../../docs/api/mnps_v3_2_ed_challengers.rst).
The shared measurement envelope is described in
[`docs/api/mnps_v3_2_contracts.rst`](../../../../../docs/api/mnps_v3_2_contracts.rst).
The bounded EEG stability protocol is in
[`docs/api/mnps_v3_2_ed_reliability_001.rst`](../../../../../docs/api/mnps_v3_2_ed_reliability_001.rst)
and its verified-source preflight is complete. Accepted window extraction and
simulations completed as an unqualified engineering experiment on 2026-10-12;
the run record and limits are in the API page. The complete implementation plan
and working protocol copies remain under `project/new_mnps_v.3.2/`.

The bounded within/between covariance diagnostic on the same recording is
documented in
[`docs/api/mnps_v3_2_ed_duration_mechanism_001.rst`](../../../../../docs/api/mnps_v3_2_ed_duration_mechanism_001.rst).
It supports an across-block mean covariance mechanism in most windows, with a
clear Native64 exception; its cause and scientific use remain unresolved.

SL-004 completed three additional bounded challenges. The synthetic NIE
reference-null evidence is in
[`docs/api/mnps_v3_2_nie_reference_null_001.rst`](../../../../../docs/api/mnps_v3_2_nie_reference_null_001.rst);
the single-recording ED rereferencing sensitivity is in
[`docs/api/mnps_v3_2_ed_reference_sensitivity_001.rst`](../../../../../docs/api/mnps_v3_2_ed_reference_sensitivity_001.rst);
and the audited fixed-83, 20-second OI EEG realization is in
[`docs/api/mnps_v3_2_oi_realization_001.rst`](../../../../../docs/api/mnps_v3_2_oi_realization_001.rst).
The execution record and remaining limits are summarized in
[`docs/api/mnps_v3_2_sl_004_execution.rst`](../../../../../docs/api/mnps_v3_2_sl_004_execution.rst).
The three pilots use different source evidence; no support-matched ED--OI--NIE
comparison or vector integration has been performed.

This implementation does not change v3.1, package version, historical exports,
the canonical MNPS chart, or active configurations.
