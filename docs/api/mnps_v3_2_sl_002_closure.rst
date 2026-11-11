MNPS v3.2: SL-002 engineering closure
=====================================

Decision
--------

SL-002 is closed as an engineering milestone following the supplied
science-lead review in ``project/new_mnps_v.3.2/science_lead/sl_003.md``.
Its approved work established reproducible ED and NIE matrix-level kernels,
completed the bounded ED reliability experiment, completed the synthetic NIE
topology control, audited the PCV/OI/AR contracts, and added the shared v3.2
measurement record.

Evidence
--------

* ED raw: deterministic numerical controls and v3.1 parity are recorded in
  :doc:`mnps_v3_2_ed`.
* ED reliability: one outcome-blind ``sub-206`` recording was fully audited
  and the frozen run completed 82 windows, 3,936 real-data measurements, and
  all 48 AR(1) simulation cells. Results and hashes are in
  :doc:`mnps_v3_2_ed_reliability_001`.
* NIE topology: exact enumeration covered 5,040 labeled assignments with a
  fixed edge-weight multiset; results are in
  :doc:`mnps_v3_2_nie_topology_001`.
* Shared result envelope: ``nmdm.mnps_v3_2.measurement_record.v1`` is
  documented in :doc:`mnps_v3_2_contracts`.
* PCV/OI/AR: the audit identifies the remaining end-to-end realization
  decisions in :doc:`mnps_v3_2_vector_contract_002`.

The focused suite combining the ED reliability runner, ED/N and shrinkage
controls, shared contracts, NIE topology and v3.1 ED parity passed **67 tests**
on 2026-10-12. This is engineering evidence only. ED and NIE remain
``qualified=false``; no empirical measurement is scientifically qualified by
this closure.

History note
------------

The file ``project/new_mnps_v.3.2/science_lead/sl_002.md`` is headed
``SL-001 — Technical PASS`` and authorizes the proposed SL-002 work. The
subsequent science-lead review in ``sl_003.md`` explicitly closes SL-002 and
authorizes SL-003. This tracked record makes that sequence explicit without
rewriting the review source.

SL-003 work is tracked separately: bounded ED duration-mechanism analysis,
NIE EEG edge-contract preparation, continued OI/PCV/AR contract work, and
vector-ablation readiness assessment. No pipeline/export integration, package
release, canonical chart change or qualification decision is included here.
