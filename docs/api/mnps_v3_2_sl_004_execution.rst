MNPS v3.2 SL-004 execution record
================================

Status
------

All three bounded SL-004 experiments completed on 2026-10-12. They address
measurement-method risks and one end-to-end OI handoff. All outputs remain
experimental and ``qualified=false``. The frozen ED v1, NIE edge/kernel
definitions, OI matrix kernel, canonical MNPS chart, and pipeline/export
contracts were not changed.

Experiments and results
-----------------------

.. list-table:: SL-004 experiment status
   :header-rows: 1
   :widths: 22 26 32 20

   * - Experiment
     - Scope and run
     - Result
     - Validation
   * - ``NIE-REFERENCE-NULL-001``
     - Synthetic independent stationary Gaussian AR(1) sources; 64 seeds; N=10, 19, 64; average reference versus unreferenced; 768 window rows. Run: ``outputs/NIE-REFERENCE-NULL-001/run_20261012T142730Z_sl004``.
     - The analytic population effect is reproduced: average reference gives equal off-diagonal correlation ``-1/(N-1)`` and absolute Pearson yields complete-graph NIE ``1/(N-1)``. Finite-window means differ because sample variability and shortest-path structure remain. No empirical EEG graph was computed.
     - All 9 analytic graph controls passed. Focused NIE suite: 27 tests passed. Results: :doc:`mnps_v3_2_nie_reference_null_001`.
   * - ``ED-REFERENCE-SENSITIVITY-001``
     - Same audited ``sub-206`` EyesClosed recording; average reference applied across all 64 source EEG channels before Native64/Core19/Custom10 selection; eight 20 s windows under two reference conditions. Run: ``outputs/ED-REFERENCE-SENSITIVITY-001/run_20261012T142500Z_sl004``.
     - Native64 common-mode alignment fell to numerical zero under full-surface average reference by construction. Between-block trace fraction, leading-axis contribution, and total/between direction alignment remained similar; paired ED shifts varied by realization and window. This establishes reference sensitivity in one recording, not its cause.
     - 48 measurement rows and 240 block rows; covariance reconstruction, axis partition, ED parity, and hash checks passed. Focused suite: 8 tests passed. Results: :doc:`mnps_v3_2_ed_reference_sensitivity_001`.
   * - ``OI-REALIZATION-001``
     - Replayed the existing ANPHY ``EPCTL01`` fixed-83 E1-G2 branch at 20 s / 250 Hz; one recording, 15 nominal windows. Run: ``outputs/OI-REALIZATION-001/run_20261012T_run2_archivedenv_sl004``.
     - Fourteen windows were audited and computed; ``[0,20)`` remains unsupported because its effective filter dependency begins before the recording boundary. Four ordinal controls behaved as specified. The first system-Python preflight stopped on a runtime coefficient-hash mismatch; rerunning with the archived project virtual environment reproduced both archived coefficient hashes exactly.
     - 15 sidecars and manifest hashes verified; all results unqualified. Focused OI/handoff suite: 14 tests passed. Results: :doc:`mnps_v3_2_oi_realization_001`.

Interpretation limits
---------------------

The NIE result establishes a consequence of the frozen average-reference plus
absolute-Pearson construction under an independent-source population model; it
does not establish that EEG NIE measures neural integration. The ED result
shows reference-dependent values and common-mode behavior for one recording;
it does not distinguish neural activity from artifact or establish a general
ED interpretation. The OI result establishes a reproducible, support-audited
handoff for one ANPHY recording at 20 seconds; it does not establish shorter
windows, other recordings, sensor/reference equivalence, or scientific
qualification.

No common ED--OI--NIE comparison was run. The ED run uses OpenNeuro
``ds005385`` / ``sub-206``; the OI run uses ANPHY ``EPCTL01``; and the NIE run
uses synthetic sources. A later comparison needs a selected common recording
and window grid, a separately frozen empirical NIE edge realization, and
pairwise support/realization accounting. No pipeline/export or vector-chart
integration is implied by SL-004.

Related contracts and readiness
-------------------------------

* :doc:`mnps_v3_2_nie_reference_null_001`
* :doc:`mnps_v3_2_ed_reference_sensitivity_001`
* :doc:`mnps_v3_2_oi_realization_001`
* :doc:`mnps_v3_2_vector_ablation_readiness_001`
