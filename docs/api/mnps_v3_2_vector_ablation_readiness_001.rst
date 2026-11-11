MNPS v3.2: VECTOR-ABLATION-READINESS-001
========================================

Status
------

**Multi-recording feasibility has engineering PASS; scientific vector ablation
and qualification remain on HOLD.** SL-005 established the initial common
ED-OI-NIE handoff on EPCTL01. Authorized SL-006 then computed the unchanged
families on 168 supported windows from twelve outcome-blind ANPHY-Sleep
participants on one audited fixed-83 realization. The bounded cohort extends
cross-recording feasibility evidence, but it does not establish test-retest
reliability, prediction, biological usefulness, or a qualified vector. This
assessment separates computability, comparability, and qualification.

The completed SL-004 experiments are documented in
:doc:`mnps_v3_2_nie_reference_null_001`,
:doc:`mnps_v3_2_ed_reference_sensitivity_001`, and
:doc:`mnps_v3_2_oi_realization_001`; their combined status and limits are in
:doc:`mnps_v3_2_sl_004_execution`. The common pilot results are in
:doc:`mnps_v3_2_vector_pilot_001` and its frozen procedure is in
:doc:`mnps_v3_2_vector_pilot_001_protocol`. The bounded cohort procedure and
results are in :doc:`mnps_v3_2_vector_feasibility_002_protocol` and
:doc:`mnps_v3_2_vector_feasibility_002`.

Readiness matrix
----------------

.. list-table:: Candidate status at completion of SL-004
   :header-rows: 1
   :widths: 14 22 32 32

   * - Candidate
     - Kernel / evidence
     - Remaining comparison dependency
     - Readiness
   * - ED raw
     - Implemented matrix kernel; one audited sub-206 recording with 4/8/12/20 s windows and native64/Core19/custom10 realizations. ``ED-DURATION-MECHANISM-001`` and ``ED-REFERENCE-SENSITIVITY-001`` provide bounded covariance and reference diagnostics.
     - The full-64 average-reference condition removes the uniform common mode by construction, while the between-block trace fraction and leading-axis alignment remain similar in this recording. Paired ED shifts vary by realization and window; the result does not identify the mechanism's cause.
     - Engineering diagnostics are complete for the frozen question. ED remains unqualified and is not support-matched to the OI recording.
   * - ED/N
     - Implemented as raw ED divided by channel count; exact row-wise parity verified.
     - It is a diagnostic derived from ED and does not remove montage, reference, rank or support differences.
     - Do not count as an independent vector coordinate.
   * - ED shrinkage ``lambda=0.5``
     - Implemented as a separately identified sensitivity estimator and included in the bounded run.
     - Its value moves toward the isotropic control; no evidence establishes it as more valid for real EEG.
     - Method challenger only; not preferred or qualified.
   * - ED shrinkage ``lambda=1``
     - Algebraic ``ED=N`` control verified within the frozen tolerance.
     - It is deliberately isotropic and does not measure an empirical neural spectrum.
     - Implementation control only; exclude from candidate ablations.
   * - NIE
     - Weighted global-efficiency matrix kernel passed graph controls; ``NIE-TOPOLOGY-001`` held the edge multiset fixed. ``NIE-REFERENCE-NULL-001`` validated the synthetic input-to-matrix path for independent colored sources at N=10/19/64.
     - Average reference followed by absolute Pearson produced the expected analytic population graph consequence. Finite-window NIE also depends on sample variability and shortest paths. The run used synthetic data only; empirical graph construction remains unvalidated.
     - Reference-null and kernel controls passed. This remains a reference-sensitive experimental graph statistic; no empirical EEG comparison or qualification was performed.
   * - OI
     - The standalone order-5/delay-1 normalized permutation-entropy matrix kernel is implemented with stable tie handling and the shared result envelope. ``OI-REALIZATION-001`` audited one fixed-83 ANPHY EPCTL01 20 s, 250 Hz realization.
     - Fourteen of 15 nominal windows passed the source-support audit; ``[0,20)`` is retained as unsupported because the effective filter dependency reaches before recording start. The source/session metadata are incomplete, and this is a different recording from the ED pilot.
     - One experimental end-to-end EEG realization is established. It does not establish shorter-window support, comparison readiness, or qualification.
   * - ED--OI--NIE collection
     - SL-005 established the common EPCTL01 handoff; SL-006 then computed all three unchanged families on 168 support-audited windows from twelve participants. The shared time-index control and raw covariance ED identity passed.
     - Twelve participants remain a bounded descriptive sample; adjacent effective window dependencies overlap; NIE uses a dense absolute-Pearson/no-threshold sensor graph; the ANPHY local archive revision is unknown and local EDF+C headers differ from the publication's EDF+D description.
     - Engineering feasibility passed. Keep experimental and unqualified. The participant-median correlations are descriptive and do not qualify a vector or authorize feature removal.
   * - PCV
     - The formula and two distinct v3.1 realizations are documented.
     - Phase/filter path, reference, channels, resampling and actual dependency support are not frozen for v3.2 windows.
     - Not ready.
   * - AR
     - The ordered adjacent-block ``1-corr`` vector is documented.
     - Activity versus connectivity input, feature construction, scaling, block partition and completeness policy remain open.
     - Not ready; preserve ordered components and do not scalarize implicitly.
   * - TP, RD, SC, spatial/TIR candidates
     - Existing v3.1 baselines or optional proposals are documented.
     - A named v3.2 estimator, realization and support contract is not yet established for this comparison.
     - Not ready as v3.2 candidates.

Minimum comparison contract
---------------------------

Before a later multi-family comparison, record for each measurement its
versioned estimator and realization ID, source and segment hashes, actual
ordered labels and reference metadata, sample grid, outer window bounds,
effective source dependency, preprocessing/edge construction, coverage, QC,
and computed/valid/qualified states. Join values on the same source recording
and window boundaries, while retaining each estimator's distinct dependency
and edge support. Report which windows are pairwise support-matched and which
are not. Do not discard otherwise computable windows, impute unsupported
values, or require one universal channel-count floor.

For vector candidates, freeze component names and order before comparison.
Keep AR pair values as an ordered vector; any scalar aggregation would be a
separate estimator. ED/N and lambda=1 are diagnostics/controls, not extra
independent coordinates. NIE empirical graphs cannot enter until their
upstream edge realization is frozen. Candidate-local failure must not suppress
other supported measurements.

Next action
-----------

SL-006 completed its prospectively selected bounded feasibility cohort. Review
its descriptive ED/OI/NIE associations and provenance caveats before deciding
whether a separately framed vector ablation is scientifically useful. Continue
the PCV and AR contract work independently. Do not treat this study as a
qualified vector, a common eight-coordinate chart, or a basis for a
Jacobian/manifold interpretation.

PCV and AR remain eligible for development, but their missing realization and
support decisions must be explicit before they enter a matched comparison. This
is a technical readiness assessment, not a scientific acceptance gate; passing
it does not qualify a candidate.
