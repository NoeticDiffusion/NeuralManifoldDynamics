MNPS v3.2: VECTOR-PILOT-001 execution
====================================

Result
------

**Engineering PASS for the authorized one-recording integration pilot.** ED,
OI, and NIE were computed from one audited ANPHY EPCTL01 E1-G2 grid. All 14
supported windows produced valid values for all three families. The nominal
``[0,20)`` window remains present with null/unsupported values for all
families because its effective source dependency begins before the recording.
Every result is ``qualified=false``. This is not scientific validation of the
v3.2 vector.

The frozen protocol is :doc:`mnps_v3_2_vector_pilot_001_protocol`. The run
contains 15 window JSON records, a 45-row long CSV, 14 saved NIE edge matrices,
pairwise descriptive associations, 64 paired synthetic-control replicates,
and a hash-bearing run manifest.

Execution identity and checks
-----------------------------

* Run ID: ``run_20261012T_sl005_vectorpilot001``.
* Local run root:
  ``outputs/VECTOR-PILOT-001/run_20261012T_sl005_vectorpilot001/``.
* Runtime: Python 3.12.0, NumPy 2.1.1, SciPy 1.14.1, MNE 1.6.1, mndm 3.1.0.
* Protocol SHA-256:
  ``c25583f31e4ca7bfe13e92d6af4230513ec2191f6fe19104054e7e467c821e99``.
* Runner SHA-256:
  ``c04dd4d465ca5ce6fa12523879f9ca159f74e3c1fd19445f674b900578fdafa1``.
* The tracked overlay is a byte-identical copy of the previously audited E1-G2
  grid and retains its SHA-256
  ``3f1d3135d45ad43648ef1f1c24b0732970ad1492209a414804c35c6eed240824``.
  The EDF SHA-256 is
  ``a16ddc068edcc4ce2ff8c499cca51b9a890b76b7ddb569f9a96978013f8b5e8c`` and
  the positions SHA-256 is
  ``b5bbc85ef1e840e0ca703c62c22a1000dd3252f32210e251b8d0e81123fa1bd2``.
* All 15 window JSON hashes, the long CSV hash, association and control JSON
  hashes, and all 14 edge-matrix hashes matched the run manifest. Across each
  supported window, ED, OI, and NIE wrapper records name the same input-matrix
  hash. The raw matrix kernels retain ``support_status=unknown``; the outer
  records mark support ``known`` only after the loader audit.
* OI values exactly match all 14 values from OI-REALIZATION-001.
* Focused ED/OI/NIE and VECTOR-PILOT-001 suite: **51 passed**.

The first window is ``[0,20)`` with source dependency ``[-3376,19997)`` and
reason ``insufficient_effective_filter_edge_support``. The 14 supported
windows retain their individual source dependencies in their JSON records and
CSV rows. Adjacent effective dependencies overlap, so these windows are not
treated as independent observations.

Measurement summary
-------------------

.. list-table:: Valid values across the 14 common supported windows
   :header-rows: 1
   :widths: 15 15 18 18 18

   * - Family
     - Realization ID
     - Mean
     - Min
     - Max
   * - ED
     - ``ED.anphy_epctl01_E1_G2_fixed83_avgref.v1``
     - 4.41607
     - 2.94598
     - 5.37489
   * - OI
     - ``OI.anphy_epctl01_G2_fixed83_PE5_delay1_median.v1``
     - 0.61557
     - 0.58002
     - 0.64230
   * - NIE
     - ``NIE.anphy_epctl01_E1_G2_fixed83_avgref_abs_pearson_no_threshold.v1``
     - 0.45700
     - 0.41565
     - 0.51769

NIE used the unchanged D1 weighted-global-efficiency kernel and its frozen
absolute-Pearson/no-threshold edge definition. Each of the 14 empirical
edge matrices was exactly symmetric, had an exact-zero diagonal, passed the
``[0,1]`` bound, and was complete (all 3,403 undirected weights were positive).
This is an expected consequence of using unthresholded absolute sample
correlations; graph density alone does not identify neural integration.

Exploratory pairwise Pearson associations
-----------------------------------------

All pairs use the same 14 valid windows. The run reports coefficients only;
there are no p-values or confidence claims.

.. list-table:: Descriptive associations within this recording
   :header-rows: 1
   :widths: 18 14 55

   * - Pair
     - ``n``
     - Pearson ``r``
   * - ED / OI
     - 14
     - +0.90242
   * - ED / NIE
     - 14
     - -0.98348
   * - OI / NIE
     - 14
     - -0.91400

These strong within-recording associations make the candidate families worth
examining for empirical duplication in a prospectively selected sample. They
do not establish orthogonality or redundancy: this is one recording, adjacent
effective dependencies overlap, and the result has not been tested across
recordings.

Synthetic negative and reference controls
-----------------------------------------

The frozen control batch used 64 paired realizations of 83 independent
stationary Gaussian AR(1) channels (``rho=0.95``), each with 5,000 samples at
250 Hz. The same three estimators ran before and after instantaneous average
reference over the 83 synthetic channels. Controls used the target-grid
signals directly; they did not emulate the EDF resampling or filtering and
are not an EEG model or empirical null distribution.

.. list-table:: Control means and paired reference-minus-unreferenced means
   :header-rows: 1
   :widths: 13 12 22 22 31

   * - Family
     - Replicates
     - Unreferenced mean
     - Referenced mean
     - Paired change mean
   * - ED
     - 64
     - 62.52126
     - 61.93911
     - -0.58215
   * - OI
     - 64
     - 0.92110
     - 0.92108
     - -0.00002
   * - NIE
     - 64
     - 0.07109
     - 0.07249
     - +0.00139

The control batch confirms that finite-window absolute-Pearson NIE is nonzero
for independent colored sources and that average reference changes its value.
No control baseline was subtracted from empirical values, and no significance
test or qualification decision was made. Because the synthetic inputs do not
reproduce the full ANPHY signal branch, the empirical-control level difference
is not interpreted as a biological effect.

Limits and next step
--------------------

This run establishes a reproducible engineering handoff for three families
on one 83-channel, 250 Hz, 20-second realization. It does not establish
population-level behavior, test-retest reliability, cross-dataset equivalence,
incremental predictive value, a neural mechanism, or a three-dimensional
manifold. Dataset release/session/task/run metadata remain unresolved. No
pipeline/export integration or cohort run was performed, and SL-006 was not
launched.

PCV and AR work remains strategically relevant; this result does not justify
another ED formula round. See :doc:`mnps_v3_2_vector_ablation_readiness_001`
for the updated candidate-readiness status.
