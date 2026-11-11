MNPS v3.2 OI realization OI-REALIZATION-001
============================================

Status
------

One bounded, single-recording EEG pilot completed on 2026-10-12. This is an
experimental v3.2 realization and remains unqualified. It demonstrates an
audited handoff from the already documented ANPHY EPCTL01 E1-G2 20-second EEG
branch to the unchanged v3.2 OI matrix kernel. It establishes no biological,
condition, group, reliability, cross-recording, or shorter-window claim.

The run retained all 15 nominal windows from 0 through 300 seconds. Fourteen
windows were supported and computed; the first window remains explicitly
unsupported because its effective FIR dependency begins before the recording.
The output does not qualify any result.

Frozen identities and provenance
--------------------------------

* Protocol: ``OI-REALIZATION-001``; protocol SHA-256
  ``6b29ba3b059aa271c3aed6fbf1cc712f4151db5e9e43e50efe4fe40781b0159e``.
  This is the exact protocol file hash recorded by the run manifest.
* Realization ID: ``OI.anphy_epctl01_G2_fixed83_PE5_delay1_median.v1``.
* Matrix estimator: ``OI.matrix_input_channel_median_normalized_PE5_delay1.v1``.
* Matrix schema: ``nmdm.mnps_v3_2.oi_matrix_result.v1``.
* Realization result schema: ``mndm.mnps_v3_2.oi_realization_result.v1``.
* Run ID: ``run_20261012T_run2_archivedenv_sl004``.
* Input branch: the previously documented ANPHY EPCTL01 E1-G2 fixed-83 EEG
  source grid in
  ``project/new_mnps_v.3.1/anphy_epctl01_e1_g2_grid_20260929.yaml``.
* Source EDF: ``K:/ExternalReceivedDatasets/ANPHY/Subjects/EPCTL01/EPCTL01 - fixed.edf``;
  SHA-256 ``a16ddc068edcc4ce2ff8c499cca51b9a890b76b7ddb569f9a96978013f8b5e8c``.
* Position file: ``K:/ExternalReceivedDatasets/ANPHY/Co-registered average positions.pos``;
  SHA-256 ``b5bbc85ef1e840e0ca703c62c22a1000dd3252f32210e251b8d0e81123fa1bd2``.
* Source overlay SHA-256:
  ``3f1d3135d45ad43648ef1f1c24b0732970ad1492209a414804c35c6eed240824``.

The exact interpreter was
``H:/SourceRepo2/NeuralManifoldDynamics/.venv/Scripts/python.exe``:
Python 3.12.0, MNE 1.6.1, NumPy 2.1.1, and SciPy 1.14.1. This matches the
archived E1-G2 sidecar environment. The first runner draft also contained a
transcription error in its expected FIR hash; that literal was corrected from
the archived sidecar without changing filter settings or code. Under system
Python with NumPy 1.26.4, the same filter generator produced a different live
FIR hash (``76a400263fbc8d366ab4defd7dbb90f5b51c984483ed40588024e575e1d3841d``),
so that environment was not used for the realization. Its failed preflight is
retained separately at
``outputs/OI-REALIZATION-001/run_20261012T143620Z_sl004/preflight_failure.json``.
The archived-environment rerun reproduced the sidecar hashes exactly; no
preprocessing or coefficient-generator changes were made.

Sensor and reference realization
---------------------------------

The ordered input and reference each contain the same 83 labels. The reference
is average reference over exactly this fixed set; no channel was silently
removed or substituted.

* Legacy case-folded newline order hash:
  ``5a3d09949a04c1a1483e8077e7e51ce96773c3092e0b92c9343255e9c1eeaad4``.
* OI exact-label order hash:
  ``9559b64ead401c7399f6218720bc47fe9f7b7c8b785c27c48526542412e2e1e7``.
* OI sorted-label set hash:
  ``9819377836ecdb54673e8eb5e84a9dc4c19ee0a42e0ff665c9f56794e914fcce``.
* Reference ID: ``average_over_fixed_channel_set``; reference labels equal the
  ordered input labels; reference-set hash is the same sorted-label hash above.

Exact ordered labels (also copied into every result record):

.. code-block:: text

   Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6,
   Fz, Cz, Pz, SO1, SO2, F9, F10, Zy1, Zy2, T9, T10, P9, P10, AF7, AF3,
   F11, F5, F1, FT11, FT9, FT7, FC5, FC3, FC1, FCz, C5, C1, TP11, TP9,
   TP7, CP5, CP3, CP1, P11, P5, P1, PO7, PO3, POz, Oz, Fpz, AFz, AF4,
   AF8, F2, F6, F12, FC2, FC4, FC6, FT8, FT10, FT12, C6, C2, CPz, CP2,
   CP4, CP6, TP8, TP10, TP12, P2, P6, P12, PO4, PO8

Preprocessing and effective support
-----------------------------------

The loader replayed the documented branch without changing its operations:

* EDF continuous input at 1000 Hz; causal ``scipy.signal.upfirdn`` downsampling
  with ``up=1``, ``down=4``, 81 taps, and coefficient SHA-256
  ``33a200a29b1076712d73b299c490abad2e3d6096e6f75a67fb8b4476af66ce73``.
  Its 40-input-sample (0.04-second) group delay is not compensated; boundary
  extension is zero before recording start with no future samples.
* MNE FIR bandpass from 1 to 45 Hz, minimum phase, constant padding, 825 taps,
  coefficient SHA-256
  ``fb33689327b6b81f630bf36cdf04b794ba66a022d3d1eb40b1b8a9dc2d081537``.
  For output sample ``k``, the FIR depends on input
  ``[max(0, k - tap_count + 1), k]``.
* Average reference over the fixed 83-label set. No notch or CSD; no phase-delay
  compensation.
* Every nominal window contains 5,000 output samples at 250 Hz. Raw-source
  dependency bounds below are half-open EDF sample ranges at 1000 Hz. Filter
  output bounds are inclusive indices on the resampled 250 Hz grid.

The old system-Python hash mismatch was environmental drift: the live hash did
not match the archived sidecar. The v3.2 runner stopped at the identity guard
and emitted no OI values for that attempt. Using the matching project virtual
environment reproduced the expected FIR hash above. The failure record and
the successful run's ``loader_preflight.json`` keep these attempts separate.

Window results and raw-source dependencies
------------------------------------------

The OI scalar is the median of 83 per-channel permutation entropies. Each row
uses all ``5000 - 4 = 4,996`` consecutive order-5, delay-1 patterns, stable
mergesort for ties, and Shannon entropy divided by ``ln(120)``. The matrix-only
record remains support-unknown; the realization wrapper records known support
only after the loader audit passes. All emitted values below have status
``computed_exploratory`` and ``qualified=false``.

.. list-table:: All 15 nominal windows; source dependency bounds are ``[start,end)``
   :header-rows: 1

   * - Index
     - Nominal seconds
     - Raw source samples at 1000 Hz
     - Filter output samples at 250 Hz
     - Status
     - Normalized OI
   * - 0
     - ``[0,20)``
     - ``[-3376,19997)``
     - unavailable
     - unsupported: insufficient effective filter edge support
     - null
   * - 1
     - ``[20,40)``
     - ``[16624,39997)``
     - ``[4176,9999]``
     - computed_exploratory
     - 0.642298902536
   * - 2
     - ``[40,60)``
     - ``[36624,59997)``
     - ``[9176,14999]``
     - computed_exploratory
     - 0.638960024898
   * - 3
     - ``[60,80)``
     - ``[56624,79997)``
     - ``[14176,19999]``
     - computed_exploratory
     - 0.623099280992
   * - 4
     - ``[80,100)``
     - ``[76624,99997)``
     - ``[19176,24999]``
     - computed_exploratory
     - 0.625425449881
   * - 5
     - ``[100,120)``
     - ``[96624,119997)``
     - ``[24176,29999]``
     - computed_exploratory
     - 0.618321195587
   * - 6
     - ``[120,140)``
     - ``[116624,139997)``
     - ``[29176,34999]``
     - computed_exploratory
     - 0.619897177930
   * - 7
     - ``[140,160)``
     - ``[136624,159997)``
     - ``[34176,39999]``
     - computed_exploratory
     - 0.626033744632
   * - 8
     - ``[160,180)``
     - ``[156624,179997)``
     - ``[39176,44999]``
     - computed_exploratory
     - 0.603971183683
   * - 9
     - ``[180,200)``
     - ``[176624,199997)``
     - ``[44176,49999]``
     - computed_exploratory
     - 0.602471065494
   * - 10
     - ``[200,220)``
     - ``[196624,219997)``
     - ``[49176,54999]``
     - computed_exploratory
     - 0.621964680888
   * - 11
     - ``[220,240)``
     - ``[216624,239997)``
     - ``[54176,59999]``
     - computed_exploratory
     - 0.594452790594
   * - 12
     - ``[240,260)``
     - ``[236624,259997)``
     - ``[59176,64999]``
     - computed_exploratory
     - 0.608916321853
   * - 13
     - ``[260,280)``
     - ``[256624,279997)``
     - ``[64176,69999]``
     - computed_exploratory
     - 0.612213117226
   * - 14
     - ``[280,300)``
     - ``[276624,299997)``
     - ``[69176,74999]``
     - computed_exploratory
     - 0.580017363691

The unsupported first row is retained as a nominal window with null matrix
value. Its effective source dependency begins at sample ``-3376`` before the
recording boundary; it is not shifted, replaced, or treated as zero. The
successful run contains ``loader_preflight.json``, ``run_manifest.json``, and
one JSON sidecar per nominal window. The manifest records SHA-256 for each
sidecar and the exact dependency, support audit, preprocessing, labels,
reference, runtime, and source metadata.

Deterministic ordinal controls
------------------------------

All controls use 5,000 samples, order 5, delay 1, and 4,996 patterns. The
permutation null is one deterministic, seeded exact-multiset shuffle; it is an
implementation control, not an inferential null distribution.

.. list-table:: Synthetic control results
   :header-rows: 1

   * - Control
     - Normalized entropy
     - Unique patterns
     - Result
   * - Monotone ramp
     - 0.000000000000
     - 1
     - expected single pattern
   * - Period-5 rank sequence
     - 0.336175583958
     - 5
     - expected five patterns
   * - Seeded exact-multiset permutation
     - 0.997532034402
     - 120
     - shuffled sample multiset equals ramp multiset
   * - Tie-heavy repeated sequence
     - 0.348226556811
     - 6
     - finite and deterministically tied

All four controls are exploratory and unqualified. The same deterministic
controls were replayed twice and returned identical results.

Implementation identities
--------------------------

The run manifest records the runtime implementation hashes for the audited
loader, support helpers, kernel, and wrapper. The focused handoff test source
hash is included below as validation provenance; tests are not runtime
components:

.. list-table:: Implementation SHA-256 values
   :header-rows: 1

   * - Component
     - Source
     - SHA-256
   * - EEG source-window loader
     - ``mndm/mnps/v3_1/eeg_windows.py``
     - ``4f315d77edd42b07ac5a168ac8592cb5de1b3367a2fc615736c607c6df11cba1``
   * - EEG metrics
     - ``mndm/mnps/v3_1/eeg_metrics.py``
     - ``a26224d59c1177e81e3eb30b734de24ba5cc9bb83ac60f290c5b7f41855727b5``
   * - Candidate protocols
     - ``mndm/mnps/v3_1/candidates.py``
     - ``40ef8fb1f703c8a38fab034d4cbac7bbc1ce4a726a4e7fcb1f35459e9b4cf678``
   * - Preprocessing/support helpers
     - ``mndm/preprocess.py``
     - ``e1b4d3c1c9e4953166f5fb8269580f2df99a5cad7a3f520d60e9385fdf847f5b``
   * - EEG permutation entropy helper
     - ``mndm/features/eeg_entropy.py``
     - ``bd3f7bcc202bf4410d9d0c9aed21e3066c8e7dc38c6955666f1b94a93027dd4f``
   * - OI matrix kernel
     - ``mndm/mnps/v3_2/oi.py``
     - ``79f6bf4fdcfd565d13b8f30a55879d76f1c6d2c302e6d3f55093c455a9e69995``
   * - OI realization runner
     - ``project/new_mnps_v.3.2/experiments/oi_realization_001.py``
     - ``5f9603184268f062d583db17498f3e47365665d19c0d3fec682a8250b7290034``
   * - OI handoff tests
     - ``project/new_mnps_v.3.2/experiments/test_oi_realization_001.py``
     - ``6ff0068d85038113db2aebd21b39aecbc316535a926ea6c17c1c2a734fa6141b``
   * - v3.2 contracts
     - ``mndm/mnps/v3_2/contracts.py``
     - ``3c37e5e7c7451b08c7e8e5875656a333b1b57aee3162c9eb98156eb05c172b7b``

Composite code SHA-256:
``c82befd0729a2bf8c4a28fa79536cbfbdb97268239eeafdc00717d8999cfd38c``.
These hashes are run provenance, not qualification evidence.

Validation and limits
---------------------

Focused synthetic tests cover an audited matrix-to-window handoff, the
matrix-only unknown-support record, incomplete end-to-end support with a
preserved computable value, an unsupported edge window with a null value, and
deterministic ordinal controls. Results are recorded in
``outputs/OI-REALIZATION-001/run_20261012T_run2_archivedenv_sl004/``.

The run is a one-recording engineering pilot. Dataset release and session,
task, and run identifiers remain unresolved. It does not establish EEG support
for windows shorter than 20 seconds, another recording, a different sensor
set or reference, comparability with Core19, or scientific qualification. No
pipeline/export integration or chart/readiness change was performed here.
