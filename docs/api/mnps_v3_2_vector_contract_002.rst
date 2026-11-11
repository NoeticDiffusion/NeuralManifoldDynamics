MNPS v3.2 VECTOR-CONTRACT-002: PCV, OI and AR
==============================================

Status: ``CONTRACT_UNRESOLVED`` for the complete v3.2 EEG realizations. The
SL-003-authorized OI matrix-level kernel is implemented under a new estimator
ID; its end-to-end EEG realization remains unresolved.

This is the tracked record of the audit requested by science-lead decision
SL-002 and updated for SL-003. The equations and versioned v3.1 examples below
are evidenced in the repository. They do not select one end-to-end v3.2
realization per family. A separately versioned OI matrix kernel is now
implemented; PCV and AR remain scaffolds, and OI still has no audited
preprocessing, EEG time-support, reference, or sensor-realization wrapper.

SL-003 OI kernel decision
-------------------------

The narrow standalone estimator
``OI.matrix_input_channel_median_normalized_PE5_delay1.v1`` freezes order 5,
delay 1 sample, stable-mergesort tie handling, Shannon entropy with natural
logarithms normalized by ``ln(5!)``, and the median over every supplied matrix
row. A ``(channels, T)`` input contributes ``T - 4`` patterns per row; fewer
than five samples and any non-finite or complex sample are unsupported. The
kernel preserves optional row labels and order/set hashes, records the shared
measurement envelope, leaves temporal/source support unknown, marks the
sensor realization incomplete/unresolved, and always emits ``qualified=false``.
It makes no preprocessing, fixed-set, reference, sample-rate, or end-to-end
window choice. This new ID does not relabel either v3.1 E1 realization. See
``mnps_v3_2_oi.rst`` for its exact contract and tests.

Decision summary
----------------

.. list-table:: Frozen definitions and unresolved v3.2 dependencies
   :header-rows: 1
   :widths: 10 31 30 29

   * - Family
     - Frozen mathematical definition
     - Repository-backed, bounded realization
     - Unresolved before v3.2 end-to-end implementation
   * - PCV
     - ``R[t] = abs(mean_c(exp(i*phi[c,t])))``; value is the population SD
       (``ddof=0``) over time. Scalar and sensor-set specific. Core ID:
       ``M1.sensor_phase_order_parameter_sd.v0``.
     - M1-G2 owner decision selected an alpha 8--13 Hz, fixed 83-label ANPHY,
       250 Hz, 20 s technical evaluation using fourth-order Butterworth SOS,
       zero-phase filtering and Hilbert phase, with the recorded source and
       resampler support. Its 15-window run computed 14 and retained one null
       for insufficient filter/resampler edge support. Verdict: partial
       technical pass; M1 remains deferred and unqualified. A separate current
       pipeline ID, ``PCV.pipeline_native_window_alpha_phase_order_parameter_sd.v1``,
       filters the supplied native window locally using the same nominal band
       and phase method; it is not the contextual G2 realization.
     - Select the realization, phase/filter identity, reference, channels,
       preprocessing branch and effective support together. The bounded 20 s
       result does not freeze shorter-window or other-sensor support. Do not
       treat the two v3.1 paths as equivalent.
   * - OI
     - Standalone v3.2 matrix estimator: per row, normalized order-5,
       delay-1 permutation entropy divided by ``ln(5!)``; stable-mergesort
       ties; median over all supplied rows. ``T`` samples supplies ``T-4``
       patterns. New ID: ``OI.matrix_input_channel_median_normalized_PE5_delay1.v1``.
       This does not change either legacy v3.1 ID.
     - E1-G2 fixed 83-label ANPHY realization: 250 Hz, 20 s/5,000 samples,
       order 5/delay 1, normalized entropy, median over all 83 channels;
       preprocessing record specifies causal resampling, 1--45 Hz
       minimum-phase FIR, average reference over the set, no notch and no CSD.
       G2 computed 14/15 windows, partial pass for support evaluation only.
       The current shared entropy helper uses ``np.argsort(kind="mergesort")``
       for stable tie ordering. The distinct pipeline ID
       ``OI.pipeline_native_window_permutation_entropy_order5_delay1.v1``
       instead aggregates over all native channels.
     - The matrix kernel does not choose between fixed-set E1 and the
       native-channel pipeline version, and does not claim parity with either.
       For a full EEG realization, still decide the channel realization,
       preprocessing/reference, sample grid, source/time support, and valid
       window policy. The 20 s E1 support result does not establish 4/8/12 s
       support. The matrix kernel's five-sample minimum is a mathematical
       computability boundary, not a scientifically qualified window rule.
   * - AR
     - For declared equal adjacent blocks and per-block activity vectors
       ``v_b``, return the ordered vector ``1 - corr_sensor(v_b,v_(b+1))``.
       Do not aggregate. Activity and connectivity variants have distinct IDs;
       zero-centered-norm inputs are undefined.
     - The low-level M3 function accepts caller-constructed block vectors and
       requires construction/scaling provenance; it does not choose the
       features. M3-G1-S1 v2 exercised 45 synthetic options but selected no
       primary representation, construction, scaling, block rule or
       aggregation; M3 remains deferred. The separate pipeline ID
       ``AR.pipeline_native_window_adjacent_block_rms_change.v1`` uses native
       channel RMS per equal block, defaults to 2 s, internally scales and
       centers each block vector, and emits each adjacent-pair value. An
       accepted H1 amendment records nine ordered components for its specific
       20 s/2 s case and prohibits collapsing them.
     - Select activity or connectivity; exact feature construction and input
       branch; sensor/edge realization; scaling; block duration or count;
       divisibility/tail and undefined-pair policy; and the consumer's vector
       shape. With 2 s blocks, 4/8/12/20 s windows yield 1/3/5/9 pair values.
       No v3.2 decision defines a scalar aggregation or fixed output shape.

Required decisions
------------------

* **PCV:** state whether to reproduce the bounded v3.1 M1-G2 path or the
  distinct native-window pipeline path. For a v3.2 realization, define the
  phase estimator, band, reference and channel realization, preprocessing,
  source dependency/edge handling and each supported window duration.
* **OI:** the standalone matrix estimator freezes order, delay, stable ties,
  normalization and row-median aggregation. For a full EEG realization,
  decide whether it uses fixed-set E1, native-channel, or another explicitly
  versioned sensor realization; define preprocessing/reference, sampling grid,
  source/time support and each candidate window duration. Do not infer
  comparability or qualification from matching channel labels.
* **AR:** choose activity or connectivity and freeze feature construction,
  scaling, sensors/edges, partition rule and vector support/completeness.
  Specify how a downstream consumer represents the ordered components. A
  scalar or another transform requires a separate estimator definition.

No candidate is scientifically qualified by the technical evidence above.
Keep ``computed``, ``valid``, ``support_status`` and ``qualified`` distinct.

Evidence locations
------------------

* Formula and common window contract:
  ``project/new_mnps_v.3.2/formulas/MNPS_successor_eight_metric_formula_contract_20261011.md``
* v3.2 decision points: ``project/new_mnps_v.3.2/IMPLEMENTATION_PLAN.md`` and
  ``project/new_mnps_v.3.2/VALIDATION_PLAN.md``.
* v3.1 PCV formula and contextual extraction: ``mndm/src/mndm/mnps/v3_1/eeg_metrics.py``,
  ``mndm/src/mndm/mnps/v3_1/m1_phase.py`` and
  ``mndm/src/mndm/mnps/v3_1/candidate_grid.py``. The separate window-local
  pipeline is in ``mndm/src/mndm/mnps/v3_1/pipeline_measurements.py``.
* v3.1 E1 constants and fixed-set contract:
  ``mndm/src/mndm/mnps/v3_1/candidates.py``; tie ordering is in
  ``mndm/src/mndm/features/eeg_entropy.py``; the distinct native-channel
  pipeline is in ``mndm/src/mndm/mnps/v3_1/pipeline_measurements.py``.
* v3.2 standalone OI kernel and focused tests: ``mndm/src/mndm/mnps/v3_2/oi.py``
  and ``mndm/tests/test_mnps_v3_2_oi.py``.
* PCV evidence: ``project/new_mnps_v.3.1/g1_candidate_owner_decisions_20260930_v2.yaml``,
  ``project/new_mnps_v.3.1/anphy_epctl01_m1_g2_alpha_20260930_v2.yaml`` and
  ``project/new_mnps_v.3.1/outputs/anphy_epctl01_m1_g2_alpha_20260930_v2/mnps_v3_1_M1_anphy_epctl01_g2_alpha_summary.json``.
* OI evidence: ``project/new_mnps_v.3.1/external_llm/g1_decision_record_20260929.md``,
  ``project/new_mnps_v.3.1/anphy_epctl01_e1_g2_grid_20260929.yaml`` and
  ``project/new_mnps_v.3.1/outputs/anphy_epctl01_e1_g2_grid_20260929/mnps_v3_1_E1_anphy_epctl01_g2_summary.json``.
* AR evidence: ``project/new_mnps_v.3.1/m3_g1_s1_option_record_20260930_v2.md``,
  ``project/new_mnps_v.3.1/outputs/m3_g1_s1_vector_options_20260930_v2/m3_g1_s1_summary.json``,
  ``project/new_mnps_v.3.1/candidate_gate_status_20260929.md`` and
  ``project/new_mnps_v.3.1/bench_001/mnps_bench_001_amendment_1_m3_vector_20261007.md``.

The working decision draft is at
``project/new_mnps_v.3.2/protocols/VECTOR-CONTRACT-002-PCV-OI-AR-decision-draft.md``.
