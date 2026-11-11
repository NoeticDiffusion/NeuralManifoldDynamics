ED-RELIABILITY-001
==================

Status
------

Protocol version 1 is frozen on 2026-10-12 for source selection, temporal
support, window grid, sensor realizations, and sensitivity arms. The SL-002
measurement arms were frozen before extraction. The bounded execution
completed successfully as an engineering experiment; the measurements remain
unqualified. The selection and support details here are the authoritative,
non-ignored record; a fuller working copy is under
``project/new_mnps_v.3.2/protocols/ED-RELIABILITY-001.md``.

Question
--------

Measure how window duration, sensor realization, temporal correlation, and
gain perturbations affect raw covariance participation ratio on one
outcome-blind selected EEG recording. Keep every window result. This is a
within-recording stability probe, not test-retest or biological reliability.

Frozen recording
----------------

Use the first subject in the existing MNPS-BENCH-001 outcome-blind selection:

* Dataset: OpenNeuro ``ds005385`` v1.0.3 / NEMAR ``on005385`` v1.0.0,
  DOI ``10.18112/openneuro.ds005385.v1.0.3``.
* Selection manifest: ``project/new_mnps_v.3.1/bench_001/selection_manifest_outcome_blind.json``;
  SHA-256 ``5c7e11b590abf7a3cc69ed0e97fd7821f8c41eab1bc729f3b00311c4550c8d18``.
  It selected 100 of 608 eligible subjects by ascending SHA-256 of
  ``MNPS-BENCH-001|sub-NNN``; the first is ``sub-206``.
* Recording: ``sub-206/ses-1/eeg/sub-206_ses-1_task-EyesClosed_acq-pre_eeg.edf``.
* EDF SHA-256: ``9a25d6079631d57c1f21c0c8a8cd3ff4f9d29e58824ef598c54a40291c4263c4``.
* EEG JSON SHA-256:
  ``77be9d77936e3d1295357704de02db992e9b30abf97903210fbb419952112927``.
* Channels TSV SHA-256:
  ``ba4ce6af865979008b4d46ed29b89e5f0096502e627b885af874d76d36cfab99``.
* Events TSV SHA-256:
  ``fbed82986d55ea0015f4d31fea8305514416059de82ffa759b1ce27a0d4441d6``.

BIDS metadata declares a continuous 211-second recording, 64 EEG channels at
1000 Hz, units of microvolts, and source reference ``FCz``. The EDF fixed
header has 211 records of one second, 65 signals (64 EEG and ``Status``), 1000
samples per signal per record, and a 16,896-byte header. The observed 27,446,896
bytes match the fixed-record layout. The Channels TSV has 64 rows, all typed
EEG.

The Events TSV marks ``boundary`` at 1 ms, 25,041 ms, and 44,001 ms. Treat these
as conservative segment boundaries. Use only the final interval after the
last marker, from sample 45,000 (45.000 s, the next whole-second sample) to
sample 211,000 (211.000 s). A pre-run reader must still verify the actual
64-by-211000 finite EEG matrix. If that check fails, retain null/status/QC; do
not silently drop or impute channels.

Preprocessing and windows
-------------------------

Use physical EEG values in microvolts, the recorded ``FCz`` reference, and the
source 1000 Hz grid. Do not filter, notch, resample, rereference, interpolate
bad channels, or reject artifacts in this protocol. Exclude ``Status``. Line
noise and slow drift remain in the declared measurement and are limitations.

Windows are half-open sample intervals, begin at sample 45,000, and use a step
equal to their length. No window crosses a boundary marker; all complete
windows are kept, with only the final incomplete tail omitted.

.. list-table:: Frozen window grid
   :header-rows: 1

   * - Length
     - Samples
     - Step
     - Windows
     - Last interval (seconds)
     - Tail
   * - 4 s
     - 4,000
     - 4,000
     - 41
     - ``[205,209)``
     - 2 s
   * - 8 s
     - 8,000
     - 8,000
     - 20
     - ``[197,205)``
     - 6 s
   * - 12 s
     - 12,000
     - 12,000
     - 13
     - ``[189,201)``
     - 10 s
   * - 20 s
     - 20,000
     - 20,000
     - 8
     - ``[185,205)``
     - 6 s

All windows satisfy ``T>N`` for the following sensor sets. This is a numeric
computability condition, not evidence of enough independent observations.

Realizations and controls
-------------------------

* ``native64``: all 64 EEG labels in Channels TSV order.
* ``core19``: ``Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2``.
* ``custom10_subset_v1``: ``Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2``.

Each realization uses the same recorded reference; subsetting does not
rereference. Store actual ordered labels, order hash, set hash, count, reference,
support, coverage, and QC per output.

Common-gain controls use factors ``0.1``, ``1``, and ``10``. A label-keyed gain
control is fixed by

.. code-block:: text

   g(label) = 0.5 + 1.5 * uint32_be(SHA256("ED-RELIABILITY-001|" + label)[0:4]) / (2^32 - 1)

The same label receives the same gain in each realization. These controls are
sensitivity probes, not proposed corrections.

Emit these separate measurements for every supported real-data window and
simulation replicate:

* Raw primary ID ``ED.sensor_covariance_participation_ratio_raw_matrix.v1``.
* Raw channel-count diagnostic ID
  ``ED.sensor_covariance_participation_ratio_normalized_by_channel_count.v1``,
  defined only as ``ED_raw/N``. Report beside absolute ED; it does not correct
  montage, reference, rank, or cross-realization comparability.
* Isotropic-shrinkage ID
  ``ED.sensor_covariance_participation_ratio_isotropic_shrinkage.v1`` with
  ``lambda=0.5`` and ``lambda=1.0`` for every window, realization, gain
  condition and simulation replicate.

The shrinkage definition is

.. math::

   C_\lambda = (1-\lambda)C + \lambda\frac{\operatorname{tr}(C)}{N}I.

Freeze ``lambda=0.5`` as the midpoint sensitivity arm and ``lambda=1.0`` as a
fully isotropic limiting control; the latter is not a promotion candidate.
For positive trace, ``ED(C_1)=N``; assert absolute error at most ``1e-12`` as
an algebraic implementation control. Lambda zero is only a regression test
against raw ED (``rel=1e-13, abs=1e-13``), not an experiment arm. Never select
lambda from recording outcomes.

For every duration, estimator/parameter, realization and gain condition,
report window count, mean, median, standard deviation (``ddof=0``) and IQR.
These describe within-recording dispersion; they are not a reliability gate or
a pooled global dimension.

Known-truth simulation
----------------------

Use stationary float64 AR(1) signals with population covariance ``I_N`` and
therefore population ED ``N``:

.. code-block:: text

   x[0] ~ N(0, I)
   x[t] = rho*x[t-1] + sqrt(1-rho^2)*epsilon[t], epsilon[t] ~ N(0, I)

Evaluate ``N in {10,19,64}``, ``fs=1000 Hz``, durations ``{4,8,12,20} s``,
``rho in {0,0.5,0.9,0.98}``, and 100 replicates per cell. Derive replicate RNG
state from ``SeedSequence([20261012,N,duration_seconds,rho_index,replicate_index])``.
Report mean/median ED, bias relative to ``N``, IQR, and Monte Carlo uncertainty.
This characterizes finite-sample/correlation behavior; it does not establish an
effective-sample correction.

Output and limits
-----------------

Write to ``outputs/ED-RELIABILITY-001/<run_id>/``. Emit long-form window values
and status/QC, a run manifest with source/sidecar/selection hashes, code and
dependency versions, protocol hash, realization labels and support. Retain all
window values. Any within-recording dispersion summary is experiment-only; do
not define a universal subject-level ED.

The initial MNE EDF read attempt did not return within 60 seconds and was
interrupted. PyEDFlib 0.1.38 then verified the full 64-by-211000 finite
float64 physical EEG matrix, BIDS/EDF signal order, 1000 Hz rate, trailing
Status exclusion, and digital-to-physical calibration across all EEG samples
(maximum residual ``1.82e-12`` microvolts). The selected recording was unchanged.

Completed execution
-------------------

Run ``run_20261012T092100Z_sl002`` completed at 2026-10-12 09:22:39 UTC in
119.83 seconds. The full command was ``python -u project/new_mnps_v.3.2/experiments/ed_reliability_001.py --run-id run_20261012T092100Z_sl002``. Focused runner tests passed (4 passed).

The 82 supported windows produced 984 window/realization/gain groups and 3,936 valid measurements: 984 raw ED, 984 ED/N diagnostics, and 1,968 shrinkage rows. The within-recording summary has 192 rows with mean, median, standard deviation (``ddof=0``), and IQR. All 48 AR(1) cells completed: 4,800 replicates, 19,200 measurement rows, and 192 cell summaries.

ED/N equals raw ED divided by channel count for every row. The lambda-one control's maximum absolute error from N was ``5.69e-14`` on recording windows and ``4.27e-14`` in simulation, below the frozen ``1e-12`` tolerance. Common-gain arms were invariant within ``7.11e-14`` across the reported measurements. These are implementation checks, not reliability criteria.

For unperturbed windows, raw ED means by duration were Native64 ``3.452, 2.452, 2.031, 1.636``, Core19 ``3.424, 2.687, 2.176, 1.574``, and Custom10 ``2.628, 2.302, 2.289, 2.162`` (4, 8, 12, 20 seconds respectively). ED/N means are included as separate rows in the result files. Values remain specific to each sensor realization.

Selected simulation means (raw, lambda 0.5, lambda 1) were ``8.843, 9.682, 10`` for N=10, 4 s, rho=0.98; ``35.680, 53.399, 64`` for N=64, 4 s, rho=0.98; and ``55.102, 61.516, 64`` for N=64, 20 s, rho=0.98. Lambda 0.5 moving values toward N does not establish improved validity or authorize an effective-sample correction.

Outputs are under ``outputs/ED-RELIABILITY-001/run_20261012T092100Z_sl002/``. Run-manifest SHA-256: ``53ad3b45bd09abc2e09573ace9b5d889882e0ef53f8afad37b1ac38bf4d8127a``. Frozen pre-run protocol SHA-256: ``558630e4f570804c43ece1046571ce580d13782775108ddb42351c0ab3b4d408``. Runner SHA-256: ``48fb7feb2a4015938f46e6098c04d3c3dc287e75170a2d190998010eb4b9bb9b``. Output hashes are in the manifest.

The separate ``run_20261012T091126Z`` directory is preflight-only (status
``PREFLIGHT_PASS_PAUSED_PENDING_SL002``); it contains the matrix audit and
manifest, with no accepted window or simulation values.

This is an engineering execution, not a scientific PASS. One recording cannot establish test-retest, between-subject, cross-dataset, biological, or clinical reliability. No post-result threshold, group contrast, or qualification claim was applied; all outputs remain ``qualified=false``.
