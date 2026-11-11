ED-DURATION-MECHANISM-001
========================

Status
------

Completed as a bounded engineering diagnostic on the already selected
``sub-206`` EyesClosed recording. The experiment used the same EDF and frozen
source identity as ED-RELIABILITY-001, with no new subjects, altered source
support, gain perturbation, preprocessing, or estimator change. All values
remain ``qualified=false``.

Question and method
------------------

The diagnostic tested whether the covariance in each frozen 20-second window
could be separated into covariance within five consecutive 4-second blocks
and covariance of the block means, and whether the latter aligns with the
leading total-covariance direction.

For each realization and 20-second window, using sample covariance (ddof=1),
``n=4000`` samples per block and ``T=20000`` samples total:

.. code-block:: text

   C_within  = ((n - 1) / (T - 1)) * sum_k(C_k)
   C_between = (n / (T - 1)) * sum_k((mu_k - mu)(mu_k - mu)^T)
   C_20      = sample_covariance(full window, ddof=1)

The analysis used unity-gain, unfiltered physical microvolt data with the
recorded FCz reference. It retained all five block covariances and means, all
component covariance matrices, and per-window spectral diagnostics. The
leading-axis between contribution is ``v20.T @ C_between @ v20 / lambda1(C20)``;
the corresponding squared-cosine alignment compares the leading eigenvectors
of ``C20`` and ``C_between``.

Source identity and execution
-----------------------------

The EDF SHA-256 matched the frozen value
``9a25d6079631d57c1f21c0c8a8cd3ff4f9d29e58824ef598c54a40291c4263c4``.
The runner also checked the frozen selection manifest and EEG JSON, Channels
TSV, and Events TSV hashes, source channel order, 1000 Hz grid, and 64 EEG
channels with Status excluded. The exact support was the eight non-overlapping
windows beginning at samples 45,000 through 185,000, each ending 20,000 samples
later. The final six seconds were unused, as specified by ED-RELIABILITY-001.

Run ``run_20261012T104000Z_sl003`` completed with 24 window/realization rows
(8 windows x 3 sensor realizations), 120 block rows, and 144 retained arrays
for covariance matrices and block means. All 24 covariance reconstructions
passed the frozen relative Frobenius tolerance of ``1e-12``; all 24 leading
axis component partitions passed the ``1e-12`` tolerance. Independent
source-backed reconstruction from the saved components had maximum absolute
covariance difference ``7.28e-12`` and maximum relative Frobenius residual
``4.12e-15``. The covariance participation ratio computed from the saved
``C20`` matched the unchanged ED v1 output for all 24 corresponding
20-second/window/realization rows, with maximum absolute difference
``3.56e-15``.

Observed results
----------------

The table summarizes the eight windows separately within each sensor
realization. Ranges are the minimum and maximum across those eight windows.
These are descriptive summaries for this recording only.

.. list-table:: Between-block variance and leading-axis diagnostics
   :header-rows: 1

   * - Realization
     - Mean between trace fraction (range)
     - Mean total leading spectral share
     - Mean between leading spectral share
     - Mean between contribution on total leading axis (range)
     - Mean squared alignment (range)
   * - Native64
     - 0.734 (0.137–0.873)
     - 0.781
     - 0.872
     - 0.845 (0.093–0.968)
     - 0.965 (0.724–0.9998)
   * - Core19
     - 0.810 (0.728–0.858)
     - 0.796
     - 0.926
     - 0.943 (0.929–0.957)
     - 0.9993 (0.9979–0.9996)
   * - Custom10
     - 0.706 (0.510–0.817)
     - 0.657
     - 0.839
     - 0.885 (0.718–0.937)
     - 0.971 (0.881–0.999)

The between-block term contributed more than half of the total leading-axis
variance in 7/8 Native64 windows and 8/8 windows in each subset realization.
In the first Native64 window, its trace fraction was 0.137 and its leading-axis
contribution was 0.093; this window is retained and prevents describing the
pattern as uniform across all windows. In the other realizations/windows, the
between-block leading direction was closely aligned with the total leading
direction, as shown by the squared-cosine diagnostics.

The component participation ratios provide another view of this structure:

.. list-table:: Mean covariance participation ratios across eight windows
   :header-rows: 1

   * - Realization
     - Total ``C20``
     - Within ``C_within``
     - Between ``C_between``
   * - Native64
     - 1.636
     - 5.292
     - 1.297
   * - Core19
     - 1.574
     - 4.581
     - 1.163
   * - Custom10
     - 2.162
     - 2.877
     - 1.401

The total 20-second values agree with the existing ED-RELIABILITY-001
descriptive means. The component values are diagnostics of the corresponding
covariance matrices, not new qualified ED estimators.

Interpretation and limits
-------------------------

For this recording and these windows, between-4-second-block mean differences
account for a substantial share of total covariance and, in most
window/realization combinations, contribute strongly to the leading total
covariance direction. This is direct evidence for an across-block mean
covariance mechanism associated with the observed low total ED. The result
does not identify why those block means differ. Slow drift, other
nonstationarity, artifacts, reference effects, and other signal structure
remain possible explanations; the frozen unfiltered measurement retains these
possibilities. The result does not demonstrate that the brain becomes lower
dimensional over longer observation windows, establish population or
test-retest behavior, or establish scientific usefulness or qualification.

Artifacts and hashes
--------------------

* Frozen protocol:
  ``project/new_mnps_v.3.2/protocols/ED-DURATION-MECHANISM-001.md``
* Runner:
  ``project/new_mnps_v.3.2/experiments/ed_duration_mechanism_001.py``
* Deterministic synthetic reconstruction test:
  ``project/new_mnps_v.3.2/experiments/test_ed_duration_mechanism_001.py``
* Run directory: ``outputs/ED-DURATION-MECHANISM-001/run_20261012T104000Z_sl003/``
* Protocol SHA-256: ``197888c047ba9a66497b8359833f4186cea6e19322ba1eb41c748997099caa67``
* Runner SHA-256: ``fc6907bce5eabe08737e1f0f2faae8be620c670994eaa8ceec6dc828733ead9e``
* Synthetic test SHA-256: ``bebcc4d8727abe391509e2a1656abc6845f08514acf64c5c5aa8086ac70cb45f``
* Run-manifest SHA-256: ``eaafa3cf648bcddb078157d3ebd4c0bb28358bdfe0a608dfa8b7ef88b9cbea73``
* ``source_audit.json`` SHA-256:
  ``b5456dcb8efca4efcb0a24f013612da7cbcb6608d9dc1228eab363d6eb99d805``
* ``window_realization_diagnostics.csv`` SHA-256:
  ``a2c60d83b70b68860ac1e511907b612e82f1f9c6e6751920533e97e6068aaab0``
* ``block_diagnostics.csv`` SHA-256:
  ``29a4e1c46ca7394cbf821539fdf0bdca807e2cc5058e1b53d7cb578ac5ac7ba9``
* ``covariance_components.npz`` SHA-256:
  ``8f7a4f3d8fc6f1ebd754068a2b9358bc215815e80973f8ed81fe5cf4cfbf7774``

The manifest also records expected and observed source/sidecar/selection
hashes, dependency versions, ordered realization labels, sample support,
per-row QC, and every output hash. The synthetic tests passed (2 passed).
