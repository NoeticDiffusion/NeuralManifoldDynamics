ED-REFERENCE-SENSITIVITY-001
===========================

Status
------

Completed as a bounded engineering diagnostic on the same audited ``sub-206``
EyesClosed recording used by ED-RELIABILITY-001 and
ED-DURATION-MECHANISM-001. The run reproduced eight non-overlapping 20-second
windows, each divided into five contiguous 4-second blocks, under two
separately identified reference conditions and three separately identified
sensor realizations. All values remain ``qualified=false``. ED v1 and earlier
run outputs were not changed.

Question and frozen method
--------------------------

The diagnostic measured the squared alignment of the leading total, within,
and between-block covariance directions with the uniform sensor vector,

.. code-block:: text

   q = ones(N) / sqrt(N)
   A_common = (v1.T @ q) ** 2

It also repeated the frozen ED-DURATION-MECHANISM-001 covariance decomposition
and raw ED calculation under an explicitly versioned average-reference
sensitivity condition. For that condition, the samplewise arithmetic mean was
subtracted across **all 64 source EEG channels in BIDS Channels TSV order**
(excluding Status) on the full source matrix before selecting any realization.
Core19 and Custom10 therefore share the same 64-channel reference transform;
neither subset is rereferenced using its own mean. The comparison condition
``recorded_fcz_v1`` uses the unchanged source values with their recorded FCz
reference. The condition is recorded in output provenance; the ED estimator ID
remains ``ED.sensor_covariance_participation_ratio_raw_matrix.v1``.

Source identity and execution
-----------------------------

The source EDF SHA-256 matched
``9a25d6079631d57c1f21c0c8a8cd3ff4f9d29e58824ef598c54a40291c4263c4``. The
runner verified the selection manifest, EEG JSON, Channels TSV, and Events TSV
hashes; source label order; the 1000 Hz grid; and the finite 64-by-211000
float64 EEG matrix with Status excluded. The source support and all eight
window intervals match the earlier frozen protocols.

Run ``run_20261012T142500Z_sl004`` completed with 48
window/reference/realization rows, 240 five-block rows, and 288 retained
covariance/mean arrays. All 48 covariance reconstructions, leading-axis
partitions, and raw-ED-to-covariance checks passed. Maximum relative
Frobenius reconstruction error was ``4.12e-15``; maximum leading-axis
partition error was ``3.33e-15``; and maximum raw ED parity error was
``2.22e-15``. The maximum absolute matrix reconstruction residual was
``1.87e-9`` square microvolts, while the relative error stayed near floating
point precision at the source covariance scale.

Observed results
----------------

Values below are means over the same eight windows within each realization.
The paired ED change is average-reference ED minus recorded-FCz ED; the ratio
range is the minimum and maximum paired per-window ratio. These are
within-recording descriptions, not thresholds or population estimates.

.. list-table:: ED and between-block common-mode diagnostics
   :header-rows: 1

   * - Realization
     - Mean ED, FCz
     - Mean ED, average reference
     - Mean paired ED change (range)
     - Mean paired ED ratio (range)
     - FCz ``A_common`` total / between
     - Average-reference ``A_common`` total / between
   * - Native64
     - 1.636
     - 1.713
     - +0.077 (-0.178 to +0.473)
     - 1.051 (0.904–1.296)
     - 0.196 / 0.191
     - approximately 0 / approximately 0
   * - Core19
     - 1.574
     - 1.688
     - +0.114 (-0.197 to +0.813)
     - 1.084 (0.904–1.523)
     - 0.191 / 0.186
     - 0.090 / 0.025
   * - Custom10
     - 2.162
     - 1.928
     - -0.233 (-0.979 to +0.308)
     - 0.934 (0.658–1.220)
     - 0.248 / 0.242
     - 0.227 / 0.150

For Native64 under full-surface average reference, ``A_common`` is
approximately zero by construction: the 64-channel data lie in the
zero-sum sensor subspace. The between-covariance uniform-axis variance
fraction was ``0.178`` under FCz and numerical zero under average reference.
This is a reference-transform property, not evidence that the removed
component was neural or artifactual.

The Native64 between-block mechanism otherwise changed little in these
descriptive summaries: between trace fraction was ``0.734`` under FCz and
``0.734`` under average reference; its contribution on the leading total axis
was ``0.845`` and ``0.848``; and squared alignment of total and between leading
directions was ``0.965`` and ``0.964``. Thus the leading between-block
structure remained visible after removal of the full-surface uniform mode in
this recording. This does not establish the cause of the remaining structure.

Core19 and Custom10 show that measured sensitivity depends on the chosen
realization: their mean paired ED changes have opposite signs, and their
between-block trace fractions decrease from ``0.810`` to ``0.770`` and from
``0.706`` to ``0.668``, respectively. These subset results use the shared
Native64 average reference before subsetting. No subset-specific reference
condition was run.

Interpretation and limits
-------------------------

The Native64 alignment results do not support a dominant between-block
direction along the FCz realization's uniform sensor vector: mean
``A_common_between`` was ``0.191``. Full-surface average reference removes
that direction by construction, but the between-block trace fraction, leading
axis contribution, and total-between alignment remain similar. In this
recording, that particular low-rank between-block structure is therefore not
strongly reference-sensitive by these diagnostics. The ED scalar still shows
window-varying reference sensitivity, and Core19/Custom10 have different
paired changes. No biological or artifact explanation is identified.

Average reference can remove genuine spatially widespread activity as well as
reference-related common variation. This experiment does not show that FCz is
preferable, establish a general property of EEG, or qualify ED. One recording
cannot establish population, test-retest, cross-dataset, or scientific
comparability.

Artifacts and hashes
--------------------

* Frozen protocol: ``project/new_mnps_v.3.2/protocols/ED-REFERENCE-SENSITIVITY-001.md``
* Runner: ``project/new_mnps_v.3.2/experiments/ed_reference_sensitivity_001.py``
* Deterministic controls: ``project/new_mnps_v.3.2/experiments/test_ed_reference_sensitivity_001.py``
* Run directory: ``outputs/ED-REFERENCE-SENSITIVITY-001/run_20261012T142500Z_sl004/``
* Protocol SHA-256: ``34b88a1f4f25a4329a29e8ca128e65199b354a9606bb3538a3df7d915391b8ed``
* Runner SHA-256: ``38258ffb695e58be2ab142b47f36b2ebfdd112df21207693c5bd9125e7ab94b7``
* Test SHA-256: ``27a1a0240f464336691e471858d9d5aa4918c8f575514822edab4342fddcd3fd``
* Unchanged ED kernel SHA-256: ``e470b405775c3052e21db90f9a6223a83513e9b74109990becfbdf54ab1a0bd3``
* Run-manifest SHA-256: ``bf42191d2049c190aa215bcfb6cd296086d7f4dd062a95a332d89c0199c8be9d``

  The manifest records expected and observed EDF/sidecar/selection hashes,
  dependency versions, ordered labels and hashes for each realization, the
  64-channel reference-set hash, exact support, QC counts, and output hashes.
  The source audit, long-form diagnostics, block table, and retained arrays
  each have their SHA-256 in the manifest.

The focused synthetic and decomposition regression suite passed (8 passed).
