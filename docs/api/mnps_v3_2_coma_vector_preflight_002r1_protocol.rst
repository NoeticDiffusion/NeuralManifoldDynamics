MNPS v3.2: COMA-VECTOR-PREFLIGHT-002-R1 protocol
================================================

Status
------

Frozen for an outcome-blind technical feasibility run. The target is the first
24 hours after ROSC for 16 deterministically selected patients (within the
authorized 12--20 range). The protocol emits all computable 20-second windows
inside that interval; it does not select patients or windows by metric values.
This is a separately versioned extension of COMA-VECTOR-PREFLIGHT-001, whose
12--24 hour scope remains unchanged.

Revision R1 repairs the smoke-test support audit after the initial 002 smoke
run. The raw read, source hashes, and crop-grid checks passed, but the runner
incorrectly required a fixed 100 Hz output while the frozen per-file resolver
selected 250 Hz for the smoke segment. That run produced no metric values and
remains preserved under ``outputs/COMA-VECTOR-PREFLIGHT-002/``. R1 validates
against the configured resolved rate and derives each 20-second window's
sample count from that rate. The measurement definitions, preprocessing
configuration, source inventory, patient draw, and window intervals are
unchanged.

Purpose and claims
------------------

Check whether raw I-CARE v2.1 EEG supports the v3.2 ED, OI and NIE measurements
under one audited preprocessing path. No CPC prediction, outcome association,
between-group comparison, research qualification, or biological claim is
authorized. All output records are experimental and ``qualified=false``.

Source, time axis, and selection
-------------------------------

The source is the raw PhysioNet I-CARE v2.1 WFDB training tree configured at
``E:/Science_Datasets/physionet/received/i-care_2_1_random100_longitudinal/training``.
The source manifest inventories EEG ``.hea`` headers and ``.mat`` payload paths;
it hashes every header and records every payload's relative path and size. Each
selected header and payload is hashed again and the payload hash is captured
before its samples are read. The release-wide payload bytes are not re-hashed
as a single archive; exact hashes are recorded for all selected input pairs.
Outcome sidecars (``*.txt``) are not opened during inventory, selection,
preprocessing, or metric extraction.

Use the WFDB header start/end clocks as elapsed time from ROSC. This is the
I-CARE release's stated time axis (see `PhysioNet v2.1 description
<https://www.physionet.org/content/i-care/2.1/>`_). Do not shift these clocks a
second time using the optional ``ROSC`` sidecar field. The sample count and
sample rate define the source segment's half-open end; the header end clock is
checked against its last sample and any discrepancy is retained as QC.

A patient is eligible when at least one EEG header has a parseable start/end
clock and a valid positive sample rate and sample count. Payload availability,
channel count, target-period overlap, and measurement values are not selection
criteria. Sort eligible opaque patient IDs by SHA-256 of
``COMA-VECTOR-PREFLIGHT-002|20261014|patient_id`` and freeze the first 16. If
fewer than 12 patients are eligible, stop and report that before metric
extraction. Retain selected patients with no first-day windows or no readable
payload in the roster and report zero support rather than replacing them.

Enumerate every full, non-overlapping 20-second interval
``[20k, 20(k+1))`` seconds from ROSC, wholly within ``[0, 86400)`` and wholly
inside one source segment. Process every segment that intersects this period;
never concatenate across segment or gap boundaries. If two source segments
overlap in ROSC time, retain their rows separately with segment identity and
an overlap QC flag. Sort segments by parsed start time and normalized
source-relative path.

Smoke test and cohort continuation
----------------------------------

Before the cohort run, use the first patient in the frozen draw with at least
nine target windows in one segment that has its paired signal payload. Process that patient's first nine full
target windows (three minutes of analyzed signal) and preserve all nine rows.
Continue to the 16-patient first-day run only if the smoke test completes the
raw WFDB read, records stable source hashes and preprocessing provenance, and
produces at least one valid native-channel value for each of ED, OI and NIE.
The smoke test does not alter the frozen selection or exclude any patient from
the cohort run.

Realizations and preprocessing
------------------------------

Use the frozen I-CARE v2.1 configuration and MNPS-only overlay:
``mndm/config/sources/other/config_ingest_physionet_i-care_2_1.yaml`` and
``config_ingest_physionet_i-care_2_1_mnps_only.yaml``. Resolve and record their
hashes before reading metric values. Preserve the configured per-file rate
policy (base 100 Hz; candidates 100, 128, 200, 250 Hz), 0.5--40 Hz bandpass,
no notch, average reference across the actual EEG input set, no artifact
removal, and ``scale_to_microvolts=false``. Preserve source ``nu`` units as
unknown; do not relabel them as microvolts.

For each segment, crop the raw array to its intersection with the first-day
interval before resampling and filtering. Record the exact source sample crop,
filter/resampling operations, margins and boundary behavior. The effective
dependency is conservatively recorded as the full preprocessed crop because
the configured resampling/filter path can use context beyond a 20-second
window. This bound must remain inside both the raw source segment and the
first-day interval. A window is unsupported if its source mapping, crop,
segment boundary, payload hash, or finite-sample support cannot be verified.

Emit each patient's native EEG-channel realization. Also emit a separate
Core19 realization only if the exact ordered set
``Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2`` is present.
Do not alias T3/T4/T5/T6, interpolate, silently drop required channels, or use
Core19 as a prerequisite for native metrics. Record each realization's exact
ordered labels, order/set hashes, channel count, reference labels/hash,
coverage, support and QC.

Compute ED, OI and NIE independently on every supported realization/window
using the unchanged v3.2 kernels. Preserve null values and family-local reasons
for unsupported estimators; one unsupported family does not suppress other
computable values. Keep each window separate. Do not aggregate to one patient
value or use an outcome label.

Reproducibility artifacts
-------------------------

* Runner: ``mndm/src/mndm/mnps/v3_2/experiments/coma_vector_preflight_002r1.py``.
* Output root: ``outputs/COMA-VECTOR-PREFLIGHT-002-R1/``.
* Freeze a source inventory and deterministic selection record before any
  signal sample or metric value is read.
* Hash each selected source pair before reading samples; include exact source,
  config, preprocessing, kernel, runner and protocol hashes in the manifests.
* Keep the three-minute smoke output separate from the first-day cohort output.

The preflight establishes technical feasibility and support only. It does not
establish representativeness of the first day, cross-patient comparability,
clinical usefulness, or qualification for downstream scientific use.
