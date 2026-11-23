# ds004148 MNPS 3.4 baseline QC package

This package records the completed session-1 eyes-closed/eyes-open QC
comparison for 60 ds004148 participants. It uses the frozen
`DS004148_BASELINE_EC_QC_POLICY_V1_1` policy and the external frozen EC anchor
`ds005385_ses1_pretask_EyesClosed_20261111_v1` with manifest SHA-256
`79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab`.

## Reproduction sources

- QC runner:
  [`project/mnps2_audit/run_ds004148_v34_baseline_qc.py`](../../../project/mnps2_audit/run_ds004148_v34_baseline_qc.py)
- Paired cohort summarizer:
  [`project/mnps2_audit/summarize_ds004148_baseline.py`](../../../project/mnps2_audit/summarize_ds004148_baseline.py)
- Frozen policy:
  [`mndm/config/v.3.4/DS004148_BASELINE_EC_QC_POLICY_V1_1.md`](../../config/v.3.4/DS004148_BASELINE_EC_QC_POLICY_V1_1.md)
- Ingest configuration:
  [`mndm/config/v.3.4/config_ingest_ds004148_v34_baseline_ec_anchor.yml`](../../config/v.3.4/config_ingest_ds004148_v34_baseline_ec_anchor.yml)
- Frozen EC anchor:
  [`mndm/config/v.3.4/anchors/mnps_v3_4_ds005385_ses1_pretask_EyesClosed_20261111_v1.json`](../../config/v.3.4/anchors/mnps_v3_4_ds005385_ses1_pretask_EyesClosed_20261111_v1.json)
- Package/dependency declarations:
  [`requirements.txt`](../../../requirements.txt)

The full run covered 120 recordings and 8,880 windows. Its run ID is
`20261118T153711495753Z`. The completed QC and baseline summaries are in
`project/mnps2_audit/outputs/20261116_ds004148_v3_4_frozen_ec_baseline`.
This package keeps the cohort-selection files, input-verification record,
QC and summary manifests, compact paired summary tables, and completed process
log under `artifacts/`.
Large window- and participant-level result tables and the 120 H5 files remain
in that output directory; their sizes and SHA-256 values are listed in
`runtime_and_artifact_inventory.json`.

## Runtime and known compatibility details

The completed run used Python 3.12.0, MNE 1.6.1, NumPy 1.26.4 and SciPy
1.14.1, with `MNE_USE_NUMBA=false` and `NUMBA_DISABLE_JIT=1`. Repository
`requirements.txt` pins NumPy 2.1.1, so the actual environment did not match
that pin. The run manifest records the versions actually used; the inventory
also records the requirements-file hash. Do not describe this run as having
been repeated under the pinned NumPy version. The inventory lists every
installed Python distribution and version detected in the run environment.

On Windows, MNE's default Numba/JIT path did not finish importing within ten
minutes in a fresh process. Either `MNE_USE_NUMBA=false` or
`NUMBA_DISABLE_JIT=1` reduced isolated MNE 1.6.1 import time to about two
seconds. The runner sets and logs both flags before its first MNE import, in
line with `mndm/src/mndm/preprocess.py`; see the Windows startup note in the
[MNDM README](../../README.md#windows-startup-mne-and-numba).

The preflight found 26 BrainVision/BIDS channel-label comparisons with a
capitalization-only difference (`CPz`/`Cpz` or `Fpz`/`FPz`). Each had 61
channels in the same case-insensitive order. The QC runner records the actual
ordered labels and hashes and logs these cases as warnings. Other channel-count
or non-case-only label/order mismatches remain errors.

## Descriptive result

The paired comparison uses participant-level medians and reports EO minus EC;
it does not pool windows or compute p-values. Core-8 coordinate pairs met the
frozen support floor for 45 participants; alpha-envelope-CV and complete-9D
support were available for 44. See
[`qc_attrition_summary.md`](qc_attrition_summary.md) and
[`participant_qc_attrition.csv`](participant_qc_attrition.csv) for condition-
and participant-level retained windows and exclusion reasons. These are
descriptive results under the locked screen, not evidence of biological
validity or clinical qualification.

## Reproduction commands

Run from the repository root with the recorded Python environment and the
ds004148 source data available at `K:/ExternalReceivedDatasets/openneuro/received/ds004148`:

```powershell
python project/mnps2_audit/run_ds004148_v34_baseline_qc.py
python project/mnps2_audit/summarize_ds004148_baseline.py
python mndm/docs/ds004148_baseline_qc/build_attrition_report.py
python mndm/docs/ds004148_baseline_qc/build_reproducibility_snapshot.py
```

The attrition and inventory scripts only read completed outputs; they do not
re-run EEG processing, change the frozen policy, or modify H5 measurements.
