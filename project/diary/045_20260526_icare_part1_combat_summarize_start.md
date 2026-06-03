## Session: I-CARE part1 0-12h regional + ComBat summarize start

Date: 2026-05-26

### Question
Run the same I-CARE part1 0-12h regional summarize setup, but with ComBat normalization enabled.

### Config change
- Updated:
  - `mndm/config/config_ingest_physionet_i-care_2_1_part1_0_12h_regional.yaml`
- Added explicit normalization block to avoid inherited `enabled: false` from base config:
  - `normalization.enabled: true`
  - `method: combat`
  - `scope: post_features`
  - `batch_key: hospital`
  - `covariates: [group, age, sex]`
  - `combat` runtime settings:
    - `chunk_size: 24`
    - `min_batch_size: 2`
    - `min_feature_observations: 16`
    - `winsorize_quantiles: [0.005, 0.995]`
  - dataset override:
    - `normalization.datasets.physionet_icare_2_1.enabled: true`

### Command launched
- Working dir: `mndm/`
- Command:
  - `../.venv/Scripts/python.exe -m mndm.cli summarize --dataset physionet_icare_2_1 --config config/config_ingest_physionet_i-care_2_1_part1_0_12h_regional.yaml --n-jobs 16`
- Environment:
  - `PYTHONPATH=H:/SourceRepo2/NeuralManifoldDynamics/mndm/src;H:/SourceRepo2/NeuralManifoldDynamics/core/src`
  - `PYTHONIOENCODING=utf-8`

### Early runtime validation
- Summarize started and passed ComBat stage:
  - `Applied ComBat normalization for physionet_icare_2_1 (rows=2846892/2846892, columns=97/97, batches=5)`
- Run output directory:
  - `E:/Science_Datasets/physionet/processed/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260526_164042`
- Worker fanout:
  - `Using 16 summarize workers for physionet_icare_2_1 (3501 grouped recordings)`

### Notes
- Numpy emitted transient runtime warnings during ComBat fitting (`Degrees of freedom <= 0 for slice` / invalid divide), but normalize step still completed and summarize proceeded.
- Ongoing warnings about degraded stratified 9D coverage (`e_m` non-finite for specific runs) match previously observed behavior in this cohort.

### Completion status
- Run finished successfully (`exit_code: 0`).
- Runtime:
  - `elapsed_ms: 20755145` (~5.77 h)
  - `ended_at: 2026-05-26T22:23:35Z`
- Manifest:
  - `run_manifest.json` written in run dir.
  - `extra.normalization.status: applied`
  - `extra.normalization.rows_harmonized: 2846892/2846892`
  - `extra.normalization.feature_columns_harmonized: 97/97`
  - `extra.normalization.batch_counts: A=1308525, B=464567, D=431032, E=338237, F=304531`
  - `extra.normalization.covariates_used: [age, sex]` (`group` had 0.0 coverage)
- Output counts:
  - `h5: 3103`
  - `summary_json: 3103`
  - `qc_summary_json: 3103`
  - `qc_reliability_json: 3103`
- Error profile:
  - `run_status: completed_with_errors`
  - `run_errors.json` written
  - `errors_total: 381 / groupings_total: 3501`
  - predominant failures remain stratified 9D normalization duplicates (e.g. `m_e -> m_a`), consistent with prior known issue class.
