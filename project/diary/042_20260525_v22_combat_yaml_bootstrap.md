## Session: v2.2 ComBat config bootstrap

Date: 2026-05-25

### Question
How to start NeuralManifoldDynamics v2.2 normalization rollout with a minimal ComBat-first setup?

### Implemented changes
- Added a new `normalization` control block (ComBat-only pilot) in YAML configs:
  - `mndm/config/config_ingest_common_eeg.yaml`
  - `mndm/config/config_ingest_physionet_i-care_2_1.yaml`
  - `mndm/config/config_ingest_physionet_i-care_2_1_next_140_0_12h_no_regional.yaml`
  - `mndm/config/config_template.yaml`
  - `mndm/config/eeg_config_ingest_template.yaml`
- Keys included:
  - `enabled`
  - `method` (`combat`)
  - `scope` (`pre_features | post_features`)
  - `batch_key`
  - `covariates`
  - `reference_policy` (`within_dataset | frozen_external`)
  - optional `datasets` override map
- Added ComBat dependency:
  - `requirements.txt`: `neuroCombat==0.2.12`
- Installed dependency in project virtualenv:
  - `.venv/Scripts/python.exe -m pip install neuroCombat`

### Validation
- Parsed all modified YAML files successfully with `yaml.safe_load`.
- Verified runtime dependency import in `.venv`:
  - `import neuroCombat` succeeded.

### Notes
- Normalization remains disabled by default in current configs (`enabled: false`) to avoid behavior changes before explicit runtime wiring in summarize/projection paths.
