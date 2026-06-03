# EEG Tier 1 conventional comparator

Date: 2026-06-02

## Question

Can the EEG pipeline expose a first-pass, YAML-driven conventional qEEG
comparator layer beside MNPS, so that standard literature-facing features can be
exported without changing the MNPS projection contract?

## Implemented

Added a new EEG-only config surface:

```yaml
conventional_eeg:
  enabled: true
  packs: ["tier1"]
  export:
    per_epoch_columns: true
    summaries: true
  tier1:
    relative_bandpower: true
    ratios: ["theta_alpha", "delta_alpha", "alpha_theta", "beta_alpha", "slowing_index"]
    peak_frequency:
      alpha_peak_frequency: true
      median_frequency: true
      spectral_edge_95: true
```

The first rollout computes the following per-epoch feature columns during EEG
feature extraction:

- `eeg_conventional_relative_<band>`
- `eeg_conventional_ratio_<name>`
- `eeg_conventional_peak_<name>`

Current Tier 1 metrics include:

- relative delta/theta/alpha/beta/gamma
- `theta_alpha`
- `delta_alpha`
- `alpha_theta`
- `beta_alpha`
- `slowing_index = (delta + theta) / (alpha + beta)`
- `alpha_peak_frequency`
- `median_frequency`
- `spectral_edge_95`

Summarize now also emits a separate conventional comparator block:

- `summary.json.conventional_eeg`
- `h5.attrs["manifest"] -> conventional_eeg`
- `/extensions/conventional_eeg/*`

This remains explicitly separate from:

- MNPS weights
- Stratified MNPS mappings
- Jacobian estimation
- existing E-Kappa / RFM / O-Koh / TIG extensions

## Files changed

- `mndm/src/mndm/features/eeg.py`
- `mndm/src/mndm/pipeline/conventional_summary.py`
- `mndm/src/mndm/pipeline/summary.py`
- `mndm/config/config_ingest_common_eeg.yaml`
- `mndm/config/eeg_config_ingest_template.yaml`
- `mndm/config/config_template.yaml`
- `mndm/tests/test_features_eeg.py`
- `mndm/tests/test_dataset_subject_runner.py`
- `mndm/README.md`
- `mndm/Output_variables_guide.md`
- `README.md`

## Commands

```powershell
python -m pytest mndm/tests/test_features_eeg.py mndm/tests/test_dataset_subject_runner.py
```

## Result

Targeted tests passed:

- `21 passed`

The implementation now provides a config-driven EEG comparator layer that is
usable in ordinary dataset overlays while keeping the current MNPS measurement
contract unchanged.

## Evidence category

- Internal validated result:
  - Tier 1 conventional EEG comparator features are now configurable from YAML
  - the EEG feature pipeline writes the new `eeg_conventional_*` columns
  - summarize exports a separate conventional comparator summary block
  - focused feature and summarize integration tests pass
