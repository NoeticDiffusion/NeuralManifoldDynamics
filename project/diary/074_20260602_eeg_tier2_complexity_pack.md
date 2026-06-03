# EEG Tier 2 complexity comparator pack

Date: 2026-06-02

## Question

Can the EEG comparator layer be extended with a small Tier 2 `complexity` pack
that reuses existing per-epoch entropy and Hjorth helpers, while staying
separate from the MNPS projection contract and from the heavier
`ndt_ingest.modalities.eeg.complexity` path?

## Implemented

Extended `conventional_eeg` with a second generic comparator pack:

```yaml
conventional_eeg:
  enabled: true
  packs: ["tier1", "complexity"]
  export:
    per_epoch_columns: true
    summaries: true
  complexity:
    spectral_entropy: true
    permutation_entropy: true
    hjorth_complexity: true
    hjorth_mobility: false
```

The implementation reuses existing helpers already present in
`mndm/src/mndm/features/eeg.py`:

- `_compute_spectral_entropy()`
- `_compute_permutation_entropy()`
- `_compute_hjorth_metrics()`

New per-epoch feature columns:

- `eeg_conventional_complexity_spectral_entropy`
- `eeg_conventional_complexity_permutation_entropy`
- `eeg_conventional_complexity_hjorth_complexity`
- `eeg_conventional_complexity_hjorth_mobility`

The summarize path now supports complexity-only conventional packs and writes
them into the existing conventional comparator surface:

- `summary.json.conventional_eeg`
- `h5.attrs["manifest"] -> conventional_eeg`
- `/extensions/conventional_eeg/*`

## Files changed

- `mndm/src/mndm/features/eeg.py`
- `mndm/src/mndm/pipeline/conventional_summary.py`
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

- `23 passed`

The conventional EEG layer now supports a second generic comparator family for
complexity, with per-epoch feature export and summarize-time family rollups,
without changing MNPS weights or 9D mappings.

## Notes

This Tier 2 rollout intentionally does **not** yet expose the heavier
`antropy`-based windowed metrics from `mndm/src/mndm/features/eeg_complexity.py`
such as:

- sample entropy
- multiscale entropy
- Lempel-Ziv complexity
- Higuchi FD

Those remain available for a later expansion if a broader conventional
complexity pack is wanted.

## Evidence category

- Internal validated result:
  - EEG configs can now opt into a conventional `complexity` pack from YAML
  - the feature table emits `eeg_conventional_complexity_*` columns
  - summarize accepts complexity-only packs and exports them under
    `families.complexity`
  - focused feature and summarize integration tests pass
