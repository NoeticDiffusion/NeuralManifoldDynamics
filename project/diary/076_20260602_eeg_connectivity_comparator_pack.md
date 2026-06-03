# EEG connectivity comparator pack

Date: 2026-06-02

## Question

Can the conventional EEG comparator layer be extended with a generic
`connectivity` pack by reusing the existing synchrony machinery, while keeping
the outputs separate from the MNPS projection contract?

## Implemented

Added a third generic comparator pack under `conventional_eeg`:

```yaml
conventional_eeg:
  enabled: true
  packs: ["tier1", "complexity", "connectivity"]
  connectivity:
    bands:
      - {name: delta, f_low: 1.0, f_high: 4.0}
      - {name: theta, f_low: 4.0, f_high: 8.0}
      - {name: alpha, f_low: 8.0, f_high: 12.0}
      - {name: beta,  f_low: 13.0, f_high: 30.0}
      - {name: gamma, f_low: 30.0, f_high: 45.0}
    windows: {length_sec: 2.0, step_sec: 0.25}
    roi_pairs:
      - {name: FP, channels: ["F3", "P3"]}
      - {name: FB, channels: ["Fz", "POz"]}
      - {name: LR_occipital, channels: ["O1", "O2"]}
    metrics:
      coherence: true
      plv: true
      pli: true
      wpli: true
      dpli: false
      ppc: false
    outputs:
      summary_stats: ["mean", "std"]
```

The implementation reuses the existing synchrony path in
`mndm/src/mndm/features/eeg_sync.py` and renames the emitted keys into the
conventional comparator namespace:

- `eeg_conventional_connectivity_<name>`

Example output columns:

- `eeg_conventional_connectivity_alpha_FP_plv_mean`
- `eeg_conventional_connectivity_alpha_FB_coh_mean`

These are recording-level synchrony summaries broadcast across epochs in the
feature table so the summarize path can roll them up via the existing
`conventional_eeg` summary surface.

## Files changed

- `mndm/src/mndm/features/eeg.py`
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

- `25 passed`

The comparator layer now supports `tier1`, `complexity`, and `connectivity`
packs, with connectivity grouped under `families.connectivity` in the summary
surface.

## Notes

- This rollout reuses the existing synchrony implementation rather than adding a
  new connectivity engine.
- Connectivity values are recording-level summaries broadcast to all epochs,
  unlike the per-epoch Tier 1 and complexity outputs.
- MNPS weights and `mnps_9d` mappings were not changed.

## Evidence category

- Internal validated result:
  - EEG configs can now opt into a conventional `connectivity` pack from YAML
  - the feature table emits `eeg_conventional_connectivity_*` columns
  - summarize exposes these under `families.connectivity`
  - focused feature and summarize integration tests pass
