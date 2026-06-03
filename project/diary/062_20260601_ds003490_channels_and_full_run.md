# ds003490 channels and full run

Date: 2026-06-01

## Question

Inspect the actual EEG channel set for `ds003490`, update YAML if needed, then run
`features` and `summarize`.

## Channel audit

Sample `*_channels.tsv` and `*_electrodes.tsv` inspection showed a stable layout:

- 64 scalp EEG channels
- `VEOG`
- `X`, `Y`, `Z`

Important details:

- `VEOG` already matches the common EEG channel-typing fallback rule and is treated
  as `eog`.
- `X/Y/Z` appear in both `channels.tsv` and `electrodes.tsv` with no usable EEG
  metadata, and the dataset note says they come from an accelerometer attached to the
  tremor-affected hand.
- The `eeg.json` sidecar reports `EEGChannelCount: 64`, which matches the scalp set.

## Config updates

Updated `mndm/config/config_ingest_ds003490.yaml` to add dataset-specific channel
typing:

- `X|Y|Z -> misc`

This prevented those accelerometer axes from being treated as EEG channels.

I also disabled:

- `preprocess.eeg_csd.enabled: false`

Reason:

- even after `X/Y/Z` were excluded from the EEG channel set, CSD still failed
  systematically on smoke runs with geometry/sphere-fit issues
- practical runtime behavior before the change was already "continue without CSD"
- disabling it makes that contract explicit and removes repeated warning/overhead

## Validation

Fresh smoke run on `sub-002` showed:

- `Pre-resample channel prune ... keeping 64/67 channels`
- `Applied EOG regression to 63 EEG channels using 1 EOG predictors`

This is the key confirmation that `X/Y/Z` no longer contaminate the EEG path.

## Full runs

### Features

Ran full dataset features:

```powershell
python -m mndm.cli features --dataset ds003490 --config mndm/config/config_ingest_ds003490.yaml --n-jobs 4
```

Result:

- completed successfully
- wrote:
  - `M:/datasets/processed/openneuro/ds003490/features.csv`
  - `M:/datasets/processed/openneuro/ds003490/features.parquet`
- final table size: `11373` epochs

Observed runtime note:

- the raw files contain BIDS/EEGLAB `boundary` events, so MNE warns about data
  discontinuities during preprocessing

### Summarize

Started full summarize:

```powershell
python -m mndm.cli summarize --dataset ds003490 --config mndm/config/config_ingest_ds003490.yaml --n-jobs 4
```

It began writing outputs under:

- `M:/datasets/processed/openneuro/ds003490/neuralmanifolddynamics_ds003490_20260601_082415/`

Observed during runtime:

- `normalization_report.json` and `features_snapshot.json` were written
- subject output folders/H5 files were being written successfully
- regional MNPS and regional block Jacobian CSVs were being appended
- repeated `FD censoring skipped: framewise_displacement column missing` info logs
- some sessions emitted:
  - `CRITICAL WARNING: 9D MNPS falsified for this session. Matrix is effectively 3-dimensional.`

That last message looks like a scientific/conditioning warning rather than a crash; it
appeared during summarize while outputs were still being written.

## Evidence category

- Internal validated result:
  - actual dataset channel set is `64 scalp EEG + VEOG + X/Y/Z`
  - `X/Y/Z` now excluded from EEG via dataset-specific config
  - full `features` run completed successfully
  - full `summarize` run started successfully and was actively producing outputs

## Next useful step

After summarize finishes, inspect:

- run-level `run_manifest.json`
- `stage_mapping_qc.json`
- a few `summary.json` files

to confirm the EO/EC stage surface and oddball event provenance look as intended in
the final summarized artifacts.
