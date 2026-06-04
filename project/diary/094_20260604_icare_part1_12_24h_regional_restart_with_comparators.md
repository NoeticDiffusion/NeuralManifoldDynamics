# 094 - 2026-06-04 - i-care part1 12-24h regional restart with EEG comparators

Date: 2026-06-04

## Context

After starting the first clean regional 12-24h rerun, user requested adding
conventional EEG comparators before continuing and explicitly restarting from a
clean output tree.

## Action taken

Stopped the active run that used:

- `mndm/config/config_ingest_physionet_i-care_2_1_part1_12_24h_regional_subject_anchor.yaml`

## Config change

Updated the same config to enable conventional EEG comparator exports:

- `conventional_eeg.enabled: true`
- `conventional_eeg.packs: ["tier1", "complexity"]`
- `conventional_eeg.export.per_epoch_columns: true`
- `conventional_eeg.export.summaries: true`

This keeps the run aligned with a reasonable first coma-style comparator layer:

- spectral distribution / slowing
- alpha peak / median frequency / spectral edge
- entropy + Hjorth-style complexity summaries

Regional MNPS remains enabled, while block Jacobians remain disabled.

## Output handling

The dedicated rerun root was cleared before restart:

- `E:/Science_Datasets/physionet/processed_part1_cpc15_12_24h_regional_subject_anchor`

This avoids mixing pre-comparator intermediates with the restarted run.

## Relaunch

Restarted chained execution:

1. `features`
2. `summarize` only if `features` exits successfully

Config:

- `mndm/config/config_ingest_physionet_i-care_2_1_part1_12_24h_regional_subject_anchor.yaml`

Output root:

- `E:/Science_Datasets/physionet/processed_part1_cpc15_12_24h_regional_subject_anchor`
