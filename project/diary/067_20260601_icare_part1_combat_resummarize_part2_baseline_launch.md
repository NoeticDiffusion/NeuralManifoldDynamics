# 067 - 2026-06-01 - i-care part1 ComBat re-summarize + part2 baseline launch

## Context

After fixing single-feature ComBat instability in summarize, user requested:

1) restart part1 ComBat summarize, and
2) launch part2 with a matching baseline setup,

in parallel with approximately 4-6 workers.

## Config updates

- Added:
  - `mndm/config/config_ingest_physionet_i-care_2_1_next_140_0_12h_no_regional_subject_anchor.yaml`
- Baseline intent for part2 config:
  - ComBat disabled (`normalization.enabled: false`)
  - normalization validation disabled
  - no regional MNPS
  - `mnps_9d.enabled: true`
  - `mnps_9d.block_jacobians.enabled: true`
  - subject anchor (`mnps_projection.anchor.enabled: false`)
  - dedicated processed root:
    - `E:/Science_Datasets/physionet/processed_next140_0_12h_no_regional_subject_anchor`

## Launch commands

- Part1 ComBat summarize re-run (6 workers):
  - `./.venv/Scripts/python.exe -m mndm.cli summarize --dataset physionet_icare_2_1 --config mndm/config/config_ingest_physionet_i-care_2_1_part1_0_12h_regional.yaml --n-jobs 6`
- Part2 baseline features+summarize (6 workers):
  - `./.venv/Scripts/python.exe -m mndm.cli all --dataset physionet_icare_2_1 --config mndm/config/config_ingest_physionet_i-care_2_1_next_140_0_12h_no_regional_subject_anchor.yaml --n-jobs 6`

Both runs were launched in parallel.

## Runtime note

Initial launch attempts failed due to module resolution (`ImportError: cannot import name 'config_loader' from 'core'`).
Resolved by setting:

- `PYTHONPATH=H:/SourceRepo2/NeuralManifoldDynamics/core/src;H:/SourceRepo2/NeuralManifoldDynamics/mndm/src`

After correction, both jobs entered active execution.
