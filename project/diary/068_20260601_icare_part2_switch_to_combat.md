# 068 - 2026-06-01 - i-care part2 switch from baseline to ComBat

## Context

User requested stopping the running part2 baseline job and relaunching part2 with ComBat so part1-ComBat and part2-ComBat are directly comparable in the analysis repo.

## Actions

1. Stopped running part2 baseline process (`mndm.cli all`) that used:
   - `mndm/config/config_ingest_physionet_i-care_2_1_next_140_0_12h_no_regional_subject_anchor.yaml`
2. Kept part1 ComBat summarize run active.
3. Added a dedicated part2 ComBat + subject-anchor config:
   - `mndm/config/config_ingest_physionet_i-care_2_1_next_140_0_12h_no_regional_subject_anchor_combat.yaml`
4. Relaunched part2 with 6 workers:
   - `mndm.cli all --dataset physionet_icare_2_1 --config ...subject_anchor_combat.yaml --n-jobs 6`

## Config contract for part2 relaunch

- ComBat: enabled (inherited from `...next_140_0_12h_no_regional.yaml`)
- Regional MNPS: disabled
- 9D MNPS: enabled
- 9D block Jacobians: enabled
- Anchor mode: subject-anchored (`mnps_projection.anchor.enabled: false`)
- Processed root:
  - `E:/Science_Datasets/physionet/processed_next140_0_12h_no_regional_subject_anchor_combat`

## Runtime note

The earlier import issue was already resolved by launching with:

- `PYTHONPATH=H:/SourceRepo2/NeuralManifoldDynamics/core/src;H:/SourceRepo2/NeuralManifoldDynamics/mndm/src`

Both active jobs continue under that environment setup.
