# 069 - 2026-06-01 - i-care disk-full recovery reroute to G

## Trigger

Part1 ComBat summarize (`config_ingest_physionet_i-care_2_1_part1_0_12h_regional.yaml`) crashed with:

- `OSError: [WinError 112]` (`No space left on device`) under `E:/Science_Datasets/physionet/processed/...`
- terminal exit status reported as `3221225477`

Part2 ComBat run also showed `No space left on device` in worker logs.

## Diagnosis

- `E:` free space was effectively exhausted (near-zero free bytes).
- Existing part1 features table still available and readable:
  - `features.parquet` (~2.27 GB)
  - `features.csv` (~5.62 GB)

## Recovery actions

1. Stopped running part2 process to avoid continued writes on a full disk.
2. Rerouted active i-care runs to `G:` where large free capacity is available.
3. Updated part2 ComBat config output root:
   - `mndm/config/config_ingest_physionet_i-care_2_1_next_140_0_12h_no_regional_subject_anchor_combat.yaml`
   - new `paths.processed_dir`:
     - `G:/Science_Datasets_longtime_storage/processed/physionet_next140_0_12h_no_regional_subject_anchor_combat`
4. Added dedicated part1 ComBat summarize config on `G:`:
   - `mndm/config/config_ingest_physionet_i-care_2_1_part1_0_12h_regional_gdrive.yaml`
5. Copied required part1 inputs from `E:` to `G:`:
   - `features.parquet`
   - `file_index.csv`
6. Relaunched both jobs (6 workers each) with corrected `PYTHONPATH`:
   - Part1: summarize only (ComBat)
   - Part2: all (features + summarize, ComBat)

## Current status

Both rerouted jobs started successfully:

- part1 log: `Summarizing physionet_icare_2_1`
- part2 log: indexed `1900` files, writing outputs under the new `G:` processed root
