# 099 - 2026-06-04 - i-care part1 12-24h orphan cleanup and clean restart

Date: 2026-06-04

## Context

The dedicated 12-24h regional rerun root unexpectedly kept receiving fresh
`features_*.csv` / `features_*.parquet` shards even after the output tree had
been deleted. This indicated that earlier failed restarts had left background
feature workers alive and still writing into the same output path.

Config in use:

- `mndm/config/config_ingest_physionet_i-care_2_1_part1_12_24h_regional_subject_anchor.yaml`

Output root:

- `E:/Science_Datasets/physionet/processed_part1_cpc15_12_24h_regional_subject_anchor`

## Investigation

Confirmed two orphaned feature runs still existed, each with surviving Python
process trees detached from their original shell parents:

- run tree rooted at PID `31124`
- run tree rooted at PID `17284`

Also confirmed that the previously failing record:

- `training/0613/0613_018_020_EEG.{hea,mat}`

had been re-downloaded successfully, matched the expected SHA256 checksums, and
could be read by `wfdb.rdrecord()`.

## Action taken

1. Killed both orphaned process trees with `taskkill /T /F`.
2. Verified no live process still referenced the regional 12-24h output root or
   config path.
3. Deleted the full output tree again to avoid mixing old shards with the new
   attempt.
4. Relaunched a single chained execution:
   - `features`
   - `summarize` only if `features` exits successfully

## Current status

The clean restart was launched successfully and reached normal startup in the
`features` stage:

- requested workers: `4`
- dataset: `physionet_icare_2_1`
- source root: `E:/Science_Datasets/physionet/received/i-care_2_1_part1_cpc15_longitudinal_12_24h/training`

At launch time, free space on `E:` was about `14.5 GB`, so disk pressure remains
the main operational risk for this rerun.
