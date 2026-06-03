# 058_20260528_ds006036_followup_qc_diagnosis

## Research question
- Diagnose remaining downstream concerns after ds006036 rerun:
  - high unlabeled (`-1`) stage window fraction,
  - sparse 25/30 Hz support,
  - unexpected 3/7 Hz detection in stage mapping QC aggregate,
  - regional surface limited to three networks.

## Findings
- Stage unlabeled fraction remains high by design for many runs:
  - mean labeled window fraction across subjects: `0.437` (thus unlabeled ~`0.563`).
  - coverage varies strongly by subject (`min=0.111`, `max=0.941`), with low-coverage subjects often having long runs and sparse stage-carrier intervals.
  - unmapped event labels are dominated by non-stage protocol/artifact labels (`Swallowing`, `speech`, `head movement`, calibration labels, encoding-noise labels).
- 25/30 Hz sparsity is data-driven:
  - subjects with 25 Hz: `6` (`sub-005`, `sub-006`, `sub-039`, `sub-043`, `sub-054`, `sub-060`)
  - subjects with 30 Hz: `2` (`sub-039`, `sub-043`)
- 3/7 Hz comes from real raw labels in one subject:
  - `sub-060` contains `PHOTO 3Hz` and `PHOTO 7Hz` in its raw events TSV.
- Regional surface limitation root cause identified:
  - all H5 files expose only `frontal`, `central`, `parietal_occipital`.
  - no `__g_temporal` features are present in `features.parquet`.
  - channel audit across all 88 `.set` files:
    - configured temporal channels (`T7`, `T8`, `TP7`, `TP8`) appear in `0/88`.
    - legacy temporal channels (`T3`, `T4`, `T5`, `T6`) appear in `88/88`.
  - conclusion: temporal group omission is primarily a channel-label mismatch against config, not absence of temporal electrodes.

## Practical implications
- Current stage surface is auditable and much improved for photic blocks, but still conservative (`-1` retained outside mapped/inferred contract).
- Robust group statistics for 25/30 Hz remain limited by subject support.
- Regional temporal outputs require config update + feature recomputation (summarize-only rerun is insufficient because group-suffixed feature columns are generated in features stage).
