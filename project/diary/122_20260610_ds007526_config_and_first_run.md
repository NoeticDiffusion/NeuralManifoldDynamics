# Session Diary 122 - ds007526 config and first run

Date: 2026-06-10

## Goal

Set up a dataset overlay for `ds007526`, preserve the rich scalar participant
metadata in H5 outputs, and run the first full `features + summarize` pipeline.

## Dataset notes

Observed from the downloaded BIDS tree:

- dataset root: `N:/received/ds007526`
- EEG recordings:
  - `144` resting recordings
  - `133` walking recordings
- `participants.tsv` contains rich scalar metadata already suitable for H5:
  - `participant_id`
  - `subject_id`
  - `group`
  - `updrs_part_iii`
  - `updrs_total`
  - `moca`
  - `age`
  - `sex`
  - `disease_duration`
  - `ledd`
  - `pigd_score`
  - `td_score`
  - `ctt`

Additional clinical files were also present under
`sourcedata/clinical_data/`:

- `full_UPDRS_data.tsv`
- `longitudinal_clinical_data.tsv`

These were not merged in the first pass because they are keyed by
`subject_id` values such as `PD0001` / `PDM001` rather than canonical
`participant_id` values like `sub-029`, so they need an explicit join policy or
an intermediate remapping step.

## Config added

Created:

- `mndm/config/config_ingest_ds007526.yaml`

Key choices:

- `received_dir: N:/received`
- `processed_dir: N:/processed/openneuro`
- explicit `dataset_received_dirs.ds007526`
- `eeg_csd.enabled: false`
- `metadata_extraction.datasets.ds007526.participants.path: participants.tsv`
- group normalization:
  - `HC -> Control`
  - `PD -> Parkinson`
- task derived from filename:
  - `rest`
  - `walk`

## Preflight

Ran:

```powershell
$env:PYTHONPATH="H:/SourceRepo2/NeuralManifoldDynamics/mndm/src;H:/SourceRepo2/NeuralManifoldDynamics/core/src;H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/apollo_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/vitaldb_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics"
python -m mndm.cli prerequisite-check --dataset ds007526 --config "H:/SourceRepo2/NeuralManifoldDynamics/mndm/config/config_ingest_ds007526.yaml" --data-dir "N:/received" --out-dir "N:/processed/openneuro"
```

Result:

- `overall_ok: True`
- participants table loaded successfully
- index preview found usable dataset files

## First full run

Started and completed:

```powershell
$env:PYTHONPATH="H:/SourceRepo2/NeuralManifoldDynamics/mndm/src;H:/SourceRepo2/NeuralManifoldDynamics/core/src;H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/apollo_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/vitaldb_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics"
python -m mndm.cli all --dataset ds007526 --config "H:/SourceRepo2/NeuralManifoldDynamics/mndm/config/config_ingest_ds007526.yaml" --data-dir "N:/received" --out-dir "N:/processed/openneuro" --n-jobs 4
```

Observed outcome:

- `features` completed successfully
- feature tables written:
  - `N:/processed/openneuro/ds007526/features.csv`
  - `N:/processed/openneuro/ds007526/features.parquet`
- total epochs written: `17109`
- summarize completed successfully with `exit_code: 0`
- run directory:
  - `N:/processed/openneuro/ds007526/neuralmanifolddynamics_ds007526_20260610_092846`

## H5 metadata verification

Inspected:

- `N:/processed/openneuro/ds007526/neuralmanifolddynamics_ds007526_20260610_092846/sub-029_rest/sub-029_rest.h5`

Verified in the H5 `participant` group:

- `row_json` contains the full `participants.tsv` row
- `mapped_json` contains normalized metadata (`group=Parkinson`, `task=rest`)
- `source_json` points back to `N:/received/ds007526/participants.tsv`
- `clinical_json` bundles task + raw participant row + mapped metadata + source

Representative raw participant fields confirmed in H5 for `sub-029`:

- `subject_id = PD0001`
- `group = PD`
- `updrs_part_iii = 6.0`
- `updrs_total = 17.0`
- `age = 70.0`
- `sex = F`
- `disease_duration = 26.0`
- `ledd = 100.0`
- `pigd_score = 0.2`
- `td_score = 0.2`
- `ctt = 58.0`

## Notes

Non-fatal runtime warnings appeared while reading EEGLAB `.set` files:

- MNE preloaded data despite `preload=False`
- boundary-event warnings from the source EEGLAB files

These did not block the run.

## Next step

If desired, add a second-pass metadata policy that maps `subject_id`-keyed
clinical sidecars into canonical `participant_id` so the item-level UPDRS and
longitudinal follow-up tables can also be embedded in H5 outputs.
