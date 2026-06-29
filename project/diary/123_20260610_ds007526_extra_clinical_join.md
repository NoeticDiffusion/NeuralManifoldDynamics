# Session Diary 123 - ds007526 extra clinical join

Date: 2026-06-10

## Goal

Extend participant metadata loading so `ds007526` can merge extra clinical TSV
files keyed by `subject_id` rather than canonical `participant_id`, then
re-run summarize so those fields land in H5 outputs.

## Code changes

Updated:

- `mndm/src/mndm/pipeline/extractors.py`

Added generic support for `participants.extra_tables` entries that:

- join via a base-table alias column such as `subject_id`
- keep canonical `participant_id` from the base participants table
- optionally normalize extra-table column names
- optionally prefix extra-table columns to avoid collisions

New config knobs supported on `participants.extra_tables` items:

- `join_to_base_column`
- `base_join_column` (alias)
- `column_normalization`
- `column_prefix`
- existing `subject_id_column` remains supported

Also fixed an existing behavior where omitting `include_columns` on an
`extra_tables` entry unintentionally reduced the table to `participant_id`
only. Now full extra tables are kept by default unless `include_columns` is
explicitly provided.

## Tests

Updated:

- `mndm/tests/test_metadata_extraction.py`

Added a focused regression test covering:

- base participants table with canonical `participant_id`
- extra clinical TSV keyed by `subject_id`
- join via `join_to_base_column: subject_id`
- prefixed / normalized columns surviving the merge

Test result:

- `31 passed`

## ds007526 config update

Updated:

- `mndm/config/config_ingest_ds007526.yaml`

Added:

- `sourcedata/clinical_data/full_UPDRS_data.tsv`
  - keyed by `Subject`
  - joined via base `subject_id`
  - columns normalized to snake_case
  - prefixed with `updrs_item_`
- `sourcedata/clinical_data/longitudinal_clinical_data.tsv`
  - keyed by `subject_id`
  - joined via base `subject_id`
  - columns normalized to snake_case
  - prefixed with `longitudinal_`

## Smoke verification

Loaded `ds007526` participant metadata with the updated config and confirmed:

- participant rows: `144`
- prefixed extra clinical columns: `71`
- representative fields present for `sub-029`:
  - `updrs_item_3_1 = 0.0`
  - `longitudinal_followup_date = 2021-08-16`

## Re-summarize run

Ran:

```powershell
$env:PYTHONPATH="H:/SourceRepo2/NeuralManifoldDynamics/mndm/src;H:/SourceRepo2/NeuralManifoldDynamics/core/src;H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/apollo_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/vitaldb_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics"
python -m mndm.cli summarize --dataset ds007526 --config "H:/SourceRepo2/NeuralManifoldDynamics/mndm/config/config_ingest_ds007526.yaml" --data-dir "N:/received" --out-dir "N:/processed/openneuro" --n-jobs 4
```

Outcome:

- summarize completed successfully with `exit_code: 0`
- new run directory:
  - `N:/processed/openneuro/ds007526/neuralmanifolddynamics_ds007526_20260610_100832`

## H5 verification

Inspected:

- `N:/processed/openneuro/ds007526/neuralmanifolddynamics_ds007526_20260610_100832/sub-029_rest/sub-029_rest.h5`

Confirmed in both `participant/row_json` and `participant/clinical_json`:

- `updrs_item_3_1`
- `longitudinal_followup_date`

Representative values:

- `updrs_item_3_1 = 0.0`
- `longitudinal_followup_date = 2021-08-16`
