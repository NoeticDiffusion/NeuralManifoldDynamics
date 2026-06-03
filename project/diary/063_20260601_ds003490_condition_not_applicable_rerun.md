# ds003490 condition not_applicable rerun

Date: 2026-06-01

## Question

Clean up `condition` grouping in `ds003490` after summarize produced:

- `ON`
- `OFF`
- `nan`

for a dataset where controls have no medication manipulation.

## Diagnosis

`participants.tsv` uses values such as:

- `sess1_Med = n/a`
- `sess2_Med = no s2`

for control participants.

The summarize output showed some control runs grouped as `nan`, which is not a useful
semantic condition label.

## Decision

Use:

- `not_applicable`

as the explicit `condition` value for controls.

Rationale:

- clearer than `nan`
- safer than `NA`, which can be reinterpreted as missing by some table readers
- does not conflate `group=Control` with `condition`
- accurately represents "medication state does not apply here"

## Config change

Updated `mndm/config/config_ingest_ds003490.yaml`:

```yaml
metadata_extraction:
  datasets:
    ds003490:
      condition:
        normalize:
          on: "ON"
          off: "OFF"
          "n/a": "not_applicable"
          "no s2": "not_applicable"
          nan: "not_applicable"
          none: "not_applicable"
          null: "not_applicable"
```

## Re-run

Re-ran summarize only:

```powershell
python -m mndm.cli summarize --dataset ds003490 --config mndm/config/config_ingest_ds003490.yaml --n-jobs 4
```

New run directory:

- `M:/datasets/processed/openneuro/ds003490/neuralmanifolddynamics_ds003490_20260601_084947`

## Validation

Confirmed from the new outputs:

- control subject directories now use:
  - `sub-046_not_applicable_rest`
  - `sub-047_not_applicable_rest`
  - etc.
- `run_manifest.json` condition values are now exactly:
  - `OFF`
  - `ON`
  - `not_applicable`
- example `summary.json` now has:
  - `dataset_id: "ds003490:sub-046:not_applicable_rest"`

## Evidence category

- Internal validated result:
  - condition metadata is now semantically explicit and no longer collapses to `nan`
  - summarize rerun completed successfully with `exit_code: 0`
