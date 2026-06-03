# conventional EEG docs clarity pass

Date: 2026-06-02

## Question

Can the README, output-schema docs, and EEG config/template comments be made
clearer now that the conventional EEG comparator layer has both `tier1` and
`complexity` packs?

## Implemented

Clarified the conventional EEG comparator documentation in:

- `mndm/README.md`
- `mndm/Output_variables_guide.md`
- `README.md`

Clarified the EEG config/template surfaces in:

- `mndm/config/config_ingest_common_eeg.yaml`
- `mndm/config/eeg_config_ingest_template.yaml`
- `mndm/config/config_template.yaml`

## What changed

Documentation now states more explicitly:

- that `conventional_eeg` is separate from the MNPS projection contract
- that `packs` can be `["tier1"]`, `["complexity"]`, or both
- what each pack is intended to contain
- the naming pattern for emitted columns:
  - `eeg_conventional_relative_*`
  - `eeg_conventional_ratio_*`
  - `eeg_conventional_peak_*`
  - `eeg_conventional_complexity_*`
- where summarize writes the comparator outputs:
  - `summary.json.conventional_eeg`
  - `h5.attrs["manifest"] -> conventional_eeg`
  - `/extensions/conventional_eeg/*`

Template comments now also show:

- what `tier1` covers
- what `complexity` covers
- that dataset-specific overrides can be added under `conventional_eeg.datasets`

## Validation

- `ReadLints` reported no issues on the edited files.

## Evidence category

- Internal validated result:
  - the conventional EEG documentation and EEG template comments are now more
    explicit and easier to use for pack selection and output interpretation
