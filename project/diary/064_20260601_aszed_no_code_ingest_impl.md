# ASZED no-code ingest implementation

Date: 2026-06-01

## Question

Can the Zenodo `ASZED` EEG release be brought into MNDM without adding new MNDM
code, by staging the custom EDF tree into a BIDS-lite layout plus config-only
dataset overlays?

Raw source examined:

- `M:/datasets/received/Zenodo/ASZED/ASZED/version_1.1/node_1`

## Decision

Yes, with an external staging layer and three dataset IDs:

- `aszed_subset1`
- `aszed_subset2`
- `aszed_subset3`

The no-code contract implemented here is:

- stage each raw `Phase N.edf` into a BIDS-lite filename carrying `sub`, `ses`,
  `task`, and `run`
- keep provenance and flattened metadata in TSV manifests
- point MNDM config overlays at the staged roots
- validate with `prerequisite-check`, then subject-scoped `features` and
  `summarize`

## Staging layout

Created external staged roots:

- `M:/datasets/received/Zenodo/ASZED_staged/version_1.1/aszed_subset1`
- `M:/datasets/received/Zenodo/ASZED_staged/version_1.1/aszed_subset2`
- `M:/datasets/received/Zenodo/ASZED_staged/version_1.1/aszed_subset3`

Per dataset root:

- `participants.tsv`
- `sessions.tsv`
- `phase_manifest.tsv`
- `dataset_description.json`

Cross-dataset manifest:

- `M:/datasets/received/Zenodo/ASZED_staged/version_1.1/all_phase_manifest.tsv`

EDF counts staged:

- `aszed_subset1`: 783
- `aszed_subset2`: 980
- `aszed_subset3`: 169

Implemented phase-to-task mapping:

- `subset_1`: `restpre`, `arithmetic`, `restpost`, `oddball`
- `subset_2` / `subset_3`: `restpre`, `fixedauditory`, `arithmetic`, `restpost`,
  `oddball`

## Repo changes

Added config-only overlays:

- `mndm/config/config_ingest_aszed_subset1.yaml`
- `mndm/config/config_ingest_aszed_subset2.yaml`
- `mndm/config/config_ingest_aszed_subset3.yaml`

Important config choices:

- `task` comes from staged filenames and is normalized to readable labels such as
  `rest_pre` and `fixed_auditory`
- `Group` / `category` are normalized to `Schizophrenia` and `Control`
- subset-specific channel typing handles:
  - `EOGR` / `EOOGL` as `eog`
  - `EMG` as `emg`
  - `EEG A2-LE` as `misc`
- dataset-specific ensemble channel groups use the real EDF channel labels
- coverage thresholds were lowered to `min_seconds: 20`, `min_epochs: 5`,
  `min_trials: 1` because many ASZED files are short

## Validation

Preflight:

```powershell
python -m mndm.cli prerequisite-check --dataset aszed_subset1 --config mndm/config/config_ingest_aszed_subset1.yaml
python -m mndm.cli prerequisite-check --dataset aszed_subset2 --config mndm/config/config_ingest_aszed_subset2.yaml
python -m mndm.cli prerequisite-check --dataset aszed_subset3 --config mndm/config/config_ingest_aszed_subset3.yaml
```

All three returned `overall_ok: True`.

Subject-scoped smoke runs completed:

- `aszed_subset1`:
  - `features --subject 010`
  - `summarize --subject 010`
- `aszed_subset2`:
  - `features --subject 101`
  - `summarize --subject 101`
- `aszed_subset3`:
  - `features --subject 140`
  - `summarize --subject 140`

Smoke summarize outputs:

- `aszed_subset1`: 4 H5 outputs
- `aszed_subset2`: 1 H5 output
- `aszed_subset3`: 2 H5 outputs

Run directories:

- `M:/datasets/processed/Zenodo/aszed/aszed_subset1/neuralmanifolddynamics_aszed_subset1_20260601_100008`
- `M:/datasets/processed/Zenodo/aszed/aszed_subset2/neuralmanifolddynamics_aszed_subset2_20260601_100026`
- `M:/datasets/processed/Zenodo/aszed/aszed_subset3/neuralmanifolddynamics_aszed_subset3_20260601_100040`

## Caveat

This is a successful no-code ingest path, but not every phase produces usable
feature windows. Several ASZED recordings are shorter than the default EEG window
expectations, so rest and fixed-auditory phases often drop out during feature
extraction while arithmetic and some oddball segments remain usable.

That is a dataset property, not an MNDM loader failure.

## Evidence category

- Internal validated result:
  - the staged dataset layout is readable by the existing MNDM EEG pipeline
  - config-only onboarding is sufficient for `prerequisite-check`, `features`,
    and `summarize`
  - short-file coverage had to be relaxed to obtain subject-level summarize
    outputs on the smoke runs
