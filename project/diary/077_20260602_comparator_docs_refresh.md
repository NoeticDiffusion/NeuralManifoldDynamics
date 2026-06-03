# comparator docs refresh

Date: 2026-06-02

## Question

Can the markdown documentation be refreshed so that the README/output docs match
the actual conventional EEG comparator surface now present in the pipeline?

## Implemented

Updated these markdown files:

- `mndm/README.md`
- `README.md`
- `mndm/Output_variables_guide.md`
- `project/ideas/sidecar_analysis/sidecar_analysis.md`

## What changed

Documentation now more clearly states that the current conventional EEG
comparator surface includes:

- `tier1`
- `complexity`
- `connectivity`

Specific clarity updates:

- `mndm/README.md`
  - example YAML now includes the `connectivity` pack
  - added a connectivity-only usage example
  - documented typical connectivity output names
  - clarified that connectivity is currently a recording-level summary surface
    broadcast across epochs
- `README.md`
  - corrected wording from “one or both” to “one or more”
  - added connectivity to the high-level EEG comparator description
- `mndm/Output_variables_guide.md`
  - added `connectivity` to the listed current packs
  - documented the granularity distinction between epoch-aligned packs and
    broadcast recording-level connectivity summaries
- `project/ideas/sidecar_analysis/sidecar_analysis.md`
  - added an implementation-status section so the design note better reflects
    what is already in the repo vs what still remains future work

## Validation

- `ReadLints` reported no issues on the edited files.

## Evidence category

- Internal validated result:
  - comparator-facing markdown docs are now more consistent with the current
    implementation state of the EEG pipeline
