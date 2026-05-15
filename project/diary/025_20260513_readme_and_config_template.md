# 025 - 2026-05-13 - README updates and generic config template

## Research question
How should we document the new Time Reference v1 capability and provide a reusable dataset config template that works across EEG, WFDB, and NWB workflows?

## Changes implemented
- Populated `mndm/config/config_template.yaml` (it was empty) with a generic, reusable ingest overlay skeleton.
- Template now includes practical sections distilled from active configs:
  - base import strategy (EEG/NWB/NWB-rodent/NWB-mouse-EEG)
  - `source`, `paths`, `preprocess`, `epoching`, `robustness`
  - optional `time_reference` (WFDB clocks)
  - optional `metadata_extraction` sidecar parsing patterns
  - optional `event_locked`, `pseudo_stage`
  - optional `regional_mnps`, `mnps_9d.block_jacobians`, `ndt_ingest`
- Updated `mndm/README.md`:
  - added Time Reference v1 to capabilities
  - added explicit config-template onboarding section
  - documented Time Reference v1 YAML block
  - expanded HDF5 schema table with:
    - `/window_start`
    - `/window_end`
    - `/extensions/time_reference/run/*`
    - `/extensions/time_reference/windows/*`
- Updated repository root `README.md`:
  - added quick-start copy command for `config_template.yaml`
  - linked template in "Where To Read Next"
  - documented time-reference H5 extensions and run-manifest capability flags

## Validation
- Parsed `mndm/config/config_template.yaml` with `yaml.safe_load` to verify valid YAML.
- Checked edited docs for consistency and path references.
- No linter errors for touched files.

## Notes
- Template is intentionally conservative: core defaults are enabled, advanced blocks are optional and commented for dataset-specific activation.
- Time Reference v1 documentation now aligns README contracts with current implemented output paths and manifest capability probing.
