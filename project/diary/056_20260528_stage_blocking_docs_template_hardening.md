# 056_20260528_stage_blocking_docs_template_hardening

## Research question
- Ensure the new event provenance + stage-blocking behavior is documented as a framework capability (not a ds006036 one-off), and make template YAML instructions explicit before broader reruns.

## Documentation updates
- Updated root `README.md`:
  - added `stage_mapping_qc.json` to run sidecars at-a-glance,
  - added a generic `Event Provenance and Stage Blocking` section with config snippet and output contract summary.
- Updated `mndm/README.md`:
  - added capability bullets for generic event provenance and config-driven stage-block inference,
  - added a new configuration section with policy-style YAML example,
  - clarified output directory sidecars (`normalization_report.json`, `run_errors.json`, `stage_mapping_qc.json`),
  - clarified `/events/*` now includes legacy arrays plus columnar event provenance.
- Updated `mndm/Output_variables_guide.md`:
  - added config knob references for stage/event provenance (`prefer_events_stage_in_summary`, `stage_blocking.*`).

## Template YAML hardening
- Updated `mndm/config/config_template.yaml` comments:
  - added a step-by-step stage-blocking setup recipe,
  - documented backward-compatible alias keys for migration safety.
- Updated `mndm/config/eeg_config_ingest_template.yaml`:
  - added explicit `epoching.sampling` guidance,
  - included dataset override example with `stage_columns`, `stage_map`, and `stage_blocking`.

## Outcome
- The new behavior is now framed as reusable framework policy.
- New datasets can enable the same contract by config only (no code fork / no dataset hardcoding).
