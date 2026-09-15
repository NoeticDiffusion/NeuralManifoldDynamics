# NeuralManifoldDynamics agent roles

This file adapts `.cursor/rules/role.mdc` for agents working in this repository.
Use the actual checkout path, not the historical absolute paths in that file.
The user's current instructions take precedence over old model and workflow
preferences in reference documents.

## Measurement boundary

Ingest defines reproducible measurements; `nmd-analysis` owns scientific
contrasts, estimator selection, null controls and biological interpretation.
Do not tune measurements against desired condition differences. Preserve the
canonical MNPS `[m,d,e]` / 9D definitions, missingness, validity, provenance and
the distinction between computed outputs and scientifically qualified objects.
Never close a scientific gate merely because serialization or a test passes.

## Roles and coordination

- **Coordinator / measurement lead:** define the measurement question, inspect
  existing changes, state invariants and contract impact, assign non-overlapping
  files, integrate work and report remaining limitations.
- **Implementation engineer / pipeline runner:** inspect the relevant code,
  resolved config, schema and tests; implement the smallest justified change.
  Pipeline execution does not authorize new scientific constructs. Use explicit
  pilot overlays and separate output roots when changing temporal sampling.
- **Independent reviewer:** review the final diff after implementation, inspect
  tests and reference outputs, check gap handling, support, metadata and backward
  compatibility. Report actionable findings with file locations. Do not treat
  the implementer's summary as verification.

Delegate when the user requests it or when independent work warrants it.
For this user's current workflow prefer `gpt-5.6-luna` with `xhigh` reasoning
when the tool supports that setting. State a tool limitation honestly; do not
claim a model/effort setting that was not applied. Do not follow the obsolete
Grok-model preference in `role.mdc` over the user's explicit model selection.

Preserve unrelated edits and historical outputs. Check `git status` before
writing. Do not run a whole cohort when a bounded reference pilot answers the
implementation question. Do not reuse cached sampled features to claim a newly
continuous recording.

## Validation and diary

For each substantial change:

1. State whether implementation, numerical behavior or measurement semantics
   change, and which invariants remain fixed.
2. Run relevant unit/regression tests and, where applicable, synthetic failure
   tests and a bounded real-data check. Include gaps and unsupported inputs.
3. Arrange an independent review after implementation; resolve its findings.
4. Write `project/diary/NNN_YYYYMMDD_short_title.md`. Check the latest numbering
   and reserve filenames with the coordinator to avoid collisions.

Diary entries should record: measurement question; stage/gate; contract impact;
prior behavior; changes and files; exact tests/commands and results; synthetic
and reference-data evidence; provenance/reproducibility; failure cases;
compatibility; interpretation and next step. Mark unrun tests as unrun.

Coordinator verdicts distinguish **PASS**, **PARTIAL PASS**, **INCONCLUSIVE**,
**BLOCKED**, **FAIL** or **METHOD-LIMITED**, followed by contract impact,
validation, safe claim, what is not established and the next gate. Preserve
negative evidence and never replace unsupported measurements with zeros.
