# coords_9d duplicate tolerance for sub-0435

## Goal

Investigate the remaining `sub-0435` summarize failures after the standard geometry invalidity policy rerun and remove the hard crash caused by duplicated `coords_9d` subcoordinates.

## What I found

- The failing runs were `run-002_acq-018` and `run-032_acq-047`.
- The crash did not come from duplicated subcoordinate names.
- Instead, different 9D subcoordinates became numerically identical after the configured feature standardization and projection.
- For `run-002_acq-018`, `m_e` collapsed exactly onto `m_a`.
- For `run-032_acq-047`, multiple exact duplicate pairs appeared in a 2-epoch run.
- This is a genuine low-rank / degenerate geometry situation, but it should be exported with flags rather than crash the whole subject rerun.

## Code changes

- Extended `mndm.schema._normalize_coords_9d()` with tolerant handling for exact duplicate subcoordinate columns.
- Preserved strict behavior by default; duplicate columns are only allowed when the caller explicitly opts in.
- Added diagnostics for:
  - `duplicate_pairs`
  - `duplicate_count`
  - `duplicate_constant_pairs`
  - `duplicate_constant_count`
- Updated payload normalization to respect duplicate-tolerance attrs and persist the duplicate diagnostics into attrs.
- Updated summarize to:
  - allow duplicate 9D columns in degraded mode,
  - log explicit warnings,
  - export the duplicate pairs/counts in provenance attrs,
  - inject duplicate diagnostics into `geometry_contract.coords_9d`.

## Validation

- `python -m pytest "mndm/tests/test_schema.py" -q`
  - Result: `11 passed`
- Live rerun:
  - `python -m mndm.cli summarize --dataset physionet_icare_2_1 --config "mndm/config/config_ingest_physionet_i-care_2_1.yaml" --subject 0435 --n-jobs 1`
- Outcome:
  - `run-002_acq-018` no longer crashes; it now exports with `geometry_contract.coords_9d.duplicate_pairs = {"m_e": "m_a"}`.
  - `run-032_acq-047` no longer produces a summarize traceback; it is skipped by normal coverage gating (`epochs=2`, below required minimum).

## Notes

- This change does not claim that the affected 9D geometry is biologically meaningful.
- It changes the behavior from "hard failure" to "degraded but explicit export" so cohort reruns are not blocked by exact duplicate subcoordinates.
