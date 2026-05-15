# 026 - 2026-05-14 - DANDI README and contract doc update

## Session goal

While PhysioNet download was running, improve repository docs coverage by:

1. adding missing `dandi_ingest/README.md`
2. checking and updating the versioned measurement-contract manuscript where appropriate

## What was changed

- Added `dandi_ingest/README.md` with:
  - purpose and scope (`list`, `download`, `probe`)
  - dependency notes (`dandi`, `h5py`, `pynwb`)
  - config sections and bundled config examples
  - output artifacts and MNDM handoff notes
- Updated root `README.md` "Where To Read Next" to point to `dandi_ingest/README.md`.
- Updated `project/articles/NeuralManifoldDynamics/NeuralManifoldDynamics A Versioned Measurement Contract for Low-Dimensional Neural-Manifold Trajectories.typ` by:
  - adding a new subsection on PhysioNet/WFDB ingest path and `time_reference` clock provenance
  - extending HDF5 naming list with:
    - `/extensions/time_reference/run/*`
    - `/extensions/time_reference/windows/*`
  - clarifying that `run_manifest.json` can expose extension capabilities (including `time_reference`)

## Validation

- Readback checks on edited files completed.
- Lint/diagnostic check returned no issues for the edited docs.

## Evidence class

- **Internal validated result**: Documentation now explicitly covers DANDI ingest usage in-package and includes WFDB/time-reference details in the measurement-contract manuscript.
- **Plausible interpretation**: The added PhysioNet/WFDB + time-reference paragraph in the manuscript aligns with current implementation and output schema semantics.

## Next suggested step

After the current PhysioNet download snapshot stabilizes, rerun sleep-spindle `features` + `summarize` so manuscript/examples and latest artifacts remain synchronized.
