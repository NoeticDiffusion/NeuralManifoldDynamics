## Standard Geometry Invalidity Policy

Date: 2026-06-05

### Goal

Promote mathematically unrealistic MNPS/MNJ handling from optional reviewer QA to a standard export contract.

The practical rule implemented in this session is:

- invalid `mnps_3d` rows are dropped before kNN / Jacobian estimation
- invalid Jacobian windows are removed from canonical export
- invalidity is recorded explicitly in machine-readable provenance
- no clamping is introduced for mathematically unusable geometry

### What changed

#### 1. Always-on geometry contract

Added a new always-on `geometry_contract` surface in the summarize pipeline.

It records:

- `policy_version = "standard_invalidity_v1"`
- retained vs dropped epoch counts on the shared MNPS time grid
- per-surface validity context for `mnps_3d` and `coords_9d`
- invalid Jacobian-window counts for primary and 9D Jacobians

This is exported to:

- `summary.json.geometry_contract`
- `qc_summary.json.geometry_contract`
- HDF5 `/provenance/geometry_contract/*`
- top-level attrs such as `geometry_invalidity_policy` and `geometry_contract_status`

#### 2. Row dropping before kNN / MNJ

The summarize runner now applies a standard hard-invalidity mask after projection and before derivatives / kNN:

- non-finite `mnps_3d` rows are always dropped
- if the active 3D contract is derived from `coords_9d`, non-finite 9D rows also trigger dropping on the shared time grid

This is intentionally separate from reviewer-facing `review_qc` YAML switches.

#### 3. Hard Jacobian-window policy

Primary and 9D Jacobian outputs now pass through a standard post-estimation filter:

- non-finite windows are removed
- windows with condition number above the hard threshold are removed
- invalid centers and counts are written into Jacobian diagnostics

This preserves the canonical export as a mathematically usable surface instead of keeping extreme or unusable windows in place.

#### 4. Documentation

Updated:

- `README.md`
- `mndm/README.md`
- `mndm/Output_variables_guide.md`
- `mndm/src/mndm/reporting/schema_docs.md`
- `mndm/src/mndm/pipeline/run_manifest.py` field guide text

The docs now distinguish:

- canonical `geometry_contract`
- optional reviewer-facing `mnps_mnj_sanity`

### Validation

Targeted tests run:

```powershell
python -m pytest "mndm/tests/test_robustness.py" "mndm/tests/test_dataset_subject_runner.py" -q
```

Result:

- `36 passed`

### Notes

- The hard policy is based on mathematical invalidity, not on statistical outlier thresholds such as “N standard deviations from the mean”.
- `mnps_mnj_sanity` remains useful as richer reviewer QA and robustified sensitivity analysis, but it is no longer the canonical gate for unusable geometry.
