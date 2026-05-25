# MNDM 2.1 Handoff For Analysis Repositories

This note summarizes the **output-contract changes introduced by MNDM 2.1** and what downstream analysis code should do to remain correct when reading new `*.h5`, `summary.json`, and `run_manifest.json` outputs.

The main change is that coordinate anchoring is now **explicit and versioned**. Downstream code should no longer assume that a single unqualified MNPS coordinate tensor is sufficient for every use case.

## Why this changed

Older runs effectively mixed two different use cases:

- **within-subject geometry and dynamics**
- **cross-subject / group comparison**

MNDM 2.1 makes this distinction explicit by exporting separate coordinate layers:

- `subject_anchored`: preserve per-subject/session-relative geometry
- `cohort_anchored`: use a frozen feature anchor for cross-subject/group comparisons

This matters because subject/session normalization can suppress biologically meaningful group differences.

## One-shot cohort anchor workflow

MNDM 2.1 now also supports a **one-shot summarize workflow** in which the anchor
is fit from the merged dataset feature table at summarize startup.

Important semantic point:

- this is still a **fit -> freeze -> apply** workflow
- the anchor artifact is written to the run directory before subject-level H5
  files are emitted
- the anchor is then reused unchanged for all subject outputs in that run
- it is **not** recomputed on-the-fly per downstream comparison

So downstream code should treat one-shot cohort-anchored runs exactly like any
other run that used a frozen external anchor artifact.

## New top-level H5 attributes

Downstream readers should inspect these attrs first:

- `schema_version`
  - `mnps_tensor_spec_v2_1` when MNDM 2.1 explicit coordinate layers or embedded anchors are present
- `mndm_version`
  - `2.1` for anchored-coordinate contract outputs
- `primary_coordinate_layer`
  - usually `coords_3d_cohort_anchored` when an anchor was configured
  - otherwise `coords_3d_subject_anchored`
- `primary_coordinate_contract`
  - `cohort_anchored` or `subject_anchored`
- `anchor_id`
  - stable anchor identifier when a cohort/external anchor was used
- `anchor_hash`
  - stable hash of the anchor artifact used for the run

## New H5 groups that analysis code should support

### Coordinate layers

These are the main new nodes:

- `/coords_3d_subject_anchored/values`
- `/coords_3d_subject_anchored/names`
- `/coords_9d_subject_anchored/values`
- `/coords_9d_subject_anchored/names`
- `/coords_3d_cohort_anchored/values`
- `/coords_3d_cohort_anchored/names`
- `/coords_9d_cohort_anchored/values`
- `/coords_9d_cohort_anchored/names`

Coordinate-layer group attrs include:

- `schema_version = "mndm.coordinate_layer.v2.1"`
- `coordinate_contract = "subject_anchored"` or `"cohort_anchored"`
- `anchor_id`, `anchor_hash`, `anchor_source` for cohort-anchored layers
- `role` for intended usage

### Embedded feature anchor provenance

When cohort anchoring is active, H5 may also contain:

- `/feature_anchors/spec`
- `/feature_anchors/per_feature/feature_name`
- `/feature_anchors/per_feature/center`
- `/feature_anchors/per_feature/scale`
- `/feature_anchors/per_feature/q25`
- `/feature_anchors/per_feature/q50`
- `/feature_anchors/per_feature/q75`
- `/feature_anchors/per_feature/iqr_sigma`
- `/feature_anchors/per_feature/mad_sigma`
- `/feature_anchors/per_feature/qn_sigma`
- `/feature_anchors/per_feature/n_subjects`
- `/feature_anchors/per_feature/n_epochs`

This lets downstream analyses trace exactly which frozen anchor produced the cohort-anchored coordinates.

## Important semantic rule

Do **not** assume that `/jacobian` and `summary.json -> meta_indices` are always tied to subject-anchored coordinates.

In MNDM 2.1 they follow the **primary coordinate contract of the run**.

Practical consequence:

- if `primary_coordinate_contract == "subject_anchored"`, then `/jacobian` is the subject-anchored MNJ
- if `primary_coordinate_contract == "cohort_anchored"`, then `/jacobian` is the cohort-anchored MNJ

The same logic applies to:

- `summary.json -> meta_indices`
- `summary.json -> jacobian`
- `summary.json -> feature_baselines`

For `jacobian_9D`, the exported 9D Jacobian also follows the run's active coordinate contract.

## Legacy paths: still readable, but no longer sufficient

Legacy paths may still exist:

- `/mnps_3d`
- `/mnps_3d_dot`
- `/coords_9d`

These should not be the primary discovery mechanism in new analysis code.

New code should prefer:

1. `primary_coordinate_layer`
2. explicit `coords_*_subject_anchored`
3. explicit `coords_*_cohort_anchored`
4. only then legacy paths as fallback for older runs

## Recommended downstream behavior

### 1. Detect run contract explicitly

On H5 open:

1. read `schema_version`, `mndm_version`
2. read `primary_coordinate_layer`
3. read `primary_coordinate_contract`
4. record `anchor_id`, `anchor_hash` if present

### 2. Separate use cases in analysis code

Recommended defaults:

- **within-subject trajectory geometry / reachability / local-shape analysis**
  - use `coords_3d_subject_anchored`
  - use `coords_9d_subject_anchored`
- **clinical group comparisons / cross-subject statistics**
  - use `coords_3d_cohort_anchored` if available
  - use `coords_9d_cohort_anchored` if available
- **Jacobian/MNJ interpretation**
  - always label results with the run's `primary_coordinate_contract`
  - do not pool subject-anchored and cohort-anchored Jacobians as if they were identical quantities

### 3. Preserve anchor provenance in downstream tables

If the analysis repo produces subject-level CSV/Parquet summaries, add columns such as:

- `primary_coordinate_contract`
- `primary_coordinate_layer`
- `anchor_id`
- `anchor_hash`
- `schema_version`
- `mndm_version`

This is especially important if the same dataset is analyzed in both subject-anchored and cohort-anchored form.

### 4. Prefer capability discovery via `run_manifest.json`

Each run directory now exposes capability flags in `run_manifest.json`, including:

- `feature_anchors`
- `coords_3d_subject_anchored`
- `coords_3d_cohort_anchored`
- `coords_9d_subject_anchored`
- `coords_9d_cohort_anchored`

Downstream repo logic can use this file to avoid probing every H5 first.

## Minimal reader algorithm

Suggested high-level reader logic:

```python
with h5py.File(path, "r") as h5:
    primary_layer = h5.attrs.get("primary_coordinate_layer")
    contract = h5.attrs.get("primary_coordinate_contract")
    anchor_id = h5.attrs.get("anchor_id")
    anchor_hash = h5.attrs.get("anchor_hash")

    def read_layer(name):
        if name in h5 and "values" in h5[name]:
            values = h5[name]["values"][...]
            names = h5[name]["names"][...] if "names" in h5[name] else None
            return values, names
        return None, None

    # Primary 3D layer for run-level interpretation
    if primary_layer and primary_layer in h5:
        x_primary, x_primary_names = read_layer(primary_layer)
    elif "mnps_3d" in h5:
        x_primary = h5["mnps_3d"][...]
        x_primary_names = ["m", "d", "e"]
    else:
        x_primary, x_primary_names = None, None

    # Explicit layers for use-case-specific analysis
    x_subject, subject_names = read_layer("coords_3d_subject_anchored")
    x_cohort, cohort_names = read_layer("coords_3d_cohort_anchored")

    x9_subject, x9_subject_names = read_layer("coords_9d_subject_anchored")
    x9_cohort, x9_cohort_names = read_layer("coords_9d_cohort_anchored")
```

## Recommended interpretation rules

### If the goal is group separation

Prefer:

- `coords_3d_cohort_anchored`
- `coords_9d_cohort_anchored`
- Jacobian outputs from a run whose `primary_coordinate_contract == "cohort_anchored"`

### If the goal is within-subject geometry

Prefer:

- `coords_3d_subject_anchored`
- `coords_9d_subject_anchored`
- Jacobian outputs from a run whose `primary_coordinate_contract == "subject_anchored"`

### If both are present

Do not collapse them into one analysis silently. Treat them as **different measurement contracts**.

## `summary.json` / manifest changes relevant downstream

`summary.json` and embedded `h5.attrs["manifest"]` now carry signals that analyses may want to preserve:

- `meta_indices` and `jacobian`
  - describe the primary 3D Jacobian for the run
- `meta_indices_v2` and `jacobian_9D`
  - describe the primary 9D Jacobian for the run
- `feature_baselines`
  - may now include both local baseline entries and additional `__cohort_anchor` entries
  - these extra entries include `anchor_id`, `anchor_hash`, and `anchor_applied = "external"`
- `mnps_3d.x_definition`
  - may now end in `_cohort_anchored`

## Backward compatibility expectations

Analysis repo code should support three classes of files:

### Older runs

- may only have `/mnps_3d`, `/coords_9d`, `/jacobian`, `/jacobian_9D`
- no explicit anchored layer groups
- no `primary_coordinate_contract`

### MNDM 2.1 subject-anchored runs

- have explicit subject-anchored layers
- may not have cohort-anchored layers
- `feature_anchors` absent
- `primary_coordinate_contract = "subject_anchored"`

### MNDM 2.1 cohort-anchored runs

- have both subject-anchored and cohort-anchored layer groups
- have embedded `feature_anchors`
- `primary_coordinate_contract = "cohort_anchored"`
- may have been created either:
  - from a previously fit anchor JSON
  - or via one-shot summarize-time anchor fitting

In both cases the downstream contract is the same: the run carries a frozen
anchor identity via `anchor_id` and `anchor_hash`.

## Minimal implementation checklist for the analysis repo

- Read and persist `primary_coordinate_contract`
- Support `coords_3d_subject_anchored` and `coords_3d_cohort_anchored`
- Support `coords_9d_subject_anchored` and `coords_9d_cohort_anchored`
- Support `feature_anchors` provenance when present
- Treat `/jacobian` and `/jacobian_9D` as contract-dependent
- Use `run_manifest.json` capability flags when scanning runs
- Keep legacy fallbacks for old H5 files

## Suggested migration summary

If the downstream repo currently assumes:

- one 3D coordinate tensor
- one 9D coordinate tensor
- one Jacobian interpretation

then it should be updated to assume:

- **multiple explicit coordinate layers**
- **contract-aware Jacobian interpretation**
- **anchor provenance as part of the measurement definition**

That is the main conceptual migration required for MNDM 2.1 support.
