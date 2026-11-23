# Jacobian outputs and v3.4-only summarize runs

MNPS 3.4 coordinates and Jacobian outputs use separate HDF5 namespaces. The
legacy outputs remain available unless their legacy configuration switches are
disabled explicitly.

## Existing HDF5 identity

In the ds004148 repeatability H5 files, the root manifest identifies the
legacy groups:

- `/jacobian` is the primary 3D MNPS field. The manifest points to it through
  `jacobian_h5.primary_path`; `mnps_axes.names` is `[m, d, e]`.
- `/jacobian_9D` is the legacy nine-subcoordinate field. The manifest points
  to it through `jacobian_h5.primary_v2_path`; `coords_9d.export_prefix` is
  `v2_` and the recorded definition is `subcoords_9d_v2_0`.
- `/jacobian_subject_anchored` and `/jacobian_9D_subject_anchored` are the
  explicitly layered legacy exports. Their HDF5 attributes identify schema
  `mndm.jacobian_layer.v2.1`, `coordinate_contract=subject_anchored`, and the
  corresponding `mnps_3d` or `coords_9d` coordinate space.
- The summary pipeline also defines the optional cohort-anchored paths
  `/jacobian_cohort_anchored` and `/jacobian_9D_cohort_anchored`; the bounded
  suppression check treats these as forbidden paths as well.
- The experimental candidate path is `/measurements/v3_4/jacobian`. It is
  separately controlled and versioned; it was absent in the repeatability
  outputs.

The two legacy primary groups do not carry their own schema attributes. Their
identity is established by the HDF5 path and the root manifest, which also
records the primary and v2 coordinate definitions. Consumers should use those
manifest fields rather than infer the chart from matrix shape alone.

## Configuration

`v3_4_jacobian.enabled: false` disables only the separate experimental v3.4
candidate. A v3.4-only export that also omits the standard legacy fields must
set all three switches:

```yaml
mnps:
  jacobian:
    enabled: false
mnps_9d:
  jacobian:
    enabled: false
v3_4_jacobian:
  enabled: false
```

The example `config_ingest_ds004148_v34_legacy_jacobian_suppression_smoke.yml`
inherits the frozen-EC configuration, keeps the v3.4 9D measurements enabled,
and writes to an isolated output directory. When no primary Jacobian payload is
available, the HDF5 writer now omits `/jacobian` instead of creating an empty
group, and the root manifest reports `primary_path: null`. The corresponding
9D and anchored paths are also absent when their estimators are disabled.

## Bounded verification

On 2026-11-19, the summarize smoke used cached features for `sub-01` and wrote
the four session-2/session-3 EC/EO recordings to a new output root. Every file
retained the frozen ds005385 EC anchor, a `(74, 9)` coordinate-value array, and
its support mask (660/666 coordinate entries valid). Each array and mask
matched the corresponding unsuppressed full-cohort H5 exactly. None contained
any of the six legacy primary/layer paths checked by the verifier, or
`/measurements/v3_4/jacobian`; their HDF5 manifests reported no active Jacobian
paths. The verification script also checked that the older full-cohort outputs
identify their legacy 3D and v2 9D paths through the root manifest and schema
attributes.

See `mndm/docs/ds004148_session_repeatability_fullcohort/artifacts/jacobian_suppression_smoke.json`
for file hashes and exact identities. The 60-participant repeatability H5 files
were not rewritten; they retain the legacy outputs produced by the earlier
configuration and those matrices were not included in the repeatability
analysis.
