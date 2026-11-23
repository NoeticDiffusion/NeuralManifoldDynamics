# MNPS 3.4 external reference anchors

For the full coordinate definition, cohort evidence, HDF5 contract, and
version boundaries, see [`../../MNPS_VERSIONED_GUIDE.md`](../../MNPS_VERSIONED_GUIDE.md).

MNPS 3.4 can apply one frozen reference anchor to every target recording in a
summarize run. The selected anchor names its reference cohort and condition;
the target `task`/condition does not select or change the anchor. An
EyesClosed reference may therefore be applied to both EyesClosed and EyesOpen
targets. To use an anchor from the other reference condition, start a separate
summarize run.

The two registered presets are `ds005385_ec_v1` and `ds005385_eo_v1`. Their
JSON assets live beside this document under `anchors/`. The files were copied
byte-for-byte from the frozen ds005385 cohort output and are verified against
the registered SHA-256 before use. The original files remain in the cohort
output directory. Relative preset assets keep the config portable across
checkouts. The registered file hashes are EC
`79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab` and EO
`f3bb94310778e91b046fbf77f0540ce860af6a65d4607e1e8593c01f800705b1`.

## Strict named preset

CLI:

```powershell
python -m mndm.cli summarize --config mndm/config/v.3.4/config_ingest_ds005385_baseline_eo_ec.yml --dataset ds005385 --mnps-v3-4-anchor-preset ds005385_ec_v1
```

The equivalent config selection is:

```yaml
scaling:
  external_anchor:
    preset: ds005385_ec_v1
    path: null
    compatibility: strict
    transfer_reason: null
```

Strict mode requires the target's v3.4 source-surface, channel-set and
reference identities to match the anchor, plus matching declared feature
units, estimator overlay, version, feature order, transforms and coordinate
mapping. Channel order hashes and original labels remain in target provenance;
order differences are QC and do not fail compatibility by themselves.
If strict compatibility is not established, the anchor is marked
`unsupported_anchor_application`; raw features, raw coordinates, support and QC
are still exported, while anchor-scaled coordinates remain invalid.

## Custom JSON anchor

CLI:

```powershell
python -m mndm.cli summarize --config path/to/target.yml --dataset ds-example --mnps-v3-4-anchor-json path/to/frozen_anchor.json
```

Relative custom paths resolve from the selected config file's directory. The
manifest's embedded `anchor_hash` is checked. An expected file SHA-256 can
also be pinned in config:

```yaml
scaling:
  external_anchor:
    preset: null
    path: anchors/my_frozen_anchor.json
    sha256: <expected-file-sha256>
    compatibility: strict
    transfer_reason: null
```

For exploratory transfer with a custom manifest, supply
`reference_feature_units` when those units are not explicit in the manifest.
The target config must declare `measurement_surface.feature_units`. Unit
conversion is not performed; unresolved or differing units leave the anchor
unsupported while computable raw measurements remain in the export.

## Explicit exploratory transfer

Apply one fixed reference to both target conditions in one run, for example:

```powershell
python -m mndm.cli summarize --config path/to/target.yml --dataset ds-example --mnps-v3-4-anchor-preset ds005385_ec_v1 --mnps-v3-4-anchor-compatibility exploratory_transfer --mnps-v3-4-transfer-reason "Exploratory transfer to a predeclared alternate EEG measurement surface"
```

The selected preset is applied to each target recording regardless of its
condition label. The H5 provenance records anchor ID and byte hash, reference
condition/cohort, target identity and condition, strict or transfer mode,
known surface mismatches, missing compatibility metadata, channel-order QC,
participant overlap, support/QC and the transfer reason. The result remains
experimental, measurement-incompatible and unqualified. A transfer option
does not harmonize scalp EEG and iEEG or establish comparability.

Read `anchor_application_status` for whether the target transform was
computed. After successful application, `evaluation_status` and
`reference_evaluation_status` describe the reference anchor's evaluation
status; they do not qualify the target transfer. Without an applied anchor,
`evaluation_status` may be `not_scaled_no_compatible_anchor`.

Exploratory transfer can authorize measurement-surface mismatches, but it does
not bypass unsupported schemas, altered feature/coordinate definitions,
feature implementation-hash mismatches, invalid scales or unresolved units.
Those cases keep raw measurements and support/QC while marking the anchor
application unsupported. A missing input invalidates the affected coordinate
support while other computable coordinates remain available. No target values
are used to refit the anchor; values are not clipped or imputed. Unsupported
coordinates remain NaN with their support masks intact.

Each run writes `mnps_v3_4_anchor_selection.json`. Resuming with a different
anchor ID, file hash, compatibility mode, target measurement surface or
reference units is refused so one run cannot mix anchor realizations or
compatibility contexts. Use a new summarize run for a different selection.

The registered ds005385 anchors are experimental cohort references. Their
source cohort artifact QC was not established; applying them does not upgrade
them to qualified population norms. The fitted center is a **reference
median vector** followed by the declared robust scaling. The implementation
tests and compatibility checks are not cross-dataset validation.
