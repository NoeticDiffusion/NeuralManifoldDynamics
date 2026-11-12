# Versioned MNPS measurements and v3.4 reference workflow

Status checked 2026-11-12. This is the entry point for choosing a measurement
contract and interpreting the experimental MNPS 3.4 EEG export. It documents
implemented behavior in this checkout; it does not qualify measurements for a
scientific comparison.

## Documentation inventory and version status

| Resource | Scope and current use |
|---|---|
| [`README.md`](../README.md), [`mndm/README.md`](README.md) | Repository/package entry points. The root README's previous “current release 3.0.2” line did not distinguish package metadata from release notes; it now links here. |
| [`CONFIG_GUIDE.md`](CONFIG_GUIDE.md), [`Command_cheat_sheet.md`](Command_cheat_sheet.md) | General v2/v3 configuration and CLI reference. Their existing recipes remain valid for those workflows; this guide supplies the opt-in v3.4 path. |
| [`Output_variables_guide.md`](Output_variables_guide.md), [`schema_docs.md`](src/mndm/reporting/schema_docs.md) | Existing canonical HDF5 schema, principally MNPS v2 and the v3.0 release line. They are not a complete description of `/measurements/v3_4/`. |
| [`config/v.3.4/ANCHOR_APPLICATION.md`](config/v.3.4/ANCHOR_APPLICATION.md) | Operational selection, compatibility, resume, and failure behavior for frozen external anchors. |
| `src/mndm/mnps/v3_4/` and `config/v.3.4/` | Authoritative v3.4 implementation and version-specific YAML/manifests. |
| `project/diary/20261112_mnps_v3_4_ds005385_condition_specific_cohort_anchor.md` | Cohort selection, fitting, summary inventory, and evidence limits. |

The repository uses several independent version labels:

- The package metadata in `mndm/src/mndm/__about__.py` is **3.1.0**.
- The latest formal release notes present in the root are **3.1.0**, matching
  the package metadata. The older root README's 3.0.2 statement was stale.
- **MNPS 3.4** is an opt-in experimental *measurement definition*. It does not
  change the package version or replace the canonical MNPS v2/v3.0 outputs.
- YAML `schema_version: 1.0-draft`, `definition_version: '3.4'`, anchor IDs,
  and anchor file hashes identify different contracts and artifacts; they are
  not package release numbers.

MNPS v2 remains the canonical/default `[m,d,e]`, optional `coords_9d`, and
optional legacy Jacobian workflow. The v3.1 directory contains a separate
native-sensor measurement family; v3.2 contains experimental protocol code.
Under `src/mndm/mnps/`, this checkout contains `v3_1`, `v3_2`, and `v3_4`;
there is no `v3_3` module there. Do not infer a promoted release sequence from
those directory names.

## Architecture and analysis boundary

1. **Index and ingest** resolve dataset files and sidecars and write a file
   index. Dataset identity, source metadata, and raw file contents are distinct
   evidence: a matching path or sidecar hash does not verify raw signal bytes.
2. **Preprocessing** applies the dataset YAML's filters, reference, resampling,
   time boundaries, and QC policies. These choices define a measurement
   surface and must match the anchor's declared surface for strict application.
3. **Feature extraction** writes per-window feature rows. The v3.4 common EEG
   configuration preserves inherited common EEG features and adds separately
   named alpha-centroid, alpha-envelope-CV, and spatial-logvariance candidates.
4. **Summarize** emits the v3.4 measurement group and may also emit canonical
   MNPS v2 objects in the same HDF5. A named v3.4 external anchor overrides
   config-based anchor selection for the v3.4 measurements. With no explicit
   external selection, the configured `anchor_manifest` or
   `anchor_manifests_by_condition` is used; when no usable manifest is
   configured, raw measurements remain and anchored coordinate values are
   invalid. There is no implicit fit or recording-local normalization.
5. **QC and export** retain window times, per-coordinate support/validity,
   channel/reference realization, anchor selection, implementation hashes,
   and qualification state. A finite value is not by itself evidence of
   comparability or qualification.

Keep three decisions separate: computability belongs to the estimator's
declared inputs and support; comparability depends on the actual sensor set,
reference, time grid, units, and implementation; qualification belongs to the
downstream scientific protocol. `nmd-analysis` owns scientific contrasts,
null controls, acceptance rules, and biological interpretation.

## MNPS 3.4 flat 9D definition

The payload schema is
`mndm.mnps_v3_4.experimental_flat_9d.v1`, under definition version `3.4`.
Its coordinate order is fixed as follows. “Legacy map” means a source-feature
correspondence in the configuration, not an equivalence claim after anchor
scaling.

| # | Coordinate | Source and implemented formula before coordinate reference scaling | Raw unit | Legacy map |
|---:|---|---|---|---|
| 1 | `delta_theta_level` | `0.5*log10(eeg_delta) + 0.5*log10(eeg_theta)` | `log10(microvolt^2)` | sign-reversed `m_a` only before coordinate scaling with matching source/transforms |
| 2 | `alpha_level` | `log10(eeg_alpha)` | `log10(microvolt^2)` | sign-reversed `m_e` only before coordinate scaling with matching source/transforms |
| 3 | `beta_alpha_ratio` | `eeg_beta_alpha` | dimensionless ratio | `m_o` |
| 4 | `gamma_level` | `log10(eeg_gamma)` | `log10(microvolt^2)` | `d_n` |
| 5 | `hjorth_mobility` | `eeg_hjorth_mobility` | dimensionless per sample | `d_l` |
| 6 | `alpha_theta_ratio` | `eeg_alpha_theta` | dimensionless ratio | `d_s` |
| 7 | `permutation_entropy` | `eeg_permutation_entropy` | dimensionless | `e_e` |
| 8 | `alpha_centroid_hz` | `mnps_v3_4_alpha_centroid_hz` | Hz | `paf_cog` |
| 9 | `alpha_envelope_cv` | `mnps_v3_4_alpha_envelope_cv` | dimensionless | `burst_alpha` |

The v3.4 representation is **flat**. It has no M/D/E family aggregation, no
3D projection, and no v3.4 Jacobian. It removes the legacy `e_s` and `e_m`
coordinates from this representation; alpha centroid and envelope CV are new
candidate estimators, not evidence that their legacy map labels are
interchangeable measurements. No coordinate is clipped or winsorized. A
nonpositive input to a log transform is unsupported; the pipeline does not
replace it with an epsilon or floor.

### Feature extraction and anchoring order

The inherited global EEG features use a per-sample median across the available
EEG channels, then the common window calculations: multitaper bandpowers and
ratios, Hjorth mobility, and normalized order-5/delay-1 permutation entropy.
The common extractor falls back to Welch only when multitaper is unavailable.
If the entropy extractor reports degraded fallback mode, its raw column is
retained but `permutation_entropy` is unsupported.

The new candidates are separate computations on the native channel realization:

- `alpha_centroid_hz`: a 7–13 Hz power-weighted centroid of the median across
  channel Welch PSDs. The implementation uses 4 s Welch segments, 50% overlap,
  a periodic Hann window, constant detrending, and per-channel PSD calculation.
  Per-channel centroid values and support are also retained.
- `alpha_envelope_cv`: per-channel standard deviation divided by mean of the
  Hilbert magnitude after a separate order-4, zero-phase 8–12 Hz Butterworth
  filter; the reported recording-window candidate is the channel median.
  Filtering is segment-aware and a symmetric 10 s source-segment edge guard is
  required. Concatenated epochs without continuous boundaries do not establish
  this candidate's support.
- `spatial_logvariance_dispersion`: an auxiliary, non-coordinate candidate;
  `1.4826 * MAD(log10(channel variance))`. It is not included in the 9D map.

For anchored coordinates, the implementation transforms the source features
first (log10 only for the stated power-level inputs), applies frozen robust
feature centers/scales, assembles the nine coordinates with the fixed
coefficients above, then applies separate frozen coordinate centers/scales.
The anchors use participant-balanced medians and `1.4826 * MAD` scales; each
participant's finite supported windows share that participant's total weight.
Feature and coordinate anchors are distinct. Degenerate scales invalidate the
affected values; no epsilon scale is substituted. `coordinate_weights` in the
manifest are all 1.0 metadata; they do not aggregate axes into M/D/E families.

The common YAML also contains geometry defaults (`weighted_euclidean`, unit
weights, no whitening, full-space invalidation) explicitly marked as neutral
design defaults pending validation. The v3.4 coordinate exporter does not
compute distances, trajectories, Jacobians, or these geometry outputs. Any
downstream geometry or aggregation is a separately declared analysis choice,
not a v3.4 subject-level default.

## Frozen ds005385 references and anchor selection

The registered presets were fitted from session-1, pre-task ds005385 EEG:

| Preset | Anchor ID / reference | Participants | Registered file SHA-256 | Embedded `anchor_hash` |
|---|---|---:|---|---|
| `ds005385_ec_v1` | `ds005385_ses1_pretask_EyesClosed_20261111_v1` / EyesClosed | 608 | `79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab` | `ecbc9b1ffb5cf64fd0c094280f3d7f6c6543690a518d80752a4fb6be1e134fd6` |
| `ds005385_eo_v1` | `ds005385_ses1_pretask_EyesOpen_20261111_v1` / EyesOpen | 608 | `f3bb94310778e91b046fbf77f0540ce860af6a65d4607e1e8593c01f800705b1` | `f571d278f1de9863dc1d71ef974c2728427560e8b4b67363e56fe2c2ad646f1f` |

The CLI preset is independent of target condition: either reference can be
applied to EO or EC targets. EO and EC anchors define **different coordinate
frames** because their feature and coordinate centers/scales differ. Use the
same preset for every target when a shared anchored frame is required. The
reference median vector is a frozen technical centering vector, not a validated
normative centroid or population norm.

Strict compatibility is the default for the explicit external-anchor CLI.
Strict mode checks the v3.4 definition and estimator identity, feature order
and transforms, declared units, source/preprocessing surface, channel-set and
reference identity. A channel-order difference is recorded as QC and does not
fail compatibility by itself. Explicit `exploratory_transfer` requires a
human-readable reason and records known mismatches and overlap status. It can
authorize certain surface mismatches, but does not bypass schema/definition,
implementation-hash, invalid-scale, or unresolved-unit blockers. No unit
conversion or modality harmonization occurs. Scalp-to-iEEG transfer is not
validated harmonization.

Each run writes `mnps_v3_4_anchor_selection.json`. Resume identity binds the
selected anchor ID and file SHA, compatibility mode, target measurement
surface, and reference units. A changed selection requires a separate run.
When an application cannot be established, the exporter retains raw values,
support, and QC, marks `unsupported_anchor_application`, sets anchored
coordinates invalid, and records a reason. Partial coordinate support remains
visible; `complete_9d_mask` marks only windows with all nine valid coordinates.
The payload stays `qualified=false` even when all nine values are finite.

Detailed CLI/config semantics and custom manifest behavior are in
[`ANCHOR_APPLICATION.md`](config/v.3.4/ANCHOR_APPLICATION.md).

## Reference cohort, output inventory, and qualification

The frozen cohort selection is session 1, `acq-pre`, with one EyesClosed and
one EyesOpen recording for each of 608 participants (1,216 recordings).
Selection used source metadata and required sidecars, not signal outcomes. The
run used 64 EEG channels, average reference, 250 Hz realized sampling, 8 s
windows with 4 s steps, and a 10 s envelope-estimator edge guard. The reference
fit is condition-specific, participant-balanced, and transductive: the same
participants define and receive the reference. Anchor fitting does not
establish artifact QC; `qc_ok_eeg=-1` means QC was not assessed. These remain
experimental technical references, not qualified population norms.

The cohort statistics were computed from existing HDF5 output under
[`project/mnps2_audit/outputs/20261111_mnps_v3_4_ds005385_baseline_cohort/statistics/`](../project/mnps2_audit/outputs/20261111_mnps_v3_4_ds005385_baseline_cohort/statistics/):

- `cohort_measurement_statistics.csv` reports, by condition and full HDF5
  measurement path/realization, mean, population SD, median, MAD, scaled MAD,
  Q1, Q3, IQR, min, max, skewness, excess kurtosis, and support/participant
  counts.
- `cohort_measurement_inventory.csv` gives the available value counts and
  support scope; `cohort_qc_support_counts.csv` carries window-level QC/support
  summaries.
- `cohort_h5_numeric_dataset_inventory.csv`,
  `cohort_h5_json_measurement_inventory.csv`,
  `cohort_h5_realization_identities.csv`, and `cohort_h5_recordings.csv`
  document numeric HDF5 paths, JSON measurement items, realizations, and
  source-to-HDF5 recording matching. The unclassified numeric dataset report
  is empty for this inventory.

HDF5 arrays intersect an exported validity mask when one is available. The
JSON measurement inventory includes every finite component even when its
`computed`, `valid`, or `qualified` indicator is false; those flags remain in
the separate QC/support inventory. Thus the report is a descriptive inventory,
not scientific acceptance. It includes legacy Jacobian outputs and derived
metrics as separate measurement paths and realizations; they are not MNPS 3.4
Jacobian measurements. `J_hat` retains its legacy local vector-field Jacobian
meaning; `J_dot` is its time derivative. Neither is a Hessian.

All 1,216 H5 identities matched the expected recording manifest through unique
basenames (`unique_manifest_basename_match`); full relative source paths were
not available for those matches. Raw EDF content hashes were not computed in
that cohort run, so H5/source matching does not independently verify raw signal
bytes. Separate EC and EO
anchors also mean their scaled values and pre-reference coordinate surfaces
must not be pooled as if they shared a scale. Raw-feature/raw-coordinate
contrasts still require verified compatible sensor/reference/estimator
realizations and a downstream protocol.

## HDF5 and sidecar contract

In a canonical HDF5, v3.4 is under `/measurements/v3_4/`; it is a separate group
from canonical `/mnps_3d`, `/coords_9d`, and `/jacobian*` outputs. When the
canonical subject summary is skipped but v3.4 is computable, a standalone
`mnps_v3_4_measurements.h5` may contain the same `/measurements/v3_4/` group,
plus root attributes `measurement_only=true`,
`source_summary_status=canonical_summary_skipped`, `skip_reason`, and a
`/recording_identity` group. The sidecar keeps the v3.4 window axis and does
not synthesize v2 MNPS arrays.

Important v3.4 payload members include:

- `raw_features/{values,names,support}` and
  `raw_coordinates/{values,support}`;
- `feature_scaled_values/{values,names}` and
  `coordinate_pre_reference_values`;
- `coordinates/{values,names}`, `coordinate_order`,
  `coordinate_validity_mask`, and `complete_9d_mask`;
- `window_start_sec`, `window_end_sec`, and `support/*` with source identity,
  channel labels/order/set hashes, channel count, reference/reference-set
  identity, and estimator/source provenance;
- `per_channel_candidates/*` for alpha centroid, alpha envelope CV, and
  channel log-variance values, plus `auxiliary/*`;
- `support/qc/*`, the scalar `qualified` and `status` datasets, and
  `provenance/*` with anchor ID/hashes, compatibility mode/status, reference
  and target condition, transfer reason, implementation hashes, and failure
  reason where applicable.

Check the actual HDF5 attributes/datasets and the sidecar/run JSON; do not
assume that an enclosing v2 group has the same validity or coordinate meaning.

## User examples

Run from the repository root in the environment used to install this checkout.
`<dataset_id>`, `<target-v3.4-config.yml>`, and `<path/to/frozen_anchor.json>`
are placeholders that must be replaced with real paths/IDs. No example below
was executed on data during this documentation round.

**Config-selected/default v3.4 anchors** (the ds005385 YAML maps EO and EC
targets to its frozen condition-specific manifests):

```powershell
python -m mndm.cli summarize --config mndm/config/v.3.4/config_ingest_ds005385_baseline_eo_ec.yml --dataset ds005385
```

**Named external preset** (example uses the same EC reference regardless of
the target task label):

```powershell
python -m mndm.cli summarize --config <target-v3.4-config.yml> --dataset <dataset_id> --mnps-v3-4-anchor-preset ds005385_ec_v1
```

**Custom frozen JSON anchor** (path resolves relative to the config when it is
not absolute):

```powershell
python -m mndm.cli summarize --config <target-v3.4-config.yml> --dataset <dataset_id> --mnps-v3-4-anchor-json <path/to/frozen_anchor.json>
```

**Explicit exploratory transfer** (the reason is required and is recorded):

```powershell
python -m mndm.cli summarize --config <target-v3.4-config.yml> --dataset <dataset_id> --mnps-v3-4-anchor-preset ds005385_ec_v1 --mnps-v3-4-anchor-compatibility exploratory_transfer --mnps-v3-4-transfer-reason "Predeclared technical transfer question"
```

If extracting fresh features, first run `python -m mndm.cli features` with the same target
configuration and dataset selection, then run `python -m mndm.cli summarize`. For HDF5
inspection, open `/measurements/v3_4`, read the `coordinate_validity_mask` and
`complete_9d_mask` alongside `coordinates/values`, read `qualified` and
`status`, and inspect `support/qc` and `provenance` for QC, anchor status,
compatibility, and failure reasons. A
strict mismatch can leave the raw groups usable while making the anchored
coordinate group invalid.

## Verification status

| Check | Evidence | What it establishes / does not establish |
|---|---|---|
| v3.4 bounded native-channel pilot | `project/diary/589_20261111_mnps_v3_4_ds004148_native_pilot.md` | Earlier 3-participant, 18-recording run wrote 18 HDF5 exports and 1,332 windows; 1,224 windows were complete 9D. It was transductive/unqualified, with artifact QC not established. It predates the SL-007 external-anchor integration. |
| ds005385 anchor fit and HDF5 statistics inventory | `project/diary/20261112_mnps_v3_4_ds005385_condition_specific_cohort_anchor.md` and the statistics directory above | 608 EO and 608 EC references; all anchors computed, H5 identities matched the selection manifest, and no numeric dataset path was unclassified. Raw EDF hashes and artifact QC were not established. |
| SL-007 focused verification | Reported in the previous task session; a dedicated `sl_007.md` or run log was not found in this M: checkout | 29 focused tests, compilation, YAML loading, CLI help, diff checks, and an independent review were reported passing. No new dataset extraction, summarize run, or anchor refit was reported. The full external-anchor orchestrator HDF5-writing path remains unverified; the broader orchestrator suite did not finish. This is not a biological or predictive validation. |
| Documentation checks in this round | To be recorded in the companion diary after completion | Static paths, configuration parsing, CLI help, manuscript compilation/rendering, and PDF visual inspection only; no dataset execution. |

## Compact unresolved issues and next check

- The exact published Zenodo record and the manuscript/PDF version associated
  with it were not established from repository metadata. This revision is a
  clearly named local draft; it does not replace the historical source/PDF.
- The SL-007 test-count and review report were relayed from the previous task
  session but are not preserved as a dedicated run artifact in this checkout.
- Full external-anchor summarize-to-HDF5 orchestrator integration remains
  unverified; the reference cohort fit and earlier pilot do not cover that
  exact route.

The single highest-value pre-release check is one bounded, fresh-feature
end-to-end pilot through the explicit external-anchor route, followed by an
inspection of its HDF5/sidecar masks and anchor provenance. That check should
not be replaced by broader dataset-scale runs until it passes.
