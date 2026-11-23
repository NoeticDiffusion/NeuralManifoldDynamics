# Experimental MEG MNPS 3.4 measurement surface

**Definition:** `3.4-meg-exp.1`  
**Schema:** `mndm.meg_mnps_v3_4.experimental.v1`  
**Status:** experimental, raw and unscaled, unqualified.

This surface is emitted beside the existing MEG shadow features. It does not
replace the legacy projection or alter its archived feature values. The
existing-coordinate experimental baseline is a corrected realization: it
keeps the inherited coordinates but keeps unsupported SampEn and permutation
entropy missing, records the realized PSD backend, and retains MAG/GRAD as
separate sensor families. The existing legacy config and projection remain
available for exact historical replay.

## Sensor and feature rules

- Only explicit `signals.meg_mag` and `signals.meg_grad` inputs are eligible.
  A combined `signals.meg` array without explicit family metadata is emitted
  with `unknown_sensor_type`; it is not relabeled GRAD in this surface.
- The seven inherited features use the current median-waveform computation
  within each family. This definition is deliberately fixed for this
  comparison.
- The new alpha candidates operate channel by channel, then aggregate over
  channels of the same family. Their channel labels, ordered and set hashes,
  per-window support counts, values, and failure reasons are exported.
- No global MAG/GRAD combination is produced. Values for the two families
  remain separate.
- Raw values are exported without anchors or recording-local z-scoring. An
  EEG anchor is not accepted. Any future scaling requires a MEG-specific
  calibration subset and a frozen MEG anchor manifest.

## Candidate definitions

| Feature | Definition | Failure/support rule |
|---|---|---|
| `alpha_centroid_hz` | Welch PSD per channel; 4 s Hann segments, 50% overlap, constant detrend and density scaling. Take the median channel PSD over 7–13 Hz, then its power-weighted centroid. | Use channels finite over the complete window and list exactly which were used. Missing labels, no finite channels, absent alpha bins, nonpositive PSD mass, and Welch errors remain explicit statuses. |
| `alpha_envelope_cv` | Fourth-order Butterworth SOS zero-phase 8–12 Hz filter and Hilbert magnitude on each channel's continuous finite run. Calculate `SD(envelope)/mean(envelope)` within the window and take the median over channels. | Windows must be contained in a continuous run and at least 10 s from that run's edges. Unsupported channels are listed; if no channels meet the rule, return missing with a reason. |

The inherited SampEn uses the existing `m=2`, `r=0.2 × SD`, maximum 300-point
subsampling definition. A failed SampEn is missing with a reason. The
permutation entropy coordinate requires a realized permutation-entropy metric
and is missing in degraded mode; gamma power is never substituted.

## Selectable coordinate variants

The YAML `mnps_v3_4_meg.variants` list selects one or more variants. The
bounded pilot selects all three and exports them from one preprocessing run:

| Variant id | Coordinates | Meaning |
|---|---|---|
| `existing_coordinate_set` | 9: seven inherited plus Hjorth complexity and native SampEn | Corrected experimental comparison baseline |
| `removal_only` | 7 inherited coordinates | Descriptive removal comparison; dimension remains seven |
| `alpha_replacement` | Seven inherited plus `alpha_centroid_hz` and `alpha_envelope_cv` | MEG-specific replacement candidate |

The coordinate values shared by variants are calculated once and sliced into
each variant, so they retain identical raw scales. Each family has its own
coordinate names, units, support masks, and failure reasons. `complete_window_mask`
is descriptive support and does not qualify a scientific use.

## Export

The standard summary writer places this payload under
`/measurements/v3_4_meg/`; a measurement-only sidecar uses
`mnps_v3_4_meg_measurements.h5`. Each export contains `variant_definitions`,
`window_axis`, separate `sensor_families/mag` and `sensor_families/grad`
groups, raw feature values, raw coordinate variants, support/failure fields,
configuration hash, estimator settings, and implementation hashes. The
historical `v3_4` EEG key and sidecar are unchanged.

### Ordinary summarize smoke

On 2026-11-18, an isolated ordinary `summarize` run wrote a full subject HDF5
for pilot recording `sub-002`, with this surface embedded under
`/measurements/v3_4_meg/`. The input table was reconstructed from the existing
pilot measurement HDF5; no MEG raw data were reprocessed. The MAG/GRAD payloads
matched the pilot export across 173 checked fields, including values, units,
support masks, estimator IDs, and failure reasons. A control run with the
experimental export disabled produced identical canonical 9D/3D coordinates,
time fields, and exported feature matrices. The candidate coordinates remain
auxiliary, and the baseline ds003645 config does not enable this experimental
overlay.

The pilot and summarize run have matching `meg.py` and `meg_candidates.py`
implementation hashes. `meg_export.py` has a newer hash in the summarize
output; the measurement values and support fields still matched, and the
summarize HDF5 records the current exporter hash.

Both families retained 121/121 centroid values and 116/121 envelope-CV values.
The five unsupported windows per family were the declared segment-edge windows
(epochs 0–2 and 119–120). The MEG export states raw unscaled units and no anchor
applied. The pilot and summary smoke both used `NUMBA_DISABLE_JIT=1`; parity
with JIT enabled was not claimed. See
`project/mnps2_audit/outputs/20261118_meg_summarize_smoke/verification_report.json`
for the machine-readable comparison and output paths.

### Raw-file `features` → ordinary `summarize` end-to-end, 2026-11-19

The earlier smoke above reused features from the bounded pilot. This separate
run exercised the new v3.4 path from the K:-resident raw FIF through the
ordinary CLI `features` stage and the full subject-HDF5 `summarize` writer.
An earlier full 18-participant ds003645 run on K: used the legacy
sensor-topography config; it did not exercise this v3.4 surface
([run record](../../../project/diary/202_20260722_ds003645_full_eeg_meg_pipeline_K.md)).
The dedicated configuration is
`config_ingest_ds003645_meg_v34_e2e_k.yml`; it inherits the frozen pilot
estimators/variants, writes to
`K:/ExternalReceivedDatasets/openneuro/processed_ds003645_meg_v34_e2e_20261119/`,
and filters the run to `sub-002`, FacePerception run 1 MEG FIF. The ds003645
indexer found 224 source files; after subject and file filtering, feature
extraction processed exactly one FIF. It emitted 121 windows in a 397-column
feature table.

The standard summary completed successfully and wrote the ordinary MNPS HDF5
with both the legacy MEG outputs and `/measurements/v3_4_meg/`. Validation
against the fresh `features.parquet` found matching window axes and matching
all raw coordinate values in all three variants at 1e-12 tolerance. MAG and
GRAD each had 102 and 204 supported sensors, respectively. Both families had
121/121 centroid windows and 121/121 complete windows for `existing_coordinate_set`
and `removal_only`. The 9D `alpha_replacement` had 116/121 complete windows:
envelope-CV exclusions were exactly epochs 0–2 and 119–120, all with
`within_declared_segment_edge_guard`. Centroid used Welch; inherited PSD
reported multitaper. The export remained raw, unscaled, anchor-free, and
`qualified=false`.

The process environment left `NUMBA_DISABLE_JIT` unset. This differs from the
earlier pilot/smoke setting (`NUMBA_DISABLE_JIT=1`); no cross-setting parity
claim is made. Non-fatal output included the known duplicate legacy 9D
subcoordinate warning and a pandas DataFrame-fragmentation performance
warning from candidate feature assembly. The complete run directory and
manifest are under
`K:/ExternalReceivedDatasets/openneuro/processed_ds003645_meg_v34_e2e_20261119/ds003645/neuralmanifolddynamics_ds003645_20261119_125959/`.

## Candidate and repeatability status (evidence through 2026-11-18)

Keep the engineering and repeatability findings distinct:

- **Computability and replay:** the centroid was supported in every window in
  the ten-recording run-to-run set. Recomputed run-1 pilot exports matched in
  all ten MAG/GRAD family comparisons. The ordinary `summarize` → full HDF5
  smoke and disabled-export legacy control both passed; the experimental
  fields remain auxiliary, with MAG and GRAD separate. These findings support
  implementation consistency under the tested configuration.
- **Between-run behavior:** a frozen, five-participant within-session analysis
  compared each run's recording median. The reported 0.1103 Hz (MAG) and
  0.0842 Hz (GRAD) are **median absolute differences across the five paired
  participants**. The separate mean signed differences (run 2 − run 1) are
  +0.0880 Hz and +0.0121 Hz. Across-participant Spearman rank correlations
  were 0.10 for MAG and 1.00 for GRAD; ICC(A,1) was 0.733 and 0.876,
  respectively. With only five pairs and different participant rankings by
  sensor family, this is preliminary within-session evidence, not test–retest
  reliability or a promotion basis. See
  `project/mnps2_audit/outputs/20261118_meg_repeatability/analysis_plan_v1.md`,
  `paired_subject_results.csv`, and `repeatability_statistics.csv`.
- **Execution setting:** both pilot integrations and this repeatability
  analysis used `NUMBA_DISABLE_JIT=1`. JIT-enabled parity remains untested;
  retain this setting when reproducing the checked outputs.

| Component | Current status |
|---|---|
| Corrected existing-coordinate baseline | Retain; separate from exact historical replay |
| `alpha_centroid_hz` | Keep available as an experimental descriptor for further evaluation |
| `alpha_envelope_cv` | Preliminary; GRAD sensor-split stability remains a limitation, and repeatability was not evaluated here |
| Removing inherited `e_s` / `e_m` | Not supported by this pilot |
| Replacement configuration | Opt-in only; no coordinate replacement or deletion |

The bounded comparison config is
`config_ingest_ds003645_meg_v34_bounded_pilot.yml`. It fixes sub-002 through
sub-006, FacePerception run 1, before candidate feature computation. It
inherits the frozen sensor sectors from the existing ds003645 topography
pilot for balanced sensor-split summaries. Those sectors remain measurement
geometry labels and are not anatomical regions.
