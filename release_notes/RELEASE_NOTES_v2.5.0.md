# NeuralManifoldDynamics v2.5.0 — Multimodal ingest expansion, phase-aware features, and contract hardening

This release extends the versioned measurement contract introduced in v2.1 and
carried through v2.3/v2.4: the canonical `mnps_3d = [m, d, e]` chart, the
stratified `coords_9d` chart, and the additive embodied/interoceptive layer
(`anchor_state`, `anchor_quality`, optional `anchor_coupling`) are unchanged.
v2.5.0 adds new source-format adapters, optional phase-aware features, and
stronger provenance/QC guarantees on top of that stable contract.

A companion methods manuscript, *"NeuralManifoldDynamics: A Versioned
Measurement Contract for Low-Dimensional Neural-Manifold Trajectories"*, with
six supplements (S1 methods, S2 audited results, S3 QC/robustness, S4 a
clinician-facing reading, S5 glossary, S6 full changelog), documents this
release contract end to end and is included in
`project/articles/NeuralManifoldDynamics/`.

## Highlights

**BDF / Figshare infant EEG**
- New BDF adapters and privacy-safe metadata extraction for Figshare infant EEG.
- `BAD_` masking, coverage-aware cohort-anchor export, and a fix for
  connectivity features going all-NaN after masked segments.
- Internal run: 71 BDF files processed, 5,273 epochs retained, 70
  coverage-passing HDF5 summaries. Ingest/QC evidence only — infant
  behavioural condition labels were not recoverable from the available
  `Status` field.

**NWB / Neuropixels / ephys and LFP (new)**
- New `neuropixel_ingest` adapter for DANDI/NWB electrophysiology assets,
  covering two source types with two distinct construction paths:
  - **Units → rate**: spike times are histogram-binned into a population
    firing-rate matrix (default 0.05 s bins / 20 Hz grid, optional Gaussian
    smoothing), windowed into `ephys_*` features, and mapped into
    `coords_9d`/`mnps_3d` through a dedicated ephys-specific subcoordinate map
    (not MEG-style shadow mapping).
  - **LFP → EEG contract**: `ElectricalSeries` channels are routed through the
    existing EEG feature/9D contract with rodent-adapted band edges and
    optional depth-balanced contact selection.
- Probe/contact/reference QC, state/stimulation-aware epoching (e.g.
  awake/isoflurane/recovery blocks), and contact-count/reference-choice
  sensitivity checks.
- Validated as **transport-and-descriptive QC only** on rodent DANDI dandisets
  `000006`/`000009` (Units) and `000458` (LFP). No readiness score analogous
  to the MEG pilot has been computed, and no cross-species chart claim is made
  — these are rodent recordings, unlike the human EEG/MEG/fMRI reference paths.

**Phase anchor and sleep-EAP extensions**
- Optional per-epoch extractor (`phase_anchor.enabled: true`) for cardiac
  phase (linear RR interpolation), respiratory phase (Hilbert transform), RR
  intervals, HR, respiration, inhale fraction, and HEP-related amplitude.
- Sleep-EAP phase-2 contract with associated quality/provenance surfaces.
- Both are optional, release-bound v2.5.0 capabilities; NaN-filled columns on
  partial physiological coverage rather than errors. Dataset-scale
  interpretation remains cohort-dependent.

**Contract, provenance, and QC hardening**
- Explicit `/epoch_id` window join key where available.
- Stricter simultaneous-MEEG row-lineage checks and expanded provenance.
- Robust-z degenerate-scale safeguards (`degenerate_scale_policy = "nan"`).
- Configuration overlay replacement semantics, additional source adapters,
  tests, and QC artifacts.

## Validation and claim boundaries

| Dataset / stream | Evidence in v2.5.0 | Claim ceiling |
|---|---|---|
| ds003838 | 130 completed HDF5 exports, 27,670 block-native windows; corrected stage statistics `N = 62`, `vagal_index` listen–mem13 Cohen's `d = 1.995` | Internal task-contrast validation; no independent replication |
| ds006848 | ECG polarity/contamination audits; encoding `m`/`d` analyses, `n = 30` | Working-memory HRV claims withheld (87.7% of 60 s windows overlap retrieval) |
| ds003645 | Five-subject MEG pilot, readiness `0.7879` | Exploratory pilot; not full-cohort production validation |
| DANDI 000006 / 000009 / 000458 | Neuropixels Units and LFP smoke/QC paths | Transport and descriptive sensitivity only; no readiness score computed; rodent data, not human |
| Figshare infant EEG | BDF ingest, masking, connectivity repair, cohort outputs | No infant behavioural condition or clinical inference |

No claim in this release extends beyond what is stated above. Exploratory,
negative, and dataset-specific findings remain labelled as such throughout the
manuscript, supplements, and `CHANGELOG.md`.

## Upgrading

- The canonical `mnps_3d` / `coords_9d` export contract and axis semantics are
  unchanged from v2.4 — existing downstream consumers of those paths are not
  affected.
- New optional HDF5 groups/columns (`ephys_*`, `phase_anchor` columns,
  `/epoch_id`) are additive and disabled unless explicitly enabled in the
  dataset YAML overlay.
- No breaking changes to existing config keys; see `mndm/CONFIG_GUIDE.md` for
  the new `preprocess.nwb.units`, `preprocess.nwb.lfp`, and `phase_anchor`
  sections.

## Full changelog

See [`CHANGELOG.md`](CHANGELOG.md) for the complete cumulative history
(v2.1.0 → v2.5.0), and `project/articles/NeuralManifoldDynamics/S6_Changelog.typ`
for the manuscript-integrated version.
