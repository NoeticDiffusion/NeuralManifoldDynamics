# MNPS 3.4 preliminary implementation freeze

**Status:** preliminary EEG implementation complete; frozen experimental
measurement specification. This is a consolidation checkpoint, not a new
package release or scientific qualification.

## Frozen identity

- Measurement schema: `mndm.mnps_v3_4.experimental_flat_9d.v1`, definition
  `3.4`, in package metadata `3.1.0`.
- Fixed order: `delta_theta_level`, `alpha_level`, `beta_alpha_ratio`,
  `gamma_level`, `hjorth_mobility`, `alpha_theta_ratio`,
  `permutation_entropy`, `alpha_centroid_hz`, `alpha_envelope_cv`.
- Frozen EC reference: `ds005385_ses1_pretask_EyesClosed_20261111_v1`, file
  SHA-256
  `79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab`.
- Frozen EO reference: `ds005385_ses1_pretask_EyesOpen_20261111_v1`, file
  SHA-256
  `f3bb94310778e91b046fbf77f0540ce860af6a65d4607e1e8593c01f800705b1`.
- The two anchors are separate, transductive reference frames: 608 participants
  per condition both define and receive the corresponding reference. They are
  not population norms, and EC and EO coordinates must not be pooled as one
  common frame without a separately justified procedure.

The effective source-code state for the cohort analyses is based on repository
commit `5253a60c6cb562db36ca4c49bf1b61acb3626944`; the checkout contained local
changes, so that commit alone is not a complete code identity. The full-cohort
analysis manifest records SHA-256 values for its code, configuration, inputs,
and all 360 analyzed H5 files. The legacy-Jacobian suppression verification
manifest additionally records hashes for the updated HDF5 writer and summary
pipeline. No new tag, package version, Zenodo record, or immutable software
archive has been created.

## Evidence frozen for this checkpoint

- The ds005385 cohort run extracted 55,183 feature windows and applied both
  registered anchors. Its 88.1% EC and 88.0% EO complete-9D support rates
  describe those exports only. Artifact QC was not established for that
  reference cohort.
- A bounded I-CARE run computed unanchored EEG features, but its native powers
  could not be converted defensibly to the EC anchor's µV² units. The transfer
  correctly remained unsupported; no relabelling or local normalization was
  used to bypass that measurement contract.
- The ds004148 full-cohort repeatability analysis reused session-1 H5/QC and
  re-extracted sessions 2–3 for 60 participants, EC and EO: 360 recordings and
  26,640 QC-matched windows. Every recording used the same frozen EC anchor.
  The locked QC rule and minimum support were set before coordinate
  inspection.
- 2,858/3,240 coordinate-by-participant/session/condition summaries passed the
  support floor; complete 9D support passed for 306/360 participant-session-
  condition rows. Coordinate-level session comparisons retained 43–52
  participants per row; EO−EC contrast comparisons retained 36–42.
- Descriptive ICC(A,1) estimates ranged from 0.041 to 0.954 for coordinate
  levels and 0.249 to 0.864 for EO−EC contrasts. Alpha-centroid level estimates
  ranged 0.762–0.954; alpha-envelope-CV EO−EC estimates ranged 0.793–0.864.
  Repeatability is uneven across coordinates and sessions. These are
  dataset-specific point estimates, not qualification thresholds or evidence
  that one measurement is superior.
- A one-participant summarize smoke verified that all four sub-01 EC/EO H5
  files retain the fixed v3.4 anchor, `(74, 9)` coordinate arrays and support
  masks (660/666 coordinate entries valid per recording), exactly matching the
  unsuppressed full-cohort outputs. It also verified absence of all six legacy
  Jacobian paths and the v3.4 candidate when their switches are disabled. The
  legacy identity and suppression behavior are documented in
  [`JACOBIAN_EXPORT_CONFIGURATION.md`](../config/v.3.4/JACOBIAN_EXPORT_CONFIGURATION.md).

## Known limitations and deferred questions

- QC eligibility is an operational screen, not evidence that a window is
  artifact-free. Session-1/session-2/session-3 comparisons have support-related
  attrition; per-coordinate and participant-level reasons are retained in the
  repeatability artifacts.
- Cross-dataset measurement equivalence, raw-file byte identity, and biological
  or clinical qualification remain unestablished. I-CARE anchor transfer is
  explicitly unsupported for the current amplitude-unit contract.
- Predictive increment, outcome associations, and clinical utility were not
  assessed. No condition-difference tuning, p-value screening, or Jacobian
  interpretation was performed for this repeatability analysis.
- The standard summarize pipeline can still emit historical `/jacobian`,
  `/jacobian_9D`, and anchored-layer fields. They belong to the legacy MNPS
  coordinate contracts, not to the flat v3.4 schema. The frozen repeatability
  output keeps these legacy fields for provenance but does not analyze them.
  Use the documented v3.4-only overlay to suppress them in future output.
- The software article can describe this evidence as a preliminary
  implementation result. A package-version change, archival DOI, or public
  release claim must wait for a clean tagged source state and durable archive.

## Reproducibility records

- Full-cohort results, selection/QC/run manifests, source scripts, exact YAML,
  and anchor file:
  [`ds004148_session_repeatability_fullcohort`](ds004148_session_repeatability_fullcohort/README.md).
- The analysis manifest includes 360 H5 SHA-256 values and the relevant source
  and configuration hashes. The suppression smoke manifest captures the
  post-fix code hashes and output files.
- Raw EEG binaries and full generated H5 tables remain in the local output
  tree; they are not copied into the documentation package. Reproduction uses
  the source dataset and recorded configuration, QC policy, anchor, and
  implementation identities.
