# Future Zenodo version changelog - draft only

Status: **not a release announcement**. Do not upload or publish this text yet.

## Record identity

- Software package version: pending release decision; current checkout metadata and latest formal release notes are 3.1.0.
- Measurement definition: experimental MNPS 3.4, opt-in EEG flat 9D payload.
- Zenodo concept DOI: not established from repository metadata.
- Zenodo version DOI: not assigned.
- Version-specific archived manuscript: not established; this repository contains a v3.0.0 article source and separate local revision drafts.
- Public release status: unchanged.

## Proposed change summary

This source-tree update documents a separate experimental MNPS 3.4 EEG measurement surface while retaining the established canonical MNPS v2/v3.0 outputs. The v3.4 coordinate payload contains nine ordered flat coordinates, frozen feature and coordinate references, per-window support/validity masks, native sensor/reference provenance, and explicit unqualified status. It does not add M/D/E family aggregation or a 3D projection. The current checkout separately provides a default-on 9D Jacobian at `/measurements/v3_4/jacobian` and opt-in drift, diffusion, and anchored lag-1 one-step measurements at `/measurements/v3_4/dynamics`; these sidecars do not change the flat coordinate definition or assert a new package release.

The v3.4 workflow includes registered, condition-specific technical references from the ds005385 session-1 pre-task EyesClosed and EyesOpen cohorts (608 participants per condition). External-anchor selection is independent of target condition. Strict compatibility is the default; exploratory transfer requires a recorded rationale and does not establish cross-dataset or scalp-to-iEEG harmonization.

The accompanying documentation inventories v2, v3.1, v3.2, and v3.4 roles; describes coexisting HDF5 schemas; and records reference-cohort statistics and evidence limits. A local article revision draft and revised figures/supplements are provided separately. Existing article source and PDFs remain unchanged pending verification of their Zenodo record mapping.

## Evidence and limits to carry into release metadata

- The ds005385 anchor fits are condition-specific and transductive: reference participants also receive their reference. Artifact QC was not established, and raw EDF content hashes were not computed.
- Earlier v3.4 native-channel pilot and cohort runs do not verify the later external-anchor summarize-to-HDF5 orchestrator route.
- The previous SL-007 task reported 29 focused tests, compilation/YAML/CLI/diff checks, and an independent review. The dedicated SL-007 markdown/run artifact was not found in this M: checkout; the full orchestrator write path remains unverified and the broader orchestrator suite did not finish.
- No evidence here establishes biological interpretation, prediction, clinical utility, normative status, cross-dataset harmonization, or superiority over MNPS v2.

## Before publication

Confirm the exact Zenodo concept and version record, the manuscript/PDF currently attached to it, package version and source commit, license and file inventory, release DOI assignment, and final external-anchor end-to-end pilot evidence. Reconcile the final change summary with those verified record fields. Do not copy this draft into a public record without that reconciliation.
