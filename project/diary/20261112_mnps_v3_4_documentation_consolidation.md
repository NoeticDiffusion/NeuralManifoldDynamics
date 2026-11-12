# MNPS 3.4 documentation consolidation - 2026-11-12

## Scope and checkout

Completed the SL-006 documentation round in `M:\repos\NeuralManifoldDynamics`.
`Get-Location` and `git rev-parse --show-toplevel` both resolved to that M:
checkout; the SL-006 task file and all v3.4 implementation/configuration
artifacts reviewed were also in that checkout. No documentation or code was
edited on H:. The current M: `git status` shows the pre-existing v3.4 code and
configuration work plus this documentation round.

The checkout does not contain a dedicated
`project/mnps2_audit/sl/sl_007.md` or preserved SL-007 run log. The 29-test
result, static checks, and reviewer outcome below are therefore reported from
the previous task session, not independently reconstructed from a local log.

## Version and measurement clarification

- Package metadata and latest formal release notes are 3.1.0; the root
  README's earlier 3.0.2 statement was stale.
- Canonical/default MNPS v2 retains `[m,d,e]`, optional `coords_9d`, and the
  legacy Jacobian surface.
- v3.1 and v3.2 remain distinct alternative/experimental code paths. This
  checkout has no `src/mndm/mnps/v3_3` module.
- MNPS 3.4 is an opt-in experimental flat 9D EEG measurement definition, not
  a package-version bump. It emits no M/D/E aggregation, 3D projection, or
  v3.4 Jacobian.
- The guide records actual coordinate assembly, units, legacy label mapping,
  feature extraction, anchor order, masks, sidecar behavior, condition-specific
  ds005385 references, and the limits of cross-surface transfer.
- The v3.4 config declares neutral geometry defaults as pending validation;
  the v3.4 exporter does not calculate geometry outputs from them.
- `J_hat` is the legacy local vector-field Jacobian estimate; `J_dot` is its
  temporal derivative. Neither is a Hessian or an MNPS 3.4 object.

## Manuscript and release metadata

The new main article, S1 methods, and S6 changelog are local revision drafts
for the experimental MNPS 3.4 measurement definition. Each imports
`template_eLife.typ`. The main article defines the current flat nine-coordinate
contract; historical `[m,d,e]` semantics, old feature mapping, and version
history are isolated in S6. The pre-existing Typst sources, PDFs, and public
Zenodo record were not replaced. No package release or DOI was invented.

## Evidence recorded

- Previous SL-007 task report: 29 focused tests passed; compilation, YAML
  loading, CLI help, diff checks, and an independent reviewer were reported
  successful. No dataset extraction, summarize run, or anchor refit was
  reported for SL-007. Full external-anchor orchestrator HDF5 writing remains
  unverified; the broader orchestrator suite did not finish.
- Earlier native ds004148 pilot: 3 participants, 18 recordings, 1,332 windows,
  18 HDF5 outputs; 1,224 windows had complete 9D support. This predates the
  SL-007 external-anchor integration and remains unqualified.
- ds005385 reference cohort: 608 participants/recordings per EO and EC; all
  1,216 source-to-H5 matches used unique manifest basenames
  (`unique_manifest_basename_match`); full relative path identity was not
  established. Anchors computed for all 608 per condition, no anchor identity
  failures, and no unclassified
  numeric HDF5 dataset path. Selection was metadata-based. Raw EDF hashes and
  artifact QC were not established.
- HDF5 statistics include means, population SD, median, MAD, scaled MAD, Q1,
  Q3, IQR, min/max, skewness, excess kurtosis, and support counts, separated by
  full path and realization. JSON finite values are inventoried separately
  from their validity and qualification flags.
- No dataset was run, no anchor was fit, and nothing was published during this
  documentation round.

The active checkout was confirmed as `M:\repos\NeuralManifoldDynamics` by both the working directory and Git root. The cited SL-006 file exists here; SL-007 markdown/run artifacts do not. No H: checkout was read or modified. The corrected HDF5 paths and basename-only cohort identity were independently rechecked. Main, S1, and S6 revision PDFs compiled; final article pages were visually checked.

## Article rewrite and final review

The article was rewritten around the implemented v3.4 measurement contract,
with a narrower Methods-paper claim set and explicit reference-cohort and
qualification limits. The prior three-axis model, its `[m,d,e]` proxy
interpretation, old nine-to-three projection, and version history now live in
Supplementary Changelog S6. The user confirmed that the previous version is
already on Zenodo and that there are no current users; this was recorded as
owner-provided context, without guessing a Zenodo DOI.

An independent GPT-6.1 Terra Low review checked the manuscript claims against
the implementation and configs. It caught two method descriptions that were
corrected: the first seven common EEG features use a per-sample median across
available channels, and the common feature PSD is configured multitaper with a
Welch fallback; Welch is used directly for the alpha centroid candidate. All
three revised Typst sources import the requested eLife template and compiled
successfully. The latest main-article PDF pages 1, 2, and 6 were visually
checked after the final source edit. No dataset was rerun and nothing was
published during this manuscript revision.

## Remaining check

The highest-value pre-release check is one bounded fresh-feature pilot through
the explicit external-anchor route, followed by inspection of the resulting
HDF5/sidecar masks and anchor provenance. Do this only after the documented
integration path is ready for runtime verification; it should resolve the
current uncertainty without a cohort-scale rerun.
