# 033 2026-05-20 ds003944 anchor rerun comparison

## Question

Can `ds003944` be re-summarized with an explicit MNDM 2.1 cohort anchor, and how does that anchored run differ from the earlier subject-anchored run?

## Runs

- First run: `M:\datasets\processed\openneuro\ds003944\neuralmanifolddynamics_ds003944_20260520_075205`
- Anchor artifact: `M:\datasets\processed\openneuro\ds003944\ds003944_anchor_iqr_v2_1.json`
- Second run: `M:\datasets\processed\openneuro\ds003944\neuralmanifolddynamics_ds003944_20260520_081959`

## What was done

1. Ran `prerequisite-check` and a full `feature + summarize` pass for `ds003944`.
2. Fit a subject-balanced IQR anchor from the first run H5 outputs.
3. Re-ran `summarize` only with `mnps_projection.anchor.enabled=true` and the fitted anchor file injected into the config object.
4. Compared the first and second run via `run_manifest.json`, sample H5 provenance, and aggregate `summary.json` metrics.

## Established internal results

- Both runs completed successfully with 82 H5 files, 82 `summary.json`, 82 `qc_summary.json`, and 82 `qc_reliability.json`.
- The second run upgraded all 82 H5 files to the full MNDM 2.1 anchored contract:
  - `/feature_anchors` present in 82/82
  - `/coords_3d_cohort_anchored` present in 82/82
  - `/coords_9d_cohort_anchored` present in 82/82
- Sample H5 provenance changed as intended:
  - first run `primary_coordinate_contract=subject_anchored`
  - second run `primary_coordinate_contract=cohort_anchored`
  - second run root attrs include `anchor_id=ds003944_all_subjects_iqr_v2_1`
  - second run root attrs include a stable `anchor_hash`
- The second run preserved local baseline entries and added cohort-anchor baseline entries in `summary.json` (for example `eeg_alpha__cohort_anchor`).

## Aggregate summary comparison

Selected group contrasts (`FEP - Control`) changed as follows:

- `meta_indices.mean_trace`: `+0.0100 -> -0.0046`
- `meta_indices.mean_rotation_fro`: `-0.0022 -> -0.0020`
- `meta_indices_v2.mean_trace`: `-0.0057 -> -0.0025`
- `meta_indices_v2.mean_rotation_fro`: `-0.0001 -> -0.0074`

Selected paired subject-level shifts (`second run - first run`):

- `meta_indices.mean_trace`: mean `+0.0020`, median absolute change `0.0124`
- `meta_indices.mean_rotation_fro`: mean `+0.0131`, median absolute change `0.0143`
- `meta_indices_v2.mean_trace`: mean `-0.0001`, median absolute change `0.0130`
- `meta_indices_v2.mean_rotation_fro`: mean `+0.1878`, median absolute change `0.1621`

## 9D Jacobian comparison

For `jacobian_9D/J_hat`, the overall group separation changed less than for the 3D coordinates:

- 9D Jacobian group-difference Frobenius norm:
  - subject-anchored `0.1118`
  - cohort-anchored `0.1143`
- `mean_trace (FEP - Control)`:
  - subject-anchored `-0.0057`
  - cohort-anchored `-0.0025`
- `mean_rotation_fro (FEP - Control)`:
  - subject-anchored `-0.0001`
  - cohort-anchored `-0.0074`

The larger difference was not total magnitude but **which blocks carried the contrast**:

- subject-anchored strongest blocks: `M<-M`, `E<-M`, `D<-D`
- cohort-anchored strongest blocks: `E<-M`, `D<-D`, `M<-D`

This suggests that cohort anchoring changes the routing/coupling pattern visible in the stratified Jacobian more than it changes the total amount of 9D Jacobian separation.

### More biologically readable 9D interpretation

Using the active EEG 9D meanings:

- `m_e`: macrostate engagement / alpha desynchronization
- `e_e`: entropy complexity
- `e_s`: entropy spectral-complexity mix
- `e_m`: embodiment / broadband arousal proxy
- `d_n`: dynamic network binding / gamma drive
- `d_s`: dynamic spectral shift

The strongest subject-anchored 9D Jacobian shifts were mainly **reduced `m_e` outflow** in FEP:

- `m_e -> e_e` lower in FEP
- `m_e -> m_o` lower in FEP
- `m_e -> m_a` lower in FEP
- `m_e -> d_l` lower in FEP

The strongest cohort-anchored 9D Jacobian shifts instead emphasized **`m_e -> e_*` couplings**:

- `m_e -> e_m` higher in FEP
- `m_e -> e_s` higher in FEP
- `e_s -> e_m` lower in FEP
- `d_n -> d_s` higher in FEP

Plausible interpretation: cohort anchoring makes the stratified dynamics look less like a generic weakening of engagement-centered coupling and more like a redistribution from macrostate engagement into entropy/embodiment channels, with a smaller additional shift inside the dynamic spectral block.

## Analysis-repo handoff note

A dedicated markdown handoff for downstream analysis repositories was added at:

- `project/feature_summary/mndm_2_1_analysis_repo_handoff.md`

The note explains the new MNDM 2.1 H5 contract, explicit anchored coordinate layers, embedded feature-anchor provenance, and the rule that exported Jacobians follow the run's primary coordinate contract.

## One-shot anchor implementation

Implemented a one-shot cohort-anchor workflow for MNDM 2.1:

- `python -m mndm.cli summarize --fit-anchor`
- `python -m mndm.cli all --fit-anchor`

Current behavior:

1. read merged `features.parquet` / `features.csv`
2. fit a subject-balanced anchor from the feature table
3. freeze and save the anchor JSON under the run directory
4. apply that frozen anchor during the same summarize pass

This removes the need for a separate bootstrap subject-anchored summarize pass on large datasets while preserving the auditable fit/freeze/apply contract.

## Warnings / limits

- BrainVision headers with missing `MarkerFile` were repaired temporarily during the feature run.
- EEG CSD fell back to standard EEG on several files because digitization metadata was missing; config already allowed `on_error: "warn"`.
- FD censoring was skipped because `framewise_displacement` was absent.
- These comparisons are descriptive only. No inferential testing or CV-based downstream classification has been run yet.

## Claim discipline

- **Internal validated result:** the anchor-enabled rerun changes the exported contract and changes run-level summary values.
- **Plausible interpretation:** cohort anchoring is materially changing subject placement and derivative summaries rather than merely adding redundant metadata.
- **Not yet established:** whether the anchored contract improves clinically useful separation on `ds003944` in a benchmarked downstream task.

## Next useful step

Run the same anchor fit + anchored summarize + comparison loop for `ds003478` and `ds004504`, then compare whether the direction and magnitude of group-contrast shifts are consistent across the three OpenNeuro EEG smoke-test datasets.
