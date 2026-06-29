# Noetic Anchors article handover

## Purpose

This handover is for the downstream analysis repo at `J:\repos\NoeticDiffusion\`.

The goal is to turn the now-implemented MNDM 2.3 embodied-anchor surface into a
clean analysis/article workflow that can show how **Noetic Anchors** interact
with:

- canonical `mnps_3d`
- stratified `coords_9d`
- Jacobian/MNJ surfaces

without collapsing these into the same concept.

The core framing should be:

> the embodied anchor surface is an **additive, time-aligned modulatory layer**
> beside MNPS, not a replacement for the canonical manifold chart and not a
> renaming of `feature_anchors`.

---

## Current status in MNDM

The MNDM side is now far enough along that the analysis repo should treat this
as a real downstream surface rather than a speculative stub.

### Implemented and validated

- `anchor_state`, `anchor_state_dot`, `anchor_quality` exported to H5
- optional `anchor_coupling` exported when enough valid windows exist
- task-state labels for `ds003838`:
  - `task_state_label`
  - `task_load_label`
  - `task_load_n`
- derived `event_locked` sidecars from task-state segments
- derived `block_native` sidecars from task-state segments
- HRV v0.1 raw surface:
  - `ecg_hrv_hr_mean_bpm`
  - `ecg_hrv_ibi_mean_ms`
  - `ecg_hrv_sdnn_ms`
  - `ecg_hrv_rmssd_ms`
  - `ecg_hrv_pnn50`
  - `ecg_hrv_nn_count`
  - `ecg_hrv_artifact_fraction`
  - `ecg_hrv_coverage_fraction`
  - `ecg_hrv_quality_score`
  - `qc_ok_ecg_hrv`
- `anchor_state` now prefers HRV v0.1 features when available
- full `ds003838` run completed successfully at dataset scale

### Full-run validation path

Validated output run:

`H:/SourceRepo2/NeuralManifoldDynamics/.full_ds003838_hrv_blocknative/ds003838/neuralmanifolddynamics_ds003838_20260607_111755`

Important validated properties from that run:

- `130` subject/task H5 outputs
- `130` `summary.json`
- `130` H5 files with `/block_windows`
- aggregate `block_native`:
  - `6543` blocks
  - `27670` windows
  - `source_window_match_fraction = 1.0`
- raw `ecg_hrv_*` columns confirmed directly in exported `block_native_windows.csv`

This means the analysis repo can start from the emitted artifacts instead of
having to add new ingest logic first.

---

## What the analysis repo should consume

The analysis repo should consume four surfaces, not one.

### 1. Canonical manifold surface

Primary H5 paths:

- `/mnps_3d`
- `/mnps_3d_dot`
- `/coords_9d/values`
- `/coords_3d_subject_anchored/values`
- `/coords_9d_subject_anchored/values`

Use this for:

- 3D trajectory analyses
- 9D subcoordinate analyses
- chart geometry and trajectory shape

### 2. Embodied anchor surface

Primary H5 paths:

- `/anchor_state/values`
- `/anchor_state_dot/values`
- `/anchor_quality/values`
- optional `/anchor_coupling/*`

Use this for:

- embodied state trajectories
- quality-aware filtering of anchor analyses
- testing whether body-state variation tracks manifold position or deformation

### 3. Derived block-native surface

Primary sidecars:

- `sub-*/block_native_windows.csv`
- `sub-*/block_native_windows.parquet`
- run-level `block_native_qc.json`

Use this for:

- sustained task-phase analysis
- HRV analysis
- within-block timing analyses
- anchor-vs-MNPS interaction at behaviorally meaningful scales

This should be the **main analysis surface** for the first noetic-anchor paper
on `ds003838`.

### 4. Derived event-locked surface

Primary sidecars:

- `sub-*/event_locked.csv`
- `sub-*/event_locked.parquet`

Use this for:

- onset/transition analyses
- pre/post comparisons around segment boundaries
- checks that complement, but should not replace, the block-native story

---

## Non-negotiable analysis rules

### Rule 1: honor `geometry_contract`

Before using any MNPS/Jacobian surface, read:

- `summary.json.geometry_contract`
- or H5 `/provenance/geometry_contract/*`

Do not analyze Jacobians or 9D surfaces naively.

At minimum, downstream loaders should gate on:

- `geometry_contract.status`
- `geometry_contract.coords_9d.degenerate_axes`
- `geometry_contract.jacobian.windows_retained`
- `geometry_contract.jacobian_9d.windows_retained`

### Rule 2: do not conflate two kinds of anchors

Keep these distinct in all analysis code and paper text:

- `feature_anchors`: frozen cohort/external scaling artifacts for coordinate contracts
- `anchor_state`: embodied/interoceptive time-aligned surface

### Rule 3: treat `anchor_coupling` as optional

`anchor_coupling` is valuable when present, but it is not guaranteed.

For the first paper, the main positive results should not depend exclusively on
this surface. Use it as:

- a confirmatory/additive result when available
- or a later section / supplement

### Rule 4: the first claim should be interaction, not replacement

Do **not** write the paper as:

> “MNPS is incomplete until a fourth bodily axis is added”

Write it as:

> “Embodied anchors provide a parallel state surface that modulates and helps
> interpret manifold position and deformation.”

That is both truer to the implementation and much more defensible.

---

## Recommended analysis-repo structure

Recommended working root:

`J:\repos\NoeticDiffusion\`

Suggested modules/notebooks:

### `analysis/noetic_anchors/loaders.py`

Should implement:

- `load_run_manifest(run_dir)`
- `load_subject_summary(run_dir, subject_task_dir)`
- `load_h5_core(h5_path, honor_geometry_contract=True)`
- `load_block_native_table(path)`
- `load_event_locked_table(path)`

### `analysis/noetic_anchors/qc.py`

Should implement:

- geometry gating
- anchor-quality filtering
- subject/task inclusion summaries
- Jacobian availability summaries

### `analysis/noetic_anchors/block_native.py`

Should implement:

- stage/block summaries
- subject-centered normalization
- anchor-MNPS association models
- within-block time covariates

### `analysis/noetic_anchors/nulls.py`

Should implement or wrap:

- within-subject time-shift nulls
- subject-shuffle nulls
- stage-label permutation checks where appropriate

### `notebooks/`

Suggested notebook order:

1. `01_ds003838_qc_and_surface_inventory.ipynb`
2. `02_block_native_anchor_vs_mnps.ipynb`
3. `03_anchor_vs_coords9d.ipynb`
4. `04_anchor_vs_mnj.ipynb`
5. `05_event_locked_transition_checks.ipynb`
6. `06_null_controls_and_robustness.ipynb`

---

## What the first paper should actually try to show

The first paper should be an **interaction paper**, not a new grand theory
paper.

### Best central question

Do embodied anchors explain meaningful variation in manifold position and local
dynamical structure during sustained task states in `ds003838`?

### Best central claim

Across sustained task-state blocks in a multimodal dataset, embodied anchors
(especially HRV-linked anchor dimensions) vary systematically with task phase
and are associated with structured changes in:

- 3D MNPS position
- 9D subcoordinate composition
- and selected Jacobian/MNJ summaries

while remaining an additive layer distinct from the canonical MNPS contract.

### Stronger than this

You can likely support:

- embodied anchors differ by task state/load
- anchor state covaries with manifold position
- 9D exposes interactions masked in 3D composites
- some dynamical summaries vary with anchor state

### Weaker than this

Avoid claiming:

- direct causal bodily control of manifold flow
- a universal fourth canonical MNPS axis
- general proof of Noetic Diffusion Theory from one dataset

---

## Recommended article architecture

## 1. Introduction

Keep the intro short and applied.

The article should start from a real problem:

- MNPS captures neural manifold position and deformation
- but some state variance may be better interpreted when an embodied surface is
  available in parallel
- current multimodal datasets make it possible to test this directly

Position noetic anchors as:

- additive
- measurable
- quality-aware
- testable through interactions with MNPS/MNJ

not as a metaphysical overlay.

## 2. Methods

The methods should be organized in this order.

### 2.1 Dataset and task structure

Focus on `ds003838`:

- rest
- digit span
- derived task-state segments:
  - `listen`
  - `mem5`
  - `mem9`
  - `mem13`

### 2.2 Neural manifold surface

Describe:

- `mnps_3d`
- `coords_9d`
- Jacobian/MNJ export
- `geometry_contract` gating

### 2.3 Embodied anchor surface

Describe:

- `anchor_state`
- `anchor_state_dot`
- `anchor_quality`
- HRV v0.1 preference logic
- optional `anchor_coupling`

### 2.4 Analysis surfaces

Explicitly separate:

- canonical H5 surfaces
- `block_native` sidecars for sustained-state analysis
- `event_locked` sidecars for transition/onset analysis

### 2.5 Statistical design

Pre-specify a small number of analyses:

- within-subject block summaries by task state
- mixed-effects or repeated-measures models
- anchor-vs-MNPS regressions controlling for task state
- null controls

## 3. Results

Suggested section structure:

### 3.1 The embodied surface is stable and measurable

Show:

- QC availability
- anchor-quality distributions
- HRV availability by task
- number of retained subjects/blocks/windows

This section establishes that the surface exists and is not noise-only.

### 3.2 Anchors vary with task state and load

This should probably be the first real result.

Use `block_native` tables to show:

- stage-wise means or distributions for:
  - `sympathetic_index`
  - `vagal_index`
  - `anchor_index`
  - possibly raw `ecg_hrv_*` columns

This is the easiest reviewer-friendly entry point.

### 3.3 Anchors interact with canonical 3D MNPS position

Show:

- block-wise or window-wise associations between anchor measures and `m`, `d`,
  `e`
- subject-centered fits
- effects controlling for task state and relative block position

This should answer:

> when embodied state shifts, where does the system move in MNPS 3D?

### 3.4 9D analysis reveals masked structure

This is where the paper becomes more interesting.

Instead of only reporting `m`, `d`, `e`, test anchor association with:

- `m_a`, `m_e`, `m_o`
- `d_n`, `d_l`, `d_s`
- `e_e`, `e_s`, `e_m`

This section should show that:

- some effects visible in 9D are weak or cancel in 3D
- embodied anchors may interact with subcoordinate redistribution rather than
  only global axis magnitude

This is probably the strongest justification for including 9D at all.

### 3.5 Anchors and MNJ / local deformation

This section should be careful.

Recommended order:

1. report Jacobian availability after `geometry_contract`
2. focus on the surface with the best retention
3. use anchor-conditioned summaries rather than overly ambitious causal
   interpretation

What to test:

- does high vs low anchor state correspond to different Jacobian summaries?
- do block-level Jacobian family summaries differ by task state and anchor
  quantile?
- when `anchor_coupling` exists, does it align with the simpler association
  analyses?

Write this as:

> embodied state is associated not only with manifold position but with local
> deformation structure

not:

> embodied anchors determine the vector field

### 3.6 Null controls and robustness

This section is essential.

Include:

- within-subject time-shift nulls
- subject-shuffle nulls
- robustness to QC thresholds
- robustness to using raw HRV columns vs composite anchor indices

## 4. Discussion

The discussion should separate four claim levels:

### Established by implementation

- MNDM can export embodied anchors beside MNPS
- these can be aligned to task-state blocks and event-centered windows
- they can be analyzed without changing the canonical MNPS contract

### Internally validated empirical findings

This is where the actual `ds003838` results go.

### Plausible interpretation

- embodied state may help explain otherwise ambiguous manifold changes
- some 9D redistributions may correspond better to embodied modulation than 3D
  alone

### Still speculative

- broad theoretical claims about noetic anchoring as a general principle across
  modalities or disorders

---

## Recommended figures

Aim for 5 main figures plus supplement.

### Figure 1. Contract and analysis surface diagram

Show:

- raw multimodal data
- feature extraction
- MNPS 3D / 9D
- anchor_state
- derived block-native and event-locked layers
- MNJ / Jacobian layer

This should visually separate:

- canonical manifold surface
- embodied anchor surface
- downstream derived analysis surfaces

### Figure 2. QC and data coverage

Show:

- subject/task counts
- retained windows
- anchor quality
- Jacobian availability after geometry gating

### Figure 3. Task-state dependence of anchors

Use block-native summaries:

- `listen`
- `mem5`
- `mem9`
- `mem13`
- optionally `rest`

Likely panels:

- `sympathetic_index`
- `vagal_index`
- one raw HRV metric such as `ecg_hrv_rmssd_ms`

### Figure 4. Anchor interactions with MNPS 3D and 9D

Possible layout:

- top row: anchor vs `m`, `d`, `e`
- bottom row: heatmap of anchor-vs-9D subcoordinate associations

This is likely the conceptual centerpiece.

### Figure 5. Anchor interactions with MNJ / deformation

Possible panels:

- Jacobian family summary by anchor quantile
- block-Jacobian summaries by task state
- optional `anchor_coupling` example where valid

### Supplement

Put here:

- event_locked boundary analyses
- extra HRV metrics
- subject-level spaghetti plots
- null-distribution panels
- sensitivity to QC thresholds

---

## Concrete analysis plan for the repo

If I were implementing this next in `J:\repos\NoeticDiffusion\`, I would do it in
this order.

### Step 1. Build a surface inventory table

For each `summary.json` / H5:

- subject
- task
- anchor-state available?
- anchor-coupling available?
- `geometry_contract` status
- `jacobian.windows_retained`
- `jacobian_9d.windows_retained`
- block-native windows count

This determines what the paper can safely claim.

### Step 2. Make the primary block-native table

Concatenate all `block_native_windows.csv` files and keep:

- identifiers
- block timing fields
- `m`, `d`, `e`
- raw `ecg_hrv_*`
- anchor indices
- anchor-quality fields
- `task_state_label`
- `task_load_n`

Add derived columns:

- z-scored within subject
- anchor tertiles / quartiles
- centered time-in-block

### Step 3. Run descriptive stage models

First models should be easy to interpret:

- anchor metric ~ task state + (1 | subject)
- MNPS axis ~ task state + (1 | subject)

Then:

- MNPS axis ~ anchor metric + task state + relative_pos_0_1 + (1 | subject)

### Step 4. Run 9D models

For each 9D coordinate:

- subcoord ~ anchor metric + task state + relative_pos_0_1 + (1 | subject)

Then use:

- FDR correction
- family grouping by `m`, `d`, `e`

### Step 5. Run MNJ/Jacobian summary analyses

Only on runs/windows that pass geometry gating.

Potential outcomes:

- trace-like summaries
- Frobenius norm
- anisotropy
- block family summaries

### Step 6. Run nulls

At minimum:

- within-subject time-shift null
- subject-shuffle null

The paper becomes much stronger if the observed anchor-vs-manifold interaction
structure exceeds these nulls.

---

## What to expect scientifically

What I expect to be strongest:

1. **task-state dependence in anchor metrics**
2. **association between anchors and 3D MNPS position**
3. **stronger/more specific associations in 9D than 3D**
4. **some dynamical modulation, but likely more fragile than the positional story**

What I expect to be weakest:

1. broad causal claims
2. dependence on `anchor_coupling` alone
3. 3D Jacobian-only story if many runs have poor retention

So the article should probably be written as:

- first: block-native anchor states
- second: MNPS position
- third: 9D specificity
- fourth: MNJ/deformation

not the other way around.

---

## Writing strategy

The best article is probably **not** “theory first”.

It should be written as a constrained multimodal methods/results paper:

### Recommended tone

- concrete
- additive
- contract-aware
- interaction-focused
- careful about claim boundaries

### Recommended title direction

Something like:

- **Embodied Anchors as an Additive State Surface for Neural Manifold Dynamics**
- **Noetic Anchors Interact with Neural Manifold Position and Deformation in a Multimodal Task Dataset**
- **An Embodied Anchor Layer for Interpreting Neural Manifold Dynamics**

### Recommended one-sentence paper claim

In a multimodal task dataset, an additive embodied-anchor surface derived from
ECG/PPG/pupil-linked signals covaries with both neural manifold position and
selected local dynamical summaries, with stronger specificity visible in the
stratified 9D chart than in the canonical 3D chart alone.

---

## Suggested immediate next actions in the analysis repo

1. Build the dataset inventory/QC loader around the completed full run at
   `H:/SourceRepo2/NeuralManifoldDynamics/.full_ds003838_hrv_blocknative/ds003838/neuralmanifolddynamics_ds003838_20260607_111755`.
2. Create one concatenated `block_native` analysis table for all subjects/tasks.
3. Produce one first-pass figure set:
   - anchor metrics by task state
   - anchor vs `m/d/e`
   - anchor vs 9D heatmap
4. Gate Jacobian analyses strictly with `geometry_contract`.
5. Add null-control notebooks before writing the main Results text.

---

## Bottom line

The first article should show:

1. embodied anchors are now a real exported analysis surface,
2. they vary systematically across sustained task states,
3. they interact with manifold position in 3D,
4. they reveal more specific structure in 9D,
5. and they have at least partial correspondence with local dynamical structure
   when MNJ surfaces are valid.

That is already a strong paper.

Trying to make the first paper prove a universal new axis theory would be much
weaker than showing a clean, validated, additive interaction story.
