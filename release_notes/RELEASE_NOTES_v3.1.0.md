# NeuralManifoldDynamics v3.1.0 — Initial alternative MNPS implementation tranche and audited EEG pilot

Package version: `mndm.__version__ = "3.1.0"`.

This release begins implementing the v3.1 alternative-chart measurement work
inside MNDM. It adds `mndm.mnps.v3_1`, records all nine proposed families as
candidate registry entries, and provides experimental EEG sensor-space
estimators for M1, M3, D1–D3, E1, and E3. E2 has an unadjusted LZ78 core probe;
M2 has no selected estimator. These implementations do not ratify a
nine-coordinate chart or promote a biological claim.

## Implemented candidate

E1 is the existing exploratory ANPHY candidate: compute normalized ordinal
permutation entropy separately for each channel (order 5, delay 1, stable
mergesort tie handling), then take the median over the declared fixed 83-label
set. This initial implementation accepts only audited windows with 5,000
samples at 250 Hz (20 seconds). Input rows are aligned by channel label to the
declared fixed order. Missing/duplicate channels, non-finite or constant
channels, gaps, BAD overlap, irregular sample times, recording-boundary
overrun, inadequate filter-edge support, or incomplete support audit produce
an unsupported result with a reason. No interpolation, imputation, or zero
substitution is performed.

## EEG sensor-space prototypes

- M1 computes the population SD across samples of the phase order parameter
  from caller-supplied wrapped phase series. Band and phase estimator IDs are
  required from the caller. It is explicitly sensor-space, not anatomical.
- M3 returns per-pair `1 − Pearson r` for adjacent, equal-length,
  nonoverlapping activity vectors or connectivity vectors. The two methods
  have separate estimator IDs and are not combined.
- D1 computes weighted global efficiency with length `1 / weight`, requiring
  symmetric weights in `[0,1]`; disconnected pairs contribute zero.
- D2 returns class-A minus class-B mean coupling under a supplied symmetric
  pair mask. Near/far and within/between variants remain separate.
- D3 computes mean off-diagonal Euclidean distance for sensor feature
  patterns under caller-declared construction and scaling IDs.
- E3 computes participation ratio from centered sample covariance without
  shrinkage. Its effective-sample-size, split-spectrum, and `N_eff/R` gates
  are not implemented.
- E2 computes a single pre-symbolized channel's LZ78 phrase count per symbol.
  It is not the proposed matched-reference-normalized E2 metric; symbolization
  and channel aggregation remain open.

All metric prototypes produce separate `computed`, `valid`, and `qualified`
fields. Every result is `qualified=false` with
`claim_status=no_biological_claim`. The opt-in
`load_audited_edf_eeg_window` path creates a source-backed support record for
one EDF window: it records all MNE annotations, sample-clock and segment
checks, exact finite FIR contributor bounds and coefficient hashes, channel
order, stable source/positions/overlay hashes, code hashes, and runtime
versions. It remains an isolated research sidecar and is not part of the
default feature pipeline.

The bounded real-data pilot used ANPHY EPCTL01, one window at 120–140 s, 83
labels, and 5,000 output samples at 250 Hz. The candidate computed
`0.6198971779299133`; the source dependency interval was 116.624–139.997 s, the
source and overlay hashes were stable, and no BAD or boundary annotation fell
inside the dependency interval. It is a **PARTIAL PASS for one-window technical
feasibility only**. The causal-FIR overlay differs from the earlier ANPHY
default-pipeline smoke. No condition contrast, reliability, cohort behavior,
nonredundancy, biological meaning, or qualification was tested.

The same contract was then run once on the frozen nominal grid
`[0,20), …, [280,300)`. Fourteen windows were computed and valid. `[0,20)`
was unsupported because its causal dependency starts at −3.376 s, and that
window was not replaced. The 120–140 s value and dependency bounds reproduced
the one-window pilot. Every G2 record remains `qualified=false`. Details are
in diary 494.

## Compatibility and claim boundary

- `mndm.mnps.v3_1` is additive and is not wired into default EEG, MEG, fMRI,
  fNIRS, or ephys profiles.
- M2 is unimplemented. E2 has only the unadjusted core probe described above.
  All other implemented estimators are technical sensor-space prototypes, not
  qualified coordinates. No M/D/E coordinate assignment is made.
- No alternative HDF5 schema, writer, reader, chart Jacobian, or 9D-to-3D
  projection is introduced.
- Existing `/mnps_3d`, `/coords_9d`, anchored coordinates, `/jacobian`, and
  `/jacobian_9D` definitions and export behavior are unchanged.

## Validation

The focused suite passed 95 tests, including synthetic support rejection and
legacy export/projection controls. Exact commands, the bounded real-data
provenance, independent review, limitations, and next gate are recorded in
[`project/diary/493_20260929_mnps_v310_eeg_audited_pilot.md`](../project/diary/493_20260929_mnps_v310_eeg_audited_pilot.md)
and [`project/diary/494_20260929_mnps_v310_e1_g2_grid.md`](../project/diary/494_20260929_mnps_v310_e1_g2_grid.md).
The initial estimator tranche is recorded in
[`project/diary/491_20260929_mnps_v310_eeg_prototypes.md`](../project/diary/491_20260929_mnps_v310_eeg_prototypes.md).
