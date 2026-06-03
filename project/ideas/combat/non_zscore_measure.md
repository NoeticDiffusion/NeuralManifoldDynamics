You are acting as a critical research-methods advisor for a neuroscience / dynamical-systems project called NeuralManifoldDynamics.

Context:
We currently map EEG/fMRI/iEEG-derived windowed features into low-dimensional neural manifold coordinates, including a 3D chart [m, d, e] and a stratified 9D chart. The pipeline has used z-score-like normalization in some places, often within subject/session, before projecting features into these coordinates.

Concern:
Subject-wise or session-wise z-scoring may remove biologically meaningful absolute or group-level differences. For example, if one group has genuinely lower variance, reduced entropy, stronger rigidity, larger dispersion, or altered amplitude structure, per-subject normalization may make that difference invisible. This is especially important for clinical or state comparisons such as psychosis vs control, coma recovery vs non-recovery, propofol sedation, sleep stages, epilepsy, dementia, or task conditions.

Task:
Openly reason about alternative normalization and anchoring strategies that could replace or complement per-subject z-scoring.

Please explore several options, including but not limited to:

1. Raw-feature preservation
   - Keeping raw or minimally transformed feature values alongside normalized coordinates.
   - When raw features are essential for interpretation.

2. Batch-level anchoring
   - Computing median/IQR or mean/std over the whole dataset or analysis batch.
   - Transforming all subjects and conditions relative to this shared batch anchor.
   - Strengths and weaknesses for within-dataset group comparisons.

3. Group-aware but non-leaky anchoring
   - How to avoid using test labels or outcome labels in a way that creates leakage.
   - Whether anchors should be computed across all subjects, only controls, only training folds, or external reference cohorts.

4. Atlas/reference anchoring
   - Building frozen modality-specific reference anchors, e.g. adult-rest EEG, sleep EEG, fMRI rest.
   - Using these to create externally anchored coordinates that allow cross-dataset comparisons.
   - Risks of site, montage, device, age, medication, or preprocessing confounds.

5. Subject-relative plus absolute dual export
   - Keeping both within-subject normalized coordinates and externally/batch-anchored coordinates.
   - How each coordinate layer should be interpreted differently.

6. Robust scaling
   - Median/MAD, median/IQR, quantile transforms, winsorization, rank transforms, or Huber-style scaling.
   - Which methods preserve group differences and which destroy them.

7. Covariate-adjusted anchoring
   - Residualizing for age, sex, site, device, montage, medication, sampling rate, or recording quality.
   - When residualization is useful versus when it may remove the signal of interest.

8. Dynamical consequences
   - How different normalization choices affect trajectory volume, speed, Jacobian estimation, reachability cones, anisotropy, and capacity metrics.
   - Which metrics are especially sensitive to scaling.

9. Reviewer-facing method design
   - How to make the choice auditable and reproducible.
   - What should be stored in metadata/manifests.
   - What sensitivity analyses should be required.

Please structure your answer as:

- A short diagnosis of why per-subject z-scoring may be problematic.
- A comparison table of candidate normalization strategies.
- A recommended default strategy for NeuralManifoldDynamics.
- A set of sensitivity analyses that should be run in every serious paper.
- A proposed HDF5/export schema for storing multiple coordinate layers.
- A claim-discipline section separating:
  - established statistical facts,
  - plausible methodological choices,
  - speculative extensions,
  - and failure modes.
- A concise recommendation for what should become the primary coordinate layer and what should remain secondary or diagnostic.

Important constraints:
Do not assume that there is one universally correct normalization. Treat normalization as part of the measurement contract. Prioritize methods that preserve both within-dataset comparability and cross-dataset interpretability. Be explicit about leakage risks, scale artifacts, and how normalization affects downstream dynamical quantities.