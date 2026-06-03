Yes — I think this is a very good next step. I would add these as a **conventional comparator layer** beside MNPS/MNJ/reachability, not inside the MNPS axes themselves. That keeps NeuralManifoldDynamics as a fixed auditable measurement contract, while making every article easier to compare with ordinary EEG literature. Your current NMD framing already supports this: it emphasizes fixed exports, deterministic preprocessing, coverage handling, optional Jacobians, provenance, and comparison-ready outputs rather than claiming superiority over HMMs, microstates, CEBRA, dynamic connectivity, or other state-space methods. 

## Recommended conventional EEG module structure

I would create something like:

```text
/conventional_eeg/
  spectral/
  ratios/
  peak_frequency/
  complexity/
  connectivity/
  graph/
  microstates/
  erp/
  sleep_events/
  coma_qeeg/
  seizure_features/
  summaries/
```

Each method should export both **window-level features** and **subject/session-level summaries**:

```text
features_conventional/<family>/<feature_name>
summaries_conventional/<family>/<feature_name>_{mean,median,std,iqr,mad}
qc_conventional/<family>/*
```

The most important general families are:

| Family               | Core methods                                                                                          | Why useful                                                                 |
| -------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Spectral qEEG**    | absolute/relative delta, theta, alpha, beta, gamma power                                              | Almost every clinical EEG literature uses these                            |
| **Ratios / slowing** | theta/alpha, delta/alpha, (delta+theta)/(alpha+beta), alpha/theta, beta/alpha                         | Directly comparable for dementia, Parkinson, coma, encephalopathy          |
| **Peak frequency**   | individual alpha peak frequency, median frequency, spectral edge frequency                            | Strong for neurodegeneration, slowing, coma/anesthesia                     |
| **Spectral shape**   | 1/f slope, aperiodic exponent, offset                                                                 | Good modern comparator, separates broadband slowing from oscillatory peaks |
| **Complexity**       | spectral entropy, permutation entropy, sample entropy, Lempel-Ziv, Higuchi/Katz FD, DFA/Hurst         | Useful across dementia, anesthesia, sleep, coma, depression                |
| **Connectivity**     | coherence, imaginary coherence, PLV, PLI/wPLI, amplitude envelope correlation                         | Needed for AD/FTD, psychosis, Parkinson, coma, epilepsy                    |
| **Graph metrics**    | strength, clustering, path length, modularity, efficiency, small-worldness                            | Good bridge to network neuroscience                                        |
| **Microstates**      | duration, occurrence, coverage, transition matrix, GEV                                                | Strong conventional dynamic-state comparator                               |
| **Event/ERP**        | P300, MMN, auditory steady-state response, ERD/ERS                                                    | Essential for oddball, Parkinson cognition, psychosis, dementia            |
| **ICU/qEEG**         | suppression ratio, burst-suppression, continuity, reactivity proxy, seizure burden, alpha-delta ratio | Needed for coma/cardiac arrest                                             |
| **Sleep events**     | spindles, slow waves, K-complexes, SO-spindle coupling, hypnogram transitions                         | Needed for sleep-stage and spindle work                                    |

MNE-Features already supports a broad feature-extraction model for epoched EEG, while Pycrostates is directly aimed at EEG microstate analysis, and YASA supports sleep staging, spindles, slow waves, REMs, bandpower, PAC, and hypnogram statistics. ([mne.tools][1])

## Domain-specific priorities

### Alzheimer’s / dementia / FTD

For AD and broader dementia work, I would prioritize:

1. **Spectral slowing**

   * delta ↑
   * theta ↑
   * alpha ↓
   * beta ↓
   * theta/alpha ↑
   * delta+theta / alpha+beta ↑
   * individual alpha peak frequency ↓

2. **Connectivity**

   * coherence / imaginary coherence / wPLI by band
   * alpha-band and beta-band network weakening
   * graph efficiency, clustering, modularity

3. **Complexity**

   * spectral entropy
   * sample entropy
   * permutation entropy
   * Lempel-Ziv complexity
   * fractal dimension

4. **Microstates**

   * class duration
   * occurrence
   * coverage
   * transition entropy
   * global explained variance

5. **ERP if task data exists**

   * P300 latency/amplitude
   * mismatch negativity

This fits the conventional AD qEEG literature: AD is commonly associated with reduced higher-frequency alpha/beta activity and increased slowing, while qEEG slowing is also studied as a marker of dementia risk and progression. ([PMC][2])

For your NMD comparison, the clean paper-facing structure would be:

```text
AD/FTD table:
- Conventional slowing index
- Alpha peak frequency
- Relative theta
- Relative alpha
- Spectral entropy
- wPLI alpha
- Microstate transition entropy
- MNPS / 9D / MNJ / reachability
```

### Parkinson’s disease

For Parkinson, especially if you have ON/OFF medication or cognitive status, I would prioritize:

1. **Resting qEEG slowing**

   * theta ↑
   * alpha peak frequency ↓
   * beta/gamma changes
   * theta/alpha ratio
   * median frequency

2. **Motor-system beta**

   * beta power over sensorimotor electrodes
   * beta coherence
   * beta burst metrics if high-quality data
   * corticomuscular coherence if EMG exists

3. **Task EEG**

   * movement-related beta ERD/ERS
   * mu rhythm suppression
   * motor imagery features

4. **Cognition-related features**

   * P300 amplitude/latency
   * alpha peak frequency
   * theta/alpha slowing

5. **Medication comparison**

   * OFF vs ON change in beta, theta, alpha peak, entropy, connectivity

PD qEEG literature often emphasizes generalized slowing and links qEEG features to cognitive decline; several studies discuss increased slow activity and altered alpha/beta dynamics in PD. ([PMC][3])

For NMD, Parkinson is especially interesting because conventional qEEG may show **slowing**, while MNPS/MNJ may show whether the slowing is also a loss of trajectory flexibility, altered local rotation, or reduced reachability.

### Sleep

For sleep, conventional methods should be quite rich because sleep has mature event-based EEG analysis.

Prioritize:

1. **Sleep staging / hypnogram**

   * Wake, N1, N2, N3, REM
   * stage proportions
   * stage latency
   * transition matrix
   * sleep fragmentation
   * WASO, sleep efficiency

2. **Spectral by stage**

   * delta/SWA in N3
   * sigma in N2
   * theta/alpha in REM and wake
   * beta/gamma arousal markers

3. **Spindles**

   * density
   * amplitude
   * duration
   * frequency
   * topography
   * fast vs slow spindles
   * spindle refractory / clustering

4. **Slow waves**

   * density
   * amplitude
   * slope
   * duration
   * up/down-state proxies

5. **Coupling**

   * slow oscillation–spindle phase coupling
   * spindle–ripple if invasive or hippocampal data exists
   * PAC: SO phase / sigma amplitude

6. **Complexity and microstates by stage**

   * entropy reduction in N3
   * REM complexity/repertoire
   * stage-wise microstate syntax

This aligns well with your existing spindle article: you already note that spindles are usually treated as spectral events, while NMD asks whether they also produce local state-space reconfiguration.  YASA is a natural tool here because it supports automatic sleep staging, spindle/slow-wave/REM detection, bandpower, PAC, and hypnogram analysis. ([GitHub][4])

For the repo, I would make sleep the first full conventional module because it has clear events and strong validation hooks:

```text
sleep_conventional:
- hypnogram_stats
- bandpower_by_stage
- spindle_features
- slow_wave_features
- so_spindle_coupling
- transition_matrix
- matched_non_spindle_controls
```

### Coma / cardiac arrest / ICU EEG

For coma and post-cardiac-arrest EEG, conventional comparators are essential, because reviewers will expect them.

Prioritize:

1. **ACNS-style background features**

   * continuity / discontinuity
   * suppression
   * burst suppression
   * voltage category
   * symmetry/asymmetry
   * variability
   * reactivity if stimulation annotations exist
   * periodic/rhythmic patterns
   * epileptiform burden

2. **qEEG severity features**

   * suppression ratio
   * burst-suppression ratio
   * alpha-delta ratio
   * delta/alpha ratio
   * total power
   * spectral entropy
   * amplitude-integrated EEG
   * regularity / discontinuity index

3. **Seizure-related features**

   * seizure burden
   * periodic discharge burden
   * rhythmic/periodic pattern count
   * ictal-interictal continuum labels if available

4. **Recovery/prognosis comparators**

   * early EEG background recovery
   * reactivity
   * continuity
   * highly malignant EEG pattern proxy
   * qEEG trend over first 12/24/48h

ACNS terminology is the obvious conventional reference point for ICU EEG, and modern neuroprognostication guidelines discuss suppressed/burst-suppression background and EEG background recovery as prognostic features after cardiac arrest. ([acns.org][5]) ERC/ESICM guidance also treats suppressed background and burst suppression as “highly malignant” EEG patterns for poor-outcome prediction, with timing and multimodal interpretation caveats. ([PMC][6])

For your coma article line, the important design is not “NMD vs EEG” but:

```text
clinical + conventional qEEG
clinical + conventional qEEG + NMD geometry
```

That is already consistent with your recovery-linked coma framing, where the geometry layer is presented as complementary to clinical + qEEG rather than as a bedside-ready replacement. 

### Epilepsy / seizure datasets

For epilepsy, add:

1. **Interictal**

   * spike/sharp-wave rate if annotations/detector available
   * focal slowing
   * alpha slowing
   * bandpower asymmetry
   * functional connectivity asymmetry

2. **Ictal**

   * seizure onset spectral change
   * line length
   * Teager energy
   * rhythmicity index
   * high-frequency oscillations if iEEG/high sampling rate
   * seizure propagation speed
   * postictal suppression/slowing

3. **Network**

   * phase synchrony
   * wPLI/coherence
   * directed connectivity if careful: Granger/transfer entropy/phase slope index

For NMD comparison, epilepsy is useful because conventional features often detect “event intensity,” while MNJ/reachability may describe **transport capture, deformation, or reduced traversal**.

### Depression / psychiatric resting EEG

For depression and psychosis, I would include these conventional comparators:

1. **Depression**

   * frontal alpha asymmetry
   * alpha power
   * theta/beta ratio
   * vigilance/arousal proxies
   * alpha peak frequency
   * entropy/complexity
   * microstate duration/transition entropy

2. **Psychosis**

   * low-frequency power
   * alpha reduction
   * gamma/ASSR if task
   * MMN/P300 if task
   * microstates
   * connectivity/dysconnectivity metrics
   * entropy/complexity

This is especially useful because your depression result suggests the strongest signal was not coarse 3D MNPS but reachability, 9D MNJ, and distribution summaries. A conventional comparator layer would help show whether those effects reduce to ordinary bandpower/entropy, or whether geometry adds something extra. 

## Minimal first implementation set

For the first version, I would not implement everything. I would implement a **Tier 1 conventional EEG pack** that works across almost all EEG datasets:

```text
Tier 1: universal conventional EEG
1. bandpower_abs: delta, theta, alpha, beta, gamma
2. bandpower_rel: delta, theta, alpha, beta, gamma
3. ratios:
   - theta_alpha
   - delta_alpha
   - alpha_theta
   - beta_alpha
   - slowing_index = (delta + theta) / (alpha + beta)
4. peak_frequency:
   - alpha_peak_frequency
   - median_frequency
   - spectral_edge_95
5. complexity:
   - spectral_entropy
   - sample_entropy
   - permutation_entropy
   - higuchi_fd
6. connectivity:
   - coherence
   - wpli
   - plv
7. microstates:
   - duration
   - occurrence
   - coverage
   - transition_matrix
   - transition_entropy
```

## Implementation status in this repo

As of the current comparator-layer rollout in this repository:

- implemented:
  - Tier 1 spectral/ratio/peak-frequency comparator surface
  - conventional `complexity` pack:
    - spectral entropy
    - permutation entropy
    - Hjorth complexity
    - optional Hjorth mobility
  - conventional `connectivity` pack built from the existing synchrony engine:
    - coherence
    - PLV
    - PLI
    - wPLI
    - optional dPLI / PPC depending on config
- not yet promoted into the conventional comparator surface:
  - microstates
  - aperiodic / 1-f slope or exponent
  - graph metrics as a separate conventional pack
  - sleep/coma/ERP domain packs
  - `antropy`-based windowed complexity outputs such as multiscale entropy, Lempel-Ziv, and Higuchi FD

So the current repo has already moved beyond the original narrow Tier 1 baseline
and now exposes three generic EEG comparator families in the main pipeline:

```text
conventional_eeg:
  - tier1
  - complexity
  - connectivity
```

Then add **Tier 2 domain packs**:

```text
sleep_pack:
  sleep_stage_stats
  spindle_features
  slow_wave_features
  so_spindle_coupling

coma_pack:
  suppression_ratio
  burst_suppression_proxy
  continuity_proxy
  alpha_delta_ratio
  seizure_burden_proxy
  acns_background_proxy

erp_pack:
  p300
  mmn
  assr_40hz
  erd_ers

epilepsy_pack:
  line_length
  teager_energy
  spike_rate_proxy
  hfo_features
  seizure_burden
```

## Best scientific positioning

I would describe this addition as:

> NeuralManifoldDynamics now includes a conventional EEG comparator layer, exporting standard qEEG, complexity, connectivity, microstate, sleep-event, and ICU-qEEG features alongside MNPS, Stratified MNPS, MNJ, and reachability. These conventional features are not treated as inferior baselines, but as necessary reference surfaces for evaluating whether NMD geometry adds information beyond established EEG summaries.

That is the reviewer-safe framing.

My strongest recommendation: start with **qEEG spectral/ratio + complexity + microstates**, then add **sleep_pack** and **coma_pack** because those directly support your strongest current article lines.

[1]: https://mne.tools/mne-features/api.html?utm_source=chatgpt.com "API Documentation — mne_features 0.2 documentation"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10642578/?utm_source=chatgpt.com "qEEG as Biomarker for Alzheimer's Disease - PMC"
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4842380/?utm_source=chatgpt.com "Quantitative EEG and Cognitive Decline in Parkinson's Disease"
[4]: https://github.com/raphaelvallat/yasa?utm_source=chatgpt.com "YASA (Yet Another Spindle Algorithm): a Python package ..."
[5]: https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf?utm_source=chatgpt.com "ACNS GUIDELINE"
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7993077/?utm_source=chatgpt.com "European Resuscitation Council and European Society of ..."
