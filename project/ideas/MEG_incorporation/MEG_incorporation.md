I’ll write this in English so it can go straight into the repo as a technical design note.

## Short answer

Yes: for **ds003645 with five subjects**, I would implement **two MEG↔EEG mapping layers**:

1. **MEG sensor-space shadow mapping** — fastest and best for validating that MEG can pass through the same NMD 3D/9D/MNJ export contract.
2. **Common source-space M/EEG mapping** — more biologically defensible, but heavier; better as v0.2 once the sensor-space ingest works.

The key research constraint is: **EEG and MEG should not be forced to match point-by-point in sensor space.** MEG and EEG observe overlapping neural generators but with different sensitivity profiles. MEG is relatively insensitive to radial sources, while EEG and MEG provide complementary information about cortical generators. ([PMC][1])

Your current NMD text already supports exactly this stance: NMD is a fixed, auditable measurement contract with 3D `mnps_3d`, 9D `coords_9d`, optional Jacobian exports, and explicit subject/cohort anchoring — not a claim of unique biological identity of the axes. 

---

# Draft: EEG–MEG Mapping Strategy for NeuralManifoldDynamics

## 1. Scientific aim

For ds003645, the aim is not to prove that EEG and MEG produce identical MNPS or MNJ trajectories. The aim is to test whether **the same event-locked neural-state geometry can be recovered through two complementary observation channels**.

The formal observation model should be:

[
Y^{EEG}*t = G*{EEG}X_t + \epsilon^{EEG}_t
]

[
Y^{MEG}*t = G*{MEG}X_t + \epsilon^{MEG}_t
]

where (X_t) is the latent neural state, while (G_{EEG}) and (G_{MEG}) are modality-specific observation operators. The MNPS mapping then becomes:

[
z^{q}*t = \phi_q(Y^q*{t-w:t})
]

[
x^{9,q}*t = W^{q}*{9D} z^q_t
]

[
x^{3,q}_t = P x^{9,q}_t
]

where (q \in {EEG, MEG}), (z^q_t) is the modality-specific feature vector, (W^q_{9D}) maps features into stratified 9D MNPS, and (P) is the fixed 9D-to-3D projection.

The internal NMD rule should be:

> EEG and MEG may use modality-specific feature extraction, but they must share axis names, window IDs, event labels, projection semantics, anchoring metadata, coverage metadata, and downstream Jacobian code.

This fits your current NMD contract, where 9D coordinates are fixed, 3D MNPS is derived from a fixed projection, and weights/anchors/serialization paths are versioned and auditable. 

---

## 2. Recommended first implementation: sensor-space MEG shadow mapping

For the first MEG ingest, I would create a **MEG electrophysiology-family path**, not a fully new modality ontology.

That means the MEG features mirror the EEG feature names:

| Current EEG feature        | MEG shadow feature         | Notes                                                                        |
| -------------------------- | -------------------------- | ---------------------------------------------------------------------------- |
| `eeg_delta`                | `meg_delta`                | computed separately for magnetometers and gradiometers, then robust-combined |
| `eeg_theta`                | `meg_theta`                | same frequency band definition                                               |
| `eeg_alpha`                | `meg_alpha`                | same band definition                                                         |
| `eeg_beta_alpha`           | `meg_beta_alpha`           | log-ratio preferred                                                          |
| `eeg_gamma`                | `meg_gamma`                | QC-sensitive; muscle/environment risk                                        |
| `eeg_hjorth_mobility`      | `meg_hjorth_mobility`      | computed on MEG sensor time series                                           |
| `eeg_alpha_theta`          | `meg_alpha_theta`          | log-ratio preferred                                                          |
| `eeg_permutation_entropy`  | `meg_permutation_entropy`  | same estimator                                                               |
| `eeg_hjorth_complexity`    | `meg_hjorth_complexity`    | same estimator                                                               |
| `eeg_highfreq_power_30_45` | `meg_highfreq_power_30_45` | fallback only                                                                |

The current EEG 9D mapping in your file is:

[
m_a = -0.5 \cdot eeg_{\delta} - 0.5 \cdot eeg_{\theta}
]

[
m_e = -1.0 \cdot eeg_{\alpha}
]

[
m_o = eeg_{\beta/\alpha}
]

[
d_n = eeg_{\gamma}
]

[
d_l = eeg_{HjorthMobility}
]

[
d_s = eeg_{\alpha/\theta}
]

[
e_e = eeg_{PermutationEntropy}
]

[
e_s = eeg_{HjorthComplexity}
]

[
e_m = ecg_{RMSSD} \rightarrow eog_{blink} \rightarrow eeg_{30-45Hz}
]

and then the current 3D projection is:

[
m = 0.62m_a + 0.55m_e + 0.45m_o
]

[
d = 0.50d_n + 0.82d_l + 0.28d_s
]

[
e = 0.85e_e + 0.62e_s + 0.03e_m
]

with runtime normalization and coverage-aware renormalization. 

For MEG v0.1, I would define the direct shadow mapping:

[
m_a^{MEG} = -0.5 \cdot meg_{\delta} - 0.5 \cdot meg_{\theta}
]

[
m_e^{MEG} = -1.0 \cdot meg_{\alpha}
]

[
m_o^{MEG} = meg_{\beta/\alpha}
]

[
d_n^{MEG} = meg_{\gamma}
]

[
d_l^{MEG} = meg_{HjorthMobility}
]

[
d_s^{MEG} = meg_{\alpha/\theta}
]

[
e_e^{MEG} = meg_{PermutationEntropy}
]

[
e_s^{MEG} = meg_{HjorthComplexity}
]

[
e_m^{MEG} = ecg_{RMSSD} \rightarrow eog_{blink} \rightarrow meg_{30-45Hz}
]

Then apply the **same 9D-to-3D projection**.

This gives you a clean scientific statement:

> The first MEG implementation is a homologous electrophysiology mapping, not a claim that MEG and EEG sensors are physiologically identical.

---

## 3. Magnetometers and gradiometers: do not mix raw units

For Elekta/Neuromag-style data, magnetometers and gradiometers have different units and noise properties. I would compute features separately:

[
z^{mag}*t = \phi(Y^{mag}*{t-w:t})
]

[
z^{grad}*t = \phi(Y^{grad}*{t-w:t})
]

Then robust-standardize each feature family before combining:

[
\tilde z^{mag}*{k,t} = \frac{z^{mag}*{k,t} - \mathrm{median}(z^{mag}*{k})}{\mathrm{MAD}(z^{mag}*{k}) + \epsilon}
]

[
\tilde z^{grad}*{k,t} = \frac{z^{grad}*{k,t} - \mathrm{median}(z^{grad}*{k})}{\mathrm{MAD}(z^{grad}*{k}) + \epsilon}
]

Then combine:

[
z^{MEG}*{k,t} =
\mathrm{robustmean}
\left(
\tilde z^{mag}*{k,t},
\tilde z^{grad}_{k,t}
\right)
]

In practice, I would store all three:

* `meg_mag_alpha`
* `meg_grad_alpha`
* `meg_alpha`

where `meg_alpha` is the standardized combined feature used in MNPS.

---

## 4. Anchoring policy

Use three coordinate surfaces:

### A. Subject-anchored EEG and MEG

Best for within-subject event geometry.

[
z^{q,subj}_{k,t}
================

\frac{z^q_{k,t} - \mathrm{median}*{subj,run}(z^q_k)}
{\mathrm{MAD}*{subj,run}(z^q_k) + \epsilon}
]

Use this for:

* face vs scrambled within subject
* early vs late event windows
* EEG–MEG shape similarity within the same subject

### B. Modality-specific cohort anchors

Best for comparing subjects within one modality.

[
z^{q,cohort}_{k,t}
==================

\frac{z^q_{k,t} - a^q_k}
{s^q_k + \epsilon}
]

where (a^q_k, s^q_k) are frozen EEG or MEG feature anchors.

Use this for:

* cross-subject MEG summaries
* cross-subject EEG summaries
* later larger sample analyses

### C. Paired MEEG bridge anchor

Optional diagnostic layer for ds003645 only:

[
z^{MEG \rightarrow EEG}_{t}
===========================

D z^{MEG}_{t} + b
]

where (D) and (b) are learned unsupervised from paired EEG–MEG windows in training subjects only, never from condition labels.

This should **not** be the primary scientific surface. It is a diagnostic alignment surface:

> “How close can MEG be made to EEG under an unsupervised paired-window affine bridge?”

---

## 5. Validation criteria

For five ds003645 subjects, I would not aim for inferential statistics. I would aim for **technical and geometric convergence**.

### Pass level 1: export validity

Both EEG and MEG should produce:

* `/mnps_3d`
* `/coords_9d/values`
* `/mnps_3d_dot`
* `/jacobian/J_hat`
* `/jacobian_9D/J_hat`
* `/window_start`
* `/window_end`
* `/labels/stage` or event labels
* `/features_raw/*`
* `/features_robust_z/*`
* modality-specific provenance

The current NMD manuscript already treats Jacobian export as conditional on support, finite coordinates, conditioning, and coverage; MEG should inherit the same validity rules rather than bypass them. 

### Pass level 2: event signature agreement

For each subject and modality, compute:

[
\Delta x^{q}_{event}
====================

## \mathrm{median}(x^{q}_{face})

\mathrm{median}(x^{q}_{scrambled})
]

Then compare EEG and MEG signs:

[
\mathrm{sign}(\Delta x^{EEG}_{j})
=================================

\mathrm{sign}(\Delta x^{MEG}_{j})
]

for (j \in {m,d,e,m_a,\ldots,e_m}).

The target is not perfect equality. The target is **above-null sign agreement and correlated event geometry**.

### Pass level 3: trajectory similarity

Use shape similarity rather than raw coordinate equality:

[
\rho_j
======

\mathrm{corr}
\left(
x^{EEG}*{j,t},
x^{MEG}*{j,t}
\right)
]

or after Procrustes alignment:

[
R^*
===

\arg\min_R
\lVert X^{EEG} - X^{MEG}R\rVert_F
]

[
\mathrm{error}
==============

\frac{
\lVert X^{EEG} - X^{MEG}R^*\rVert_F
}{
\lVert X^{EEG}\rVert_F
}
]

### Pass level 4: temporal null

Time-shuffle MEG windows within subject:

[
X^{MEG}_{null} = \pi(X^{MEG})
]

and verify:

[
\mathrm{sim}(X^{EEG}, X^{MEG})

>

\mathrm{sim}(X^{EEG}, X^{MEG}_{null})
]

This is crucial. Otherwise EEG–MEG “similarity” may just reflect shared marginal distributions.

---

## 6. Python stubs

### 6.1 Read one ds003645 file and split EEG/MEG

```python
from pathlib import Path
import mne

def load_meeg_raw(fif_path: str | Path):
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    ch_types = raw.get_channel_types(unique=True)

    eeg = raw.copy().pick("eeg") if "eeg" in ch_types else None
    mag = raw.copy().pick("mag") if "mag" in ch_types else None
    grad = raw.copy().pick("grad") if "grad" in ch_types else None

    return {
        "raw": raw,
        "eeg": eeg,
        "meg_mag": mag,
        "meg_grad": grad,
        "ch_types": ch_types,
        "sfreq": raw.info["sfreq"],
    }
```

MNE-Python is a reasonable basis here because it supports complete M/EEG pipelines, including preprocessing, epoching, forward modeling, inverse methods, connectivity, and source estimation. ([Frontiers][2])

---

### 6.2 Robust standardization

```python
import numpy as np
import pandas as pd

def robust_z(x, eps=1e-9, clip=8.0):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < eps:
        mad = np.nanstd(x)
    z = (x - med) / (mad + eps)
    return np.clip(z, -clip, clip)

def logistic_squash(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

def standardize_feature_table(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        out[c + "_rz"] = robust_z(out[c].values)
        out[c + "_mapped"] = logistic_squash(out[c + "_rz"].values)
    return out
```

---

### 6.3 MEG shadow feature mapping into 9D

```python
import numpy as np
import pandas as pd

NINE_NAMES = [
    "m_a", "m_e", "m_o",
    "d_n", "d_l", "d_s",
    "e_e", "e_s", "e_m",
]

P_9D_TO_3D = {
    "m": {"m_a": 0.62, "m_e": 0.55, "m_o": 0.45},
    "d": {"d_n": 0.50, "d_l": 0.82, "d_s": 0.28},
    "e": {"e_e": 0.85, "e_s": 0.62, "e_m": 0.03},
}

def choose_existing(row, candidates, default=np.nan):
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return row[c]
    return default

def build_meg_coords_9d(df: pd.DataFrame, suffix="_mapped") -> pd.DataFrame:
    """
    Expects mapped MEG features, e.g.
    meg_delta_mapped, meg_theta_mapped, ...
    """
    out = pd.DataFrame(index=df.index)

    out["m_a"] = (
        -0.5 * df[f"meg_delta{suffix}"]
        -0.5 * df[f"meg_theta{suffix}"]
    )
    out["m_e"] = -1.0 * df[f"meg_alpha{suffix}"]
    out["m_o"] = df[f"meg_beta_alpha{suffix}"]

    out["d_n"] = df[f"meg_gamma{suffix}"]
    out["d_l"] = df[f"meg_hjorth_mobility{suffix}"]
    out["d_s"] = df[f"meg_alpha_theta{suffix}"]

    out["e_e"] = df[f"meg_permutation_entropy{suffix}"]
    out["e_s"] = df[f"meg_hjorth_complexity{suffix}"]

    # e_m fallback: ECG -> EOG -> MEG high-frequency proxy.
    e_m = []
    source = []
    for _, row in df.iterrows():
        candidates = [
            f"ecg_rmssd{suffix}",
            f"eog_blink_rate{suffix}",
            f"meg_highfreq_power_30_45{suffix}",
        ]
        val = choose_existing(row, candidates)
        e_m.append(val)

        chosen = next((c for c in candidates if c in row and pd.notna(row[c])), None)
        source.append(chosen)

    out["e_m"] = e_m
    out["e_m_source"] = source

    return out

def project_9d_to_3d(coords9: pd.DataFrame) -> pd.DataFrame:
    coords3 = pd.DataFrame(index=coords9.index)

    for axis, weights in P_9D_TO_3D.items():
        num = np.zeros(len(coords9), dtype=float)
        den = np.zeros(len(coords9), dtype=float)

        for subcoord, w in weights.items():
            values = coords9[subcoord].to_numpy(dtype=float)
            valid = np.isfinite(values)
            num[valid] += w * values[valid]
            den[valid] += abs(w)

        coords3[axis] = np.where(den > 0, num / den, np.nan)

    return coords3[["m", "d", "e"]]
```

---

### 6.4 Combine magnetometer and gradiometer feature tables

```python
def combine_meg_sensor_types(
    mag_df: pd.DataFrame,
    grad_df: pd.DataFrame,
    feature_names: list[str],
    key_cols=("subject", "run", "window_id", "window_start", "window_end"),
) -> pd.DataFrame:
    """
    Combines magnetometer and gradiometer features only after robust-z scaling.
    """
    merged = mag_df.merge(
        grad_df,
        on=list(key_cols),
        suffixes=("_mag", "_grad"),
        how="inner",
    )

    out = merged[list(key_cols)].copy()

    for f in feature_names:
        mag_col = f"{f}_mag"
        grad_col = f"{f}_grad"

        mag_z = robust_z(merged[mag_col])
        grad_z = robust_z(merged[grad_col])

        out[f"meg_{f}"] = np.nanmedian(
            np.vstack([mag_z, grad_z]),
            axis=0,
        )

    return out
```

---

### 6.5 EEG–MEG coordinate comparison

```python
from scipy.stats import spearmanr
from scipy.linalg import orthogonal_procrustes
import numpy as np

def coordinate_correlations(eeg3: pd.DataFrame, meg3: pd.DataFrame):
    rows = []
    for axis in ["m", "d", "e"]:
        valid = np.isfinite(eeg3[axis]) & np.isfinite(meg3[axis])
        if valid.sum() < 5:
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(eeg3.loc[valid, axis], meg3.loc[valid, axis])
        rows.append({"axis": axis, "rho": rho, "p": p, "n": int(valid.sum())})
    return pd.DataFrame(rows)

def procrustes_error(eeg_coords, meg_coords):
    """
    Shape similarity, not raw equality.
    """
    X = np.asarray(eeg_coords, dtype=float)
    Y = np.asarray(meg_coords, dtype=float)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[valid]
    Y = Y[valid]

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    R, _ = orthogonal_procrustes(Y, X)
    Y_aligned = Y @ R

    err = np.linalg.norm(X - Y_aligned, ord="fro") / (
        np.linalg.norm(X, ord="fro") + 1e-9
    )
    return {"procrustes_error": float(err), "n": int(valid.sum())}
```

---

## 7. Source-space v0.2: more defensible but heavier

Once sensor-space MEG works, the stronger biological bridge is source-space mapping:

[
\hat S^{EEG}*t = L*{EEG}Y^{EEG}_t
]

[
\hat S^{MEG}*t = L*{MEG}Y^{MEG}_t
]

[
z^{q}_t = \phi(\mathrm{labels}(\hat S^q_t))
]

Then compute MNPS from source-label features rather than raw sensors.

This is attractive because combined EEG+MEG source localization can outperform either modality alone, and MNE provides forward/inverse tooling for M/EEG source estimation. ([PMC][3])

Minimal source-space stub:

```python
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

def compute_inverse_epochs(raw, epochs, trans, src, bem, noise_cov, modality):
    """
    modality: 'eeg', 'meg', or 'meeg'
    """
    if modality == "eeg":
        meg = False
        eeg = True
    elif modality == "meg":
        meg = True
        eeg = False
    elif modality == "meeg":
        meg = True
        eeg = True
    else:
        raise ValueError("modality must be 'eeg', 'meg', or 'meeg'")

    fwd = mne.make_forward_solution(
        info=epochs.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=meg,
        eeg=eeg,
        mindist=5.0,
        n_jobs=1,
    )

    inv = make_inverse_operator(
        info=epochs.info,
        forward=fwd,
        noise_cov=noise_cov,
        loose=0.2,
        depth=0.8,
    )

    stcs = apply_inverse_epochs(
        epochs,
        inverse_operator=inv,
        lambda2=1.0 / 9.0,
        method="dSPM",
        pick_ori=None,
    )

    return stcs
```

For ds003645 five subjects, I would **not start here** unless FreeSurfer/BEM/source files are already easy to use. Use source-space as the second validation layer.

---

## 8. Recommended pilot for your five subjects

For the first run:

1. Load each file.
2. Confirm actual channel types: EEG, MAG, GRAD.
3. Extract continuous sliding-window features separately for EEG, MAG, GRAD.
4. Build:

   * `coords_9d_eeg`
   * `coords_9d_meg_sensor_shadow`
   * `mnps_3d_eeg`
   * `mnps_3d_meg_sensor_shadow`
5. Use the same `window_id`, `window_start`, `window_end`.
6. Join HED/event labels to windows.
7. Compare face/scrambled or event-family contrasts.
8. Run temporal nulls.
9. Export both modality surfaces into the same HDF5 or sibling HDF5s with shared provenance.

The first claim should be:

> In five ds003645 subjects, a MEG sensor-shadow ingest can reproduce the NeuralManifoldDynamics export contract and recover event-locked geometry that is partially homologous to EEG under the same windowing and annotation structure.

Not:

> MEG replicates EEG.

That stronger statement should wait until you have source-space validation or more subjects.

## My concrete recommendation

Implement **`modality: meg_shadow_eeg_contract`** as the first MEG configuration.

Use:

[
W^{MEG}*{9D} = W^{EEG}*{9D}
]

with feature-name substitution:

[
eeg_* \rightarrow meg_*
]

and record in the manifest:

```yaml
modality: meg
mapping_family: electrophysiology_shadow
mapping_reference: eeg_ds004511_v2_1
sensor_types:
  - mag
  - grad
feature_combination: robust_z_then_median
primary_claim: measurement_contract_validation
not_claimed:
  - sensor-level physiological identity with EEG
  - identical MNPS values across EEG and MEG
  - clinical or diagnostic inference
```

That gives you the cleanest path from the current EEG contract to a working MEG ingest without category drift.

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2914866/?utm_source=chatgpt.com "Sensitivity of MEG and EEG to Source Orientation - PMC - NIH"
[2]: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2013.00267/full?utm_source=chatgpt.com "MEG and EEG data analysis with MNE-Python"
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2706118/?utm_source=chatgpt.com "The advantage of combining MEG and EEG - PMC - NIH"
