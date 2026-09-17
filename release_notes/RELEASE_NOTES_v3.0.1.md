# NeuralManifoldDynamics v3.0.1 — Level identities, withheld locks, overlay defaults

v3.0.1 is a patch on the v3.0.0 measurement contract. Canonical `mnps_3d = [m, d, e]`,
stratified `coords_9d`, `J_hat`, and existing family schema IDs are unchanged.
Itô drift/diffusion, peak-gain L4, reactivity-gap L3, anisotropy-001 names,
FAR observational names, p50/p90 thresholds, empirical future spread, attractors,
and `SOURCE_CROSSFIT` as independent \(b\) remain withheld.

Package version: `mndm.__version__ = "3.0.1"`.

## Highlights

**New writable dynamical-family identities (opt-in overlay, now default-on there)**
- Chart drift: `realized_velocity_level0`, `conditional_mean_rate_level1`
  (pooled / blocked_crossfit / lag_diagnostics). Not `/mnps_3d_dot`, not Itô \(b\).
- Affine one-step map L2, direct lag-2 when `declared_lags` includes 2, generator
  proxies L3 from `logm(Phi)/dt`, Euclidean functionals, iterated horizon map L4.
  Identification remains `mndm.one_step_fit_fidelity.v1` with threshold **0.9**.
  That threshold is not retuned.
- Amplification L1: neighbor-gain q90, separation rate, gain-rate, cloud-volume rate.
- History L1 predictive gain and nested L2 history-conditioned operator.
  History L3/L4 remain withheld.
- Turning L0: successive-increment angle and rate. Undefined displacement is NaN,
  not zero.

**Destination / FAR / reachability identity locks**
- Resolved first-hit `q_A_to_B` stays `destination_first_hit_fraction_resolved_level1`.
  Unresolved-included hit probability, unresolved fraction, and L0 first-hit
  fraction are withheld.
- Existing `amplitude_curve` stays FAR L4. Spontaneous return and p50/p90
  supremum names are withheld. Discrete r50 remains a diagnostic.
- Existing `w_q` stays Lyapunov predictive spread. `conditional_future_spread_level1`
  is withheld.

**Grain and status (queue 25)**
- Every writable family v1 group has nested `grain/` even when not computed.
- `computation_status`, `measurement_validity`, `claim_status`, and
  `qualification_status` are siblings. There is no global `qualified` flag.
- YAML `translation_qualification.qualified` remains a destination/resilience
  gate only.

**Overlay defaults**
- `config_ingest_common_dynamical_families.yaml` now defaults
  `dynamical_families.enabled` and the writable families
  (diffusion, destination, resilience, drift, one_step, amplification,
  history, turning) plus local-dynamics FTR, Gate E residuals, and Gate F
  \(W_Q\) to **true**.
- Common EEG/fMRI/ephys profiles still do **not** import this overlay.
- Nested C1 `diffusion.drift.enabled` stays **false** (no independent \(b\);
  `A_bD` remains `not_testable`).
- Destination and resilience still fail closed without protocol + TQ.

## Validation and claim boundaries

| Surface | Evidence in v3.0.1 | Claim ceiling |
|---|---|---|
| MNPS / 9D / `J_hat` | Unchanged from v3.0.0 | Chart coordinates and chart Jacobian |
| New family identities | Synthetic tests + named I-CARE smokes | Schema objects, not biology |
| Withheld 001/002 names | YAML/H5/register refuse | Not landed |
| Overlay default-on | Common dynamical-families YAML | Only datasets that import that overlay |
| `A_bD` | Still `not_testable` without independent \(b\) | Not empirical alignment |

**Safe claims**

- The new family identities serialize under `/dynamical_families/.../v1` with
  grain and status siblings.
- Closed names cannot be aliased onto nearby objects.
- Common EEG/fMRI/ephys ingest paths are unchanged unless they import the
  dynamical-families overlay.

**Not established in v3.0.1**

- A latent SDE, licensed NDT \(\alpha/\omega/G_{\mathrm{peak}}\), Itô
  qualification, observational FAR, empirical future covariance, attractors,
  independent chart \(b\) for `A_bD`, or CPC-tuned one-step thresholds.

## Upgrading

- Existing MNPS, 9D, Jacobian, jacobian_metrics, and FTR paths remain compatible.
- Datasets that already imported `config_ingest_common_dynamical_families.yaml`
  now compute the additional families when support holds. Destination/FAR still
  require protocol inputs. Gate F \(W_Q\) still requires Gate E residuals.
- Common EEG/fMRI/ephys profiles are unchanged.
- Do not treat YAML `qualified: true` as a global measurement license.

See [`mndm/CONFIG_GUIDE.md`](../mndm/CONFIG_GUIDE.md) and
[`RELEASE_NOTES_v3.0.0.md`](RELEASE_NOTES_v3.0.0.md).
