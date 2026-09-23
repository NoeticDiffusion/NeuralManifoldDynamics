# Dynamical families

`mndm.dynamical_families` is separate from MNPS coordinates and
Jacobian-derived `mndm.dynamics` measurements.

Canonical write paths:

```text
/dynamical_families/diffusion/v1
/dynamical_families/destination/v1
/dynamical_families/resilience/v1
/dynamical_families/drift/v1
/dynamical_families/one_step/v1
/dynamical_families/amplification/v1
/dynamical_families/history/v1
/dynamical_families/turning/v1
```

Legacy read-only paths (pre-v3 files):

```text
/orthogonal_dynamics/diffusion_geometry/v1
/orthogonal_dynamics/committor/v1
/orthogonal_dynamics/finite_amplitude_resilience/v1
```

Schema IDs are unchanged. Spread as a family YAML key
(`mndm.stochastic_reachability.v1`) remains Gate F `gate_closed` and is not
written under `/dynamical_families`. Opt-in discrete `W_Q` is
`/stochastic_reachability/v1` via `local_dynamics.stochastic_reachability`.

Each family v1 group writes three sibling certificate fields:
`computation_status`, `measurement_validity`, and `claim_status`, plus
`qualification_status` for the estimator identity. These are not one
field. `computation_status` is whether a value exists.
`qualification_status` is the identity token (for example
`one_step_identified` or `ito_not_qualified`). YAML
`translation_qualification.qualified` is a destination/resilience gate
only; it is not a global `qualified` flag and is not serialized on the
family payload. `provenance.validation_level` (`mndm_translation_validated` /
`simulator_validated`) is a method-validation tag, not
`measurement_validity`. New writes set `claim_status` to
`no_biological_claim` and never emit `ndt_licensed`.
`measurement_validity=translation_qualified` requires a computed **destination
or resilience** object plus an already recorded TQ id and contract hash.
Diffusion writes `not_assessed` when computed; OD-TQ1 tags live in
`provenance` and do not gate computation. Legacy readers must use
`not_recorded` for missing fields and must not infer regime validity or NDT
licensing. Missing `claim_status` may be copied from `provenance/claim_status`
when that dataset exists; `certificate_origin` records whether the three
siblings were already present.

Round 3 adds nested `grain/` on each family v1 group. Diffusion, destination,
and drift are `native=window`; amplification is `native=window`;
one_step and history are `native=recording` (one operator or one gain
per recording; window-shaped series are broadcasts, not replicates);
turning is `native=window`;
resilience is `native=event`. `biological_unit` is
`subject`; `direct_between_subject_inference` is `forbidden`. Grain is schema
metadata and is written even when `computation_status` is not `computed`.
Legacy readers must use `not_recorded` for missing grain fields and must not
infer `window` from series.

## Diffusion geometry

The diffusion tensor estimates local **centered** covariance of state
increments divided by nominal \(\Delta t\)
(`estimand=centered_increment_covariance_over_nominal_dt`).
`a_semantics=raw_increment_covariance` means the increments are not
residualized by a drift field (C1). It is not the uncentered second moment
\(E[\Delta X\Delta X^\top]/\Delta t\) and not \(D=a/2\). Existing `a_hat`
is the documented identity `conditional_covariance_rate_level1`; the series
name is not renamed. `D_total` / `d_diff` / `c_diff` remain trace and
anisotropy scalars of \(a\). They are not `diffusion_effective_dimension_level1`
or `diffusion_condition_number_level1`. `ito_diffusion_tensor_level3` is withheld.
`increment_covariance_level0` is the nested unconditional centered
\(\mathrm{Cov}(\Delta X)\) (not divided by \(\Delta t\), not local kNN,
not `a_hat`). `innovation_covariance_level2` is written by the one-step family, not here.
It is not the
transition-residual covariance proxy. Jacobian derivative residuals are not
accepted as diffusion \(a\) (`jacobian_derivative_residual_as_diffusion`).
MNPS \(\dot x\) is not an SDE drift (`mnps_xdot_as_sde_drift`). Jacobian
intercepts and same-sample increment means are also refused as \(b\).

`contract_status=standard` is the schema-contract class on provenance. It is
not an "experimental vs licensed" scientific tag and is not
`measurement_validity`.

Ingest always calls the estimator with C1 defaults (`drift=None`,
`residualize_increments=False`). A `computed` object is increment-covariance
geometry, not a testable \(A_{bD}\) / \(R_{b/a}\) object. Those series remain
NaN and are labelled `summary.A_bD_computation_status=not_testable` and
`summary.R_b_over_a_computation_status=not_testable` with
`drift_alignment_failure_reason=independent_drift_not_supplied`. Do not
treat NaN as zero alignment. Library C1 may align an externally qualified
chart-space velocity to that increment covariance without changing `a_hat`.
The opt-in `/dynamical_families/drift/v1` identities are not a
`drift_source` for these alignment scalars. Level 1 conditional mean rate is
not an independent \(b\) for \(A_{bD}\) / \(R_{b/a}\). Those stay
`independent_drift_not_supplied` until a later alignment gate. Under C1,
`ratio_semantics=chart_velocity_to_increment_spread`; this is not
an Itô drift-to-diffusion ratio. C2 residualization is not authorized on
ingest. An empirical C1 overlay is not authorized.

All diffusion estimates remain chart-dependent. When the family is enabled,
ingest computes the increment-covariance object if estimator support holds
and writes `measurement_validity=not_assessed`. Truth-known OD-TQ1 tags in
`provenance` are method-validation, not an empirical-interpretation license
and not a YAML on/off switch. Common profiles do not enable the family.

## Chart drift identities (SL-LEV-MES-003)

`/dynamical_families/drift/v1` (`mndm.chart_drift.v1`) is a Type D opt-in
family. YAML default is off. Nested names are measurement identities, not the
older parallel Level 0–3 evidence classes. 3D subject-anchored chart only.

| measurement_id | level | physical path | notes |
|---|---|---|---|
| `smoothed_velocity_savgol_level0` | 0 | `/mnps_3d_dot` | register identity of the existing Savitzky–Golay dataset; not written by this family |
| `realized_velocity_level0` | 0 | `/dynamical_families/drift/v1/realized_velocity_level0` | per-step \((x_{t+1}-x_t)/\Delta t\); not an alias of `/mnps_3d_dot` |
| `conditional_mean_rate_level1` | 1 | `/dynamical_families/drift/v1/conditional_mean_rate_level1/{pooled,blocked_crossfit,lag_diagnostics}` | mean increment over nominal \(\Delta t\). Cross-fit is a `variant_id`, not a level upgrade. Multi-lag is diagnostics, not `ito_drift_level3` |
| `conditional_covariance_rate_level1` | 1 | `/dynamical_families/diffusion/v1/series/a_hat` | documented identity of existing centered increment covariance; not a rename; convention \(a\) not \(D=a/2\) |
| `increment_covariance_level0` | 0 | `/dynamical_families/diffusion/v1/increment_covariance_level0` | nested unconditional centered \(\mathrm{Cov}(\Delta X)\), not / \(\Delta t\), not local kNN, not `a_hat` |
| `innovation_covariance_level2` | 2 | `/dynamical_families/one_step/v1/innovation_covariance_level2` | affine residual covariance of the qualified one-step map; not diffusion `a_hat`; C2 still closed |
| `ito_drift_level3` | 3 | withheld | not written |
| `ito_diffusion_tensor_level3` | 3 | withheld | not written |

`blocked_crossfit` embargo is `embargo_semantics=index_steps` (index lag, not
full raw-window/filter support). `raw_window_support_independence` is stamped
`not_established`; overlapping analysis windows, lags, history triples, and
filter support are not proven independent by disjoint row indices. Shared lag-1 **source transitions** (not kNN
neighborhoods) are identified by `series/source_idx` and
`summary/transition_support_id` on pooled drift and on diffusion.
`source_idx` has length `n_increment_pairs` (typically \(T-1\)), not \(T\);
it is not aligned to `a_hat[t]` or `b_hat[t]`. When both families compute
in one export, `lag1_support_ids_match` and `n_common_source_idx` report
whether those source indices agree. Cross-fit writes its own `support_id`
for the embargoed subset; `variant_id` stays `blocked_crossfit`. Lag 2/4
diagnostics have per-lag `transition_support_id_lag*`, not the lag-1 id.
`lag_inconsistent` does not erase `pooled`.
The string `ito_qualified` is never written. Provenance sets
`not_sde_drift=true`. None of these fields is a `drift_source` for
\(A_{bD}\) (`chart_drift_as_independent_b`; cross-fit as
`crossfit_not_authorized_before_m3`). Ingest C1 leaves `A_bD` /
`R_b_over_a` `not_testable`. 9D drift is out of scope.
Coverage is fail-closed (`insufficient_local_support`); missing windows stay
NaN. Computation, support, and qualification statuses are separate fields.

## Affine one-step operator (SL-LEV-MES-003 level 2–4)

`/dynamical_families/one_step/v1` (`mndm.affine_one_step.v1`) is a Type D
opt-in family. YAML default is off. PhysioNet I-CARE 2.1 dynamical-families
overlays may set `enabled: true` as a coverage opt-in; they do not retune
the 0.9 threshold. The v1 operator is a recording-level lag-1 affine map
\(x_{t+\Delta}\approx\Phi(x_t-\bar x)+c\) with two temporal blocks and an
index embargo (`mndm.one_step_fit_fidelity.v1`, threshold 0.9 vs a
mean-next-state baseline). It is not `expm(J_hat dt)`, not the production Jacobian, and
not I-CARE `jacobian_metrics`. Neighborhood `k` is logged and unused for
\(\Phi\). YAML `declared_lags: [1]`. Direct lag-2 identities write
under `declared_lag_2/` when `declared_lags` includes 2; they are the
same object class as lag 1, fitted on pairs \((x_t, x_{t+2\Delta})\).
Composing \(\Phi_1\) is not that identity (`iterated_one_step_horizon_map_level4`).
Lag-2 generator proxies write under `declared_lag_2/` from
\(\logm(\Phi_2)/\mathrm{nominal\_dt}\) when that map is identified
(`nominal_dt` is the median lag-2 span, \(\approx 2\Delta t\)). They use
the same `qualification_status=generator_proxy_from_qualified_one_step_not_ito`
token. Still not Itô. Euclidean functionals of a qualified \(\Phi\)
write at the requested lag with
`qualification_status=one_step_functional_of_qualified_map_not_independent_oos`.

| measurement_id | level | physical path | notes |
|---|---|---|---|
| `affine_one_step_map_level2` | 2 | `/dynamical_families/one_step/v1/affine_one_step_map_level2` | recording \(\Phi\); `qualification_status=one_step_identified` only when holdout rel-MSE beats the baseline |
| `conditional_affine_mean_rate_level2` | 2 | `/dynamical_families/one_step/v1/conditional_affine_mean_rate_level2` | one-step functional \((\Phi(x-\bar x)+c-x)/\Delta t\) |
| `innovation_covariance_level2` | 2 | `/dynamical_families/one_step/v1/innovation_covariance_level2` | affine residual covariance / \(\Delta t\); not diffusion `a_hat`; C2 remains closed |
| `operator_max_gain_rate_level2` | 2 | `/dynamical_families/one_step/v1/operator_max_gain_rate_level2` | \(\log\sigma_{\max}(\Phi)/\Delta t\); Euclidean SVD; not spectral abscissa; not peak gain |
| `operator_volume_gain_rate_level2` | 2 | `/dynamical_families/one_step/v1/operator_volume_gain_rate_level2` | \(\log\lvert\det\Phi\rvert/\Delta t\); rank-deficient \(\Phi\) is not epsilon-rescued |
| `operator_rotation_rate_level2` | 2 | `/dynamical_families/one_step/v1/operator_rotation_rate_level2` | \(\lVert\log R\rVert_F/(\sqrt{2}\,\Delta t)\) from polar \(\Phi=RP\); not generator-proxy rotation |
| `spectral_abscissa_level3` | 3 | `/dynamical_families/one_step/v1/spectral_abscissa_level3` | \(\max\mathrm{Re}\,\mathrm{eig}(\logm(\Phi)/\Delta t)\); only if the map is identified |
| `numerical_abscissa_level3` | 3 | `/dynamical_families/one_step/v1/numerical_abscissa_level3` | max eigenvalue of the symmetric part of the same generator proxy |
| `reactivity_gap_level3` | 3 | withheld | \(\omega-\alpha\) is not its own leaf; jacobian-metrics `reactivity_gap` is not this identity |
| `operator_gain_anisotropy_level2` | 2 | withheld | not `operator_max_gain_rate_level2` |
| `generator_symmetric_anisotropy_level3` | 3 | withheld | not `numerical_abscissa_level3` |
| `divergence_level3` | 3 | `/dynamical_families/one_step/v1/divergence_level3` | trace of the generator proxy |
| `generator_rotation_norm_level3` | 3 | `/dynamical_families/one_step/v1/generator_rotation_norm_level3` | Frobenius norm of the skew part |
| `affine_one_step_map_level2/declared_lag_2` | 2 | `/dynamical_families/one_step/v1/declared_lag_2/affine_one_step_map_level2` | **direct** lag-2 map when `declared_lags` includes 2; same object class as lag 1; not \(\Phi_1^2\) |
| `operator_max_gain_rate_level2/declared_lag_2` | 2 | `/dynamical_families/one_step/v1/declared_lag_2/operator_max_gain_rate_level2` | SVD max-gain of \(\Phi_2\); not spectral abscissa; not peak gain |
| `operator_volume_gain_rate_level2/declared_lag_2` | 2 | `/dynamical_families/one_step/v1/declared_lag_2/operator_volume_gain_rate_level2` | \(\log\lvert\det\Phi_2\rvert/\mathrm{nominal\_dt}\); rank-deficient maps fail closed |
| `operator_rotation_rate_level2/declared_lag_2` | 2 | `/dynamical_families/one_step/v1/declared_lag_2/operator_rotation_rate_level2` | polar rotation of \(\Phi_2\); not generator-proxy rotation |
| `spectral_abscissa_level3/declared_lag_2` | 3 | `/dynamical_families/one_step/v1/declared_lag_2/spectral_abscissa_level3` | \(\max\mathrm{Re}\,\mathrm{eig}(\logm(\Phi_2)/\mathrm{nominal\_dt})\); only if the lag-2 map is identified; not Itô |
| `numerical_abscissa_level3/declared_lag_2` | 3 | `/dynamical_families/one_step/v1/declared_lag_2/numerical_abscissa_level3` | max eigenvalue of the symmetric part of the same lag-2 generator proxy |
| `divergence_level3/declared_lag_2` | 3 | `/dynamical_families/one_step/v1/declared_lag_2/divergence_level3` | trace of the lag-2 generator proxy |
| `generator_rotation_norm_level3/declared_lag_2` | 3 | `/dynamical_families/one_step/v1/declared_lag_2/generator_rotation_norm_level3` | Frobenius norm of the skew part |
| `iterated_one_step_horizon_map_level4` | 4 | `/dynamical_families/one_step/v1/iterated_one_step_horizon_map_level4` | \(\Phi_1\) applied twice; own blocked holdout at horizon \(2\Delta t\); `qualification_status=horizon_propagation_identified`; not the direct lag-2 map |

Level-3 leaves use `qualification_status=generator_proxy_from_qualified_one_step_not_ito`.
`logm` must be real; there is no silent Euler fallback.
`ito_drift_level3` / `ito_qualified` are never written. `reactivity_gap_level3`
is withheld; abscissa proxies are not that gap. Iterating \(\Phi\)
is level 4 (`iterated_one_step_horizon_map_level4`) and is not the lag-2
identity. The family is not a `drift_source` for
\(A_{bD}\). 3D subject-anchored chart only.

## Amplification (neighbor gain)

`neighbor_gain_q90_level1` is a window-level observed statistic of
same-pair Euclidean neighbor distances: neighbors are selected among
lag-1 sources and followed one real step without re-kNN at the
successor. The written value at each source is the 0.90 quantile of
that source's pair gains
\((d_{ij}^{(1)}+\varepsilon)/(d_{ij}^{(0)}+\varepsilon)\). This is a
window series, not a pooled pair-level \(Q_{0.90}\) over the recording. \(\varepsilon\)
is a documented distance floor, not a volume rescue. It is not
`operator_max_gain_rate_level2`, not spectral abscissa, and not peak
gain. `history_predictive_gain_level1` is a separate family, not written
here. Nested same-pair identities `neighbor_separation_rate_level1`,
`neighbor_gain_rate_q90_level1`, and `cloud_volume_change_rate_level1`
write when the family is enabled; they are not YAML toggles. Common YAML
`amplification.enabled` is false. I-CARE overlays do not enable this
family except named amplification pilots.

| measurement_id | level | physical path | notes |
|---|---|---|---|
| `neighbor_gain_q90_level1` | 1 | `/dynamical_families/amplification/v1/neighbor_gain_q90_level1` | same-pair observed gain; `qualification_status=same_pair_observed_gain_not_operator_max_gain` |
| `neighbor_separation_rate_level1` | 1 | `/dynamical_families/amplification/v1/neighbor_separation_rate_level1` | per-source median log-ratio / \(\Delta t\); `qualification_status=same_pair_observed_separation_not_spectral_abscissa` |
| `neighbor_gain_rate_q90_level1` | 1 | `/dynamical_families/amplification/v1/neighbor_gain_rate_q90_level1` | \(\log G_{q90}/\Delta t\) of the written q90; `qualification_status=same_pair_observed_gain_rate_not_operator_max_gain` |
| `cloud_volume_change_rate_level1` | 1 | `/dynamical_families/amplification/v1/cloud_volume_change_rate_level1` | same-cloud logdet rate; `qualification_status=same_pair_cloud_volume_not_operator_volume`; \(\varepsilon\) is a logdet floor |

## History (predictive gain)

`history_predictive_gain_level1` is a recording-level out-of-sample
error reduction on the same lag-1 triples:

* \(M_0\): frozen ridge affine \(x_t\to x_{t+1}\)
* \(M_1\): frozen ridge affine \((x_t,x_{t-1})\to x_{t+1}\)

\(H_{\mathrm{gain}}=\mathrm{MSE}(M_0)-\mathrm{MSE}(M_1)\). Both maps
are fit on two chronological blocked-holdout folds with an index
embargo (`embargo_semantics=index_steps`, `embargo_steps=4`). The
index embargo does not establish raw-window or filter independence
(`raw_window_support_independence=not_established`). The
comparison does not restore Markovianity. Nested
`history_conditioned_operator_level2` is identified only when M1 itself
passes the frozen one-step OOS gate (`mndm.one_step_fit_fidelity.v1`,
median fold rel-MSE strictly less than 0.9). Positive \(H_{\mathrm{gain}}\)
is not identification. The identified map is \(3\times 6\), not lag-1
\(\Phi\). Grain is `native=recording`, `repeated_measure=false`. Common YAML
`history.enabled` is false. I-CARE production overlays and amplification
pilots do not enable this family. Named
`*_history_turning_pilot.yaml` overlays may set it true for coverage. `n_blocks` is frozen at 2.
`embargo_steps` is frozen at 4. The 0.9 threshold is frozen. Both OOS
folds must succeed. Series `source_idx` / `transition_support_id` are
the filtered triples, not the unfiltered lag-1 pair set
(`lag1_transition_support_id`). YAML key
`history_conditioned_operator_level2` is refused as a toggle. 3D
subject-anchored only.

| measurement_id | level | physical path | notes |
|---|---|---|---|
| `history_predictive_gain_level1` | 1 | `/dynamical_families/history/v1/history_predictive_gain_level1` | OOS MSE reduction; `qualification_status=history_error_reduction_not_markov_restoration` |
| `history_conditioned_operator_level2` | 2 | `/dynamical_families/history/v1/history_conditioned_operator_level2` | M1 affine identified by the frozen 0.9 OOS gate; not lag-1 \(\Phi\) |
| `history_augmented_generator_level3` | 3 | withheld | not `logm` of \(3\times 6\) M1 and not `ito_drift_level3` |
| `history_augmented_propagator_level4` | 4 | withheld | not iteration of M1 and not `iterated_one_step_horizon_map_level4` |

## Turning (realized rotation)

`turning_angle_level0` is the Euclidean angle between consecutive
lag-1 displacements \(v_t=x_{t+1}-x_t\) and \(v_{t+1}=x_{t+2}-x_{t+1}\).
`turning_rate_level0` is that angle divided by the observed first-step
\(\Delta t\). If either \(\lVert v\rVert\) is below the declared
`min_displacement`, the value is undefined (NaN), not zero. Grain is
`native=window`, `repeated_measure=true`. Common YAML `turning.enabled`
is false. I-CARE production overlays and amplification pilots do not
enable this family. Named `*_history_turning_pilot.yaml` overlays may
set it true for coverage.
`cloud_volume_change_rate_level1` writes under amplification, not this
family.

| measurement_id | level | physical path | notes |
|---|---|---|---|
| `turning_angle_level0` | 0 | `/dynamical_families/turning/v1/turning_angle_level0` | realized angle; `qualification_status=realized_turning_not_operator_rotation` |
| `turning_rate_level0` | 0 | `/dynamical_families/turning/v1/turning_rate_level0` | angle / observed \(\Delta t\); undefined not zero |

## Compatibility (SL-LEV-MES-003 §8)

Machine-readable rows live in `COMPATIBILITY_ROWS`. This markdown table is
the 003 drift/diffusion landing plus nested amplification/history/turning
identities. Destination and resilience identities are in the register and
in the family sections. `/local_dynamics/` is a logical 003 proposal, not
a writable HDF5 root. Nested `dynamical_families.local_dynamics` is
refused. Top-level YAML `local_dynamics.stochastic_reachability` remains
the opt-in Gate F switch and writes `/stochastic_reachability/v1`.
Nested `crossfit_local_chart_b` under `dynamical_families.drift` is
refused; `drift.source: crossfit_local_chart_b` stays resolver-closed
(`crossfit_not_authorized_before_m3`) and is not an independent \(b\).

| export or historical name | formula / estimand | identity | relation |
|---|---|---|---|
| `/mnps_3d_dot` | Savitzky–Golay of `/mnps_3d` | `smoothed_velocity_savgol_level0` | `existing_dataset_documented_identity` |
| `smoothed_velocity_savgol_level0` | same | `/mnps_3d_dot` | `existing_dataset_documented_identity` |
| `realized_velocity_level0` | \((x_{t+1}-x_t)/\Delta t_{\mathrm{obs}}\) | `realized_velocity_level0` | `new_measure_not_alias` |
| `conditional_mean_rate_level1/pooled` | weighted mean increment / nominal \(\Delta t\) | `conditional_mean_rate_level1` / `pooled` | `new_measure_not_alias` |
| `b_hat` | same as pooled | `conditional_mean_rate_level1` / `pooled` | `new_measure_not_alias` |
| `conditional_mean_rate_level1/blocked_crossfit` | same estimand, index-embargoed subset | same `measurement_id` | `same_measure_variant` |
| `conditional_mean_rate_level1/lag_diagnostics` | multi-lag consistency | same `measurement_id` | `diagnostics_not_level_upgrade` |
| `a_hat` | centered `np.cov(..., ddof=1)` / nominal \(\Delta t\) | `conditional_covariance_rate_level1` | `documented_identity_of_existing` |
| `conditional_covariance_rate_level1` | same as `a_hat` | `/series/a_hat` | `documented_identity_of_existing` |
| `diffusion_tensor` | same array as `a_hat` | `conditional_covariance_rate_level1` | `series_alias_same_array` |
| `D_total` | trace of \(a\) | scalar of `a_hat` | `derived_scalar_of_a` |
| `d_diff` | anisotropy of \(a\) | scalar of `a_hat` | `derived_scalar_of_a`; not `diffusion_effective_dimension_level1` |
| `c_diff` | concentration of \(a\) | scalar of `a_hat` | `derived_scalar_of_a`; not `diffusion_condition_number_level1` |
| `A_bD` | alignment scalar | ingest C1 `not_testable` | `alignment_scalar_not_testable` |
| `R_b_over_a` | ratio scalar | ingest C1 `not_testable` | `alignment_scalar_not_testable` |
| `lag1_source_transitions` | source indices | support object | `support_object_not_measurement` |
| `increment_covariance_level0` | unconditional centered \(\mathrm{Cov}(\Delta X)\) | `increment_covariance_level0` | `new_measure_not_alias` |
| `innovation_covariance_level2` | affine residual covariance / nominal \(\Delta t\) | `innovation_covariance_level2` | `new_measure_not_alias` |
| `ito_drift_level3` | — | withheld | `withheld_not_written` |
| `ito_diffusion_tensor_level3` | — | withheld | `withheld_not_written` |
| `finite_lag` | — | diary 397 name | `superseded_unreleased_name_no_alias` |
| `crossfit` | — | diary 397 name | `superseded_unreleased_name_no_alias` |
| `crossfit_local_chart_b` | C1 SOURCE_CROSSFIT token | withheld | nested YAML refused; `drift.source` stays `not_testable` |
| `ito_candidate` | — | diary 397 name | `superseded_unreleased_name_no_alias` |
| `/local_dynamics` | 003 §9 proposal | not a writable root | `logical_path_not_written` |
| `neighbor_gain_q90_level1` | q90 of same-pair \((d_1+\varepsilon)/(d_0+\varepsilon)\) | `neighbor_gain_q90_level1` | `new_measure_not_alias` |
| `neighbor_separation_rate_level1` | median \(\log((d_1+\varepsilon)/(d_0+\varepsilon))/\Delta t\) | `neighbor_separation_rate_level1` | `new_measure_not_alias` |
| `neighbor_gain_rate_q90_level1` | \(\log G_{q90}/\Delta t\) of the written q90 | `neighbor_gain_rate_q90_level1` | `new_measure_not_alias` |
| `cloud_volume_change_rate_level1` | \((\log\det(C_1+\varepsilon I)-\log\det(C_0+\varepsilon I))/(2\Delta t)\) | `cloud_volume_change_rate_level1` | `new_measure_not_alias` |
| `history_predictive_gain_level1` | OOS MSE(\(M_0\))-MSE(\(M_1\)) on the same triples | `history_predictive_gain_level1` | `new_measure_not_alias` |
| `history_conditioned_operator_level2` | history-augmented affine \(3\times 6\) OOS-identified | `history_conditioned_operator_level2` | `new_measure_not_alias` |
| `history_augmented_generator_level3` | — | withheld | `withheld_not_written` |
| `history_augmented_propagator_level4` | — | withheld | `withheld_not_written` |
| `finite_time_peak_gain_level4` | — | withheld | `withheld_not_written`; not \(\Phi\) powers and not `iterated_one_step_horizon_map_level4` |
| `ito_qualified` | — | withheld | `withheld_not_written`; status token, not a measurement |
| `reactivity_gap_level3` | — | withheld | `withheld_not_written`; not jacobian-metrics `reactivity_gap` |
| `diffusion_effective_dimension_level1` | — | withheld | `withheld_not_written`; not `d_diff` |
| `operator_gain_anisotropy_level2` | — | withheld | `withheld_not_written`; not operator max-gain |
| `reachability_effective_dimension_level4` | — | withheld | `withheld_not_written`; not `d_eff` of \(W_Q\) |
| `turning_rate_level0` | successive-increment angle / observed \(\Delta t\) | `turning_rate_level0` | `new_measure_not_alias` |
| `turning_angle_level0` | successive-increment angle | `turning_angle_level0` | `new_measure_not_alias` |

## Synthetic pilot (SL-LEV-MES-003 §8)

Bounded synthetic checks live in `mndm/tests/test_df_003_pilot.py`. They
exercise the frozen identities on a known linear SDE, a Brownian null,
NaN gaps, a rank-1 chart with PSD floor, insufficient local support,
denied Itô qualification that still leaves realized / pooled / `a_hat`
computed, and irregular \(\Delta t\). Realized velocity stays
`qualification_status=not_assessed`; pooled and `a_hat` stay
`ito_not_qualified`. The common overlay stays `enabled: false`. The
I-CARE overlay is audited only and is not flipped or re-run for this
naming gate. Denied qualification does not write `ito_drift_level3` or
`ito_diffusion_tensor_level3`.

## Destination (committor)

The committor module requires explicit, disjoint A/B regime sets and multiple
independent segments containing both regimes. Stage labels alone are not
committor truth. Do not export anything simply called `committor` as a
measurement_id.

Production ingest uses the 1-D O2b adapter (`local_law_dense_grid_o2b`)
only. That `q_A_to_B` is the documented identity
`restricted_1d_local_law_quadrature_q` (no `_levelN` suffix): constant-D
quadrature on an explicit reaction coordinate. It is not
`generator_committor_level3` and not a 2-D/3-D/9D MNPS committor.
`q_hat` is a same-array alias. The adapter uses an internal 1-D potential
for quadrature and does **not** serialize \(V_{1/2}\). Missing
`interpretation_level` on HDF5 means the object is unnumbered; the written
token is `summary/interpretation_level_token=not_numbered`.

The first-hit estimator (`local_first_hit_outcome_average`) serializes
the same series name `q_A_to_B` as
`destination_first_hit_fraction_resolved_level1` (003: x-conditioned is
level1, not `destination_first_hit_fraction_level0`). It averages
resolved A/B hits only. `destination_hit_probability_level1` (including
unresolved) and `destination_unresolved_fraction_level1` are withheld.
`series/resolved_first_hit_outcome` is a 0/1/NaN encoding, not the
unresolved-fraction leaf. `summary/n_resolved_first_hit_outcomes` is a
count, not that leaf. YAML nested names and substitutions
(`resolved_q_as_destination_hit_probability_level1`,
`include_unresolved_in_first_hit_q`) are refused.
Neither estimator emits \(\lvert\nabla q\rvert\).
`transition_model_first_hit_probability_level2` is a superseded 002 name
when H composes steps; the 003 identity is withheld level4.
`/possible_futures/destination/` is a logical path, not a writable root.
Denied destination qualification leaves the O2b identity marked
`not_testable` and does not write zeros or `generator_committor_level3`.

## Resilience (finite-amplitude)

Resilience accepts observed finite-perturbation outcomes. It does not infer
basin stability from spontaneous trajectories, a Jacobian, or finite-time
tangent gain. Existing `amplitude_curve` / `basin_return_probability` is the
documented identity `far_recovery_probability_level4`. `return_fraction` is
the same numeric field. Standard retrospective ingest reports
`no_perturbation_protocol` when requested without explicit perturbation
data; that refusal still carries the FAR identity and empty series, not
`spontaneous_return_fraction_level0`.

`spontaneous_return_fraction_level0`, `excursion_recovery_probability_level1`,
and `excursion_recovery_time_median_level1` are withheld and are not FAR.
`matched_perturbation_recovery_level2` is a superseded 002 name (003: matching
is not automatically level2). The existing
`r50_discrete_first_bin_at_or_below_half` is a nested diagnostic on the FAR
family summary, not `far_threshold_p50_level4` (the 002 supremum). That
scalar keeps its own estimand; it is not a FAR threshold. Nested YAML
substitutions `amplitude_curve_as_spontaneous_return` and
`r50_as_far_threshold_p50` are refused. `/perturbation/`
is a logical path, not a writable root. A computed result additionally
requires an explicit outcome protocol and a valid family qualification
certificate.

## Persistence / attractor geometry

Ingest does not write attractor, basin, persistence, or recurrence groups.
`state_return_probability_level0`, `local_recurrence_rate_level0`,
`region_survival_probability_level1`, and `region_dwell_time_level1` are
withheld observational / persistence names and are not attractors. Visual
clustering does not establish an attractor.
`transition_self_retention_level2` is withheld; the one_step family does
not write retention. `transition_escape_rate_level2` is withheld:
\(-\log(P_{RR})/\Delta t\) is transformed retention, not an exit rate, and
being back in \(R\) at the next sample is not staying in \(R\) for the
whole interval. `basin_attractor_geometry_level4` is reserved for a
validated long-horizon structure and is not written. There is no level 3.
`/state_geometry/` and `/possible_futures/persistence/` are logical paths.
YAML keys `attractor`, `basin`, `persistence`, `recurrence`, and
`state_geometry` are refused.

## Reachability / future spread

Family YAML `spread` remains Gate F `gate_closed`. Opt-in discrete `W_Q`
is `/stochastic_reachability/v1`, not `/dynamical_families/spread`. Existing
`w_q` is discrete Lyapunov predictive spread of Gate E Q through \(\Phi\).
One-step (`n_propagator_steps=1`) is the documented identity
`transition_reachability_covariance_level2`. Composed horizons and ingest
refusals are `finite_time_reachability_level4`. This is not controllability,
occupancy, empirical future covariance, or generator spread.
`observed_future_spread_level0` is superseded: state-matched future
covariance is `conditional_future_spread_level1` and is not written.
Existing `d_eff` / `v_norm` are derived scalars of `W_Q`, not the 002
observational dimension/volume names. Nested YAML
`w_q_as_conditional_future_spread_level1` is refused.
`/possible_futures/spread/` is logical.

## Hysteresis / recovery

Observational hysteresis and return-path comparison are withheld register
identities. There is no physical `/dynamical_families/hysteresis` group and
no physical `/perturbation/` tree. `return_distance_level0` is a descriptive
post-to-baseline chart distance. `matched_return_distance_level1` and
`recovery_time_level1` are baseline/state-matched observational comparisons.
`recovery_time_level1` is not `excursion_recovery_time_median_level1`.
`induction_recovery_path_asymmetry_level2` is superseded: matching and
aligned up/down trajectories are not automatically a discrete model.
The 003 preferred name `induction_recovery_path_asymmetry_level1` is also
not written. Hysteresis is not FAR. `/possible_futures/hysteresis/` is
logical; `/perturbation/` remains FAR-only. YAML keys `hysteresis`,
`recovery`, and the 002/003 measurement names are refused.
