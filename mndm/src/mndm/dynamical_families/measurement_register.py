"""Logical measurement register (SL-LEV-MES-003, then 002 groups).

This is a documented mapping, not a new HDF5 root. Physical paths stay under
``/dynamical_families/{drift,diffusion,destination,one_step,amplification,history,turning,resilience}/v1``,
``/stochastic_reachability/v1``, and ``/mnps_3d_dot``.
Qualified names such as ``ito_drift_level3``, ``ito_diffusion_tensor_level3``,
``generator_committor_level3``, ``reactivity_gap_level3``, and the 001
anisotropy names are withheld. Existing ``a_hat``, ``d_diff``, ``d_eff``,
``q_A_to_B``, and resilience ``amplitude_curve`` / ``basin_return_probability``
are documented identities or derived scalars, not those 001 names.
Spontaneous return is not FAR. Jacobian-metrics ``reactivity_gap`` is
omega-alpha of ``J_hat``, not ``reactivity_gap_level3``.
``COMPATIBILITY_ROWS`` maps export and historical names to a closed relation
token. It does not treat a computed destination ``q`` as a generator committor
or a spontaneous excursion as FAR. Recurrence and region survival are not
an attractor. ``-log(P_RR)/Δ`` is transformed retention, not an escape rate.
State-matched future spread is level1, not level0. Existing ``W_Q`` is
discrete Lyapunov predictive spread, not controllability or occupancy.
"""

from __future__ import annotations

from typing import Any, Mapping

PHYSICAL_DRIFT_ROOT = "/dynamical_families/drift/v1"
PHYSICAL_MNPS_DOT = "/mnps_3d_dot"
PHYSICAL_DIFFUSION_ROOT = "/dynamical_families/diffusion/v1"
PHYSICAL_DIFFUSION_A_HAT = f"{PHYSICAL_DIFFUSION_ROOT}/series/a_hat"
PHYSICAL_DIFFUSION_SOURCE_IDX = f"{PHYSICAL_DIFFUSION_ROOT}/series/source_idx"
PHYSICAL_DESTINATION_ROOT = "/dynamical_families/destination/v1"
PHYSICAL_DESTINATION_Q = f"{PHYSICAL_DESTINATION_ROOT}/series/q_A_to_B"
PHYSICAL_DESTINATION_OUTCOME = f"{PHYSICAL_DESTINATION_ROOT}/series/resolved_first_hit_outcome"
PHYSICAL_DESTINATION_N_RESOLVED = (
    f"{PHYSICAL_DESTINATION_ROOT}/summary/n_resolved_first_hit_outcomes"
)
PHYSICAL_ONE_STEP_ROOT = "/dynamical_families/one_step/v1"
PHYSICAL_AMPLIFICATION_ROOT = "/dynamical_families/amplification/v1"
PHYSICAL_HISTORY_ROOT = "/dynamical_families/history/v1"
PHYSICAL_TURNING_ROOT = "/dynamical_families/turning/v1"
PHYSICAL_RESILIENCE_ROOT = "/dynamical_families/resilience/v1"
PHYSICAL_RESILIENCE_CURVE = f"{PHYSICAL_RESILIENCE_ROOT}/amplitude_curve"
DISTANCE_EUCLIDEAN_CHART = "euclidean_on_release_scaled_chart"
DIFFUSION_CONVENTION_A = "a_not_D_over_2"

MEASUREMENT_ID_SMOOTHED_VELOCITY = "smoothed_velocity_savgol_level0"
MEASUREMENT_ID_REALIZED_VELOCITY = "realized_velocity_level0"
MEASUREMENT_ID_CONDITIONAL_MEAN_RATE = "conditional_mean_rate_level1"
MEASUREMENT_ID_INCREMENT_COVARIANCE = "increment_covariance_level0"
MEASUREMENT_ID_CONDITIONAL_COVARIANCE = "conditional_covariance_rate_level1"
MEASUREMENT_ID_INNOVATION_COVARIANCE = "innovation_covariance_level2"
MEASUREMENT_ID_AFFINE_MAP = "affine_one_step_map_level2"
MEASUREMENT_ID_AFFINE_MEAN_RATE = "conditional_affine_mean_rate_level2"
MEASUREMENT_ID_OPERATOR_MAX_GAIN = "operator_max_gain_rate_level2"
MEASUREMENT_ID_OPERATOR_VOLUME_GAIN = "operator_volume_gain_rate_level2"
MEASUREMENT_ID_OPERATOR_ROTATION_RATE = "operator_rotation_rate_level2"
MEASUREMENT_ID_NEIGHBOR_GAIN = "neighbor_gain_q90_level1"
MEASUREMENT_ID_NEIGHBOR_SEPARATION = "neighbor_separation_rate_level1"
MEASUREMENT_ID_NEIGHBOR_GAIN_RATE = "neighbor_gain_rate_q90_level1"
MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN = "history_predictive_gain_level1"
MEASUREMENT_ID_HISTORY_OPERATOR = "history_conditioned_operator_level2"
MEASUREMENT_ID_HISTORY_GENERATOR = "history_augmented_generator_level3"
MEASUREMENT_ID_HISTORY_PROPAGATOR = "history_augmented_propagator_level4"
MEASUREMENT_ID_TURNING_RATE = "turning_rate_level0"
MEASUREMENT_ID_TURNING_ANGLE = "turning_angle_level0"
MEASUREMENT_ID_CLOUD_VOLUME = "cloud_volume_change_rate_level1"
MEASUREMENT_ID_SPECTRAL_ABSCISSA = "spectral_abscissa_level3"
MEASUREMENT_ID_NUMERICAL_ABSCISSA = "numerical_abscissa_level3"
MEASUREMENT_ID_DIVERGENCE = "divergence_level3"
MEASUREMENT_ID_GENERATOR_ROTATION = "generator_rotation_norm_level3"
MEASUREMENT_ID_ITERATED_ONE_STEP_MAP = "iterated_one_step_horizon_map_level4"
MEASUREMENT_ID_PEAK_GAIN = "finite_time_peak_gain_level4"
MEASUREMENT_ID_REACTIVITY_GAP = "reactivity_gap_level3"
MEASUREMENT_ID_DIFFUSION_DEFF = "diffusion_effective_dimension_level1"
MEASUREMENT_ID_DIFFUSION_COND = "diffusion_condition_number_level1"
MEASUREMENT_ID_DIFFUSION_ENTROPY = "diffusion_directional_entropy_level1"
MEASUREMENT_ID_OPERATOR_GAIN_ANISO = "operator_gain_anisotropy_level2"
MEASUREMENT_ID_GENERATOR_SYM_ANISO = "generator_symmetric_anisotropy_level3"
MEASUREMENT_ID_REACH_DEFF_L4 = "reachability_effective_dimension_level4"
MEASUREMENT_ID_REACH_ANISO_L4 = "reachability_anisotropy_level4"
MEASUREMENT_ID_ITO_DRIFT = "ito_drift_level3"
MEASUREMENT_ID_ITO_DIFFUSION = "ito_diffusion_tensor_level3"
MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0 = "destination_first_hit_fraction_level0"
MEASUREMENT_ID_DESTINATION_HIT_L1 = "destination_hit_probability_level1"
MEASUREMENT_ID_DESTINATION_RESOLVED = "destination_first_hit_fraction_resolved_level1"
MEASUREMENT_ID_DESTINATION_UNRESOLVED = "destination_unresolved_fraction_level1"
MEASUREMENT_ID_O2B_QUADRATURE = "restricted_1d_local_law_quadrature_q"
MEASUREMENT_ID_TRANSITION_HIT_L2 = "transition_model_first_hit_probability_level2"
MEASUREMENT_ID_TRANSITION_HIT_L4 = "transition_model_first_hit_probability_level4"
MEASUREMENT_ID_GENERATOR_COMMITTOR = "generator_committor_level3"
MEASUREMENT_ID_SPONTANEOUS_RETURN = "spontaneous_return_fraction_level0"
MEASUREMENT_ID_EXCURSION_RECOVERY = "excursion_recovery_probability_level1"
MEASUREMENT_ID_EXCURSION_RECOVERY_TIME = "excursion_recovery_time_median_level1"
MEASUREMENT_ID_MATCHED_RECOVERY_L2 = "matched_perturbation_recovery_level2"
MEASUREMENT_ID_MATCHED_DISPLACEMENT_L2 = "matched_perturbation_displacement_level2"
MEASUREMENT_ID_MATCHED_RECOVERY_L1 = "matched_perturbation_recovery_level1"
MEASUREMENT_ID_FAR_RECOVERY = "far_recovery_probability_level4"
MEASUREMENT_ID_FAR_P50 = "far_threshold_p50_level4"
MEASUREMENT_ID_FAR_P90 = "far_threshold_p90_level4"
MEASUREMENT_ID_R50_DISCRETE = "discrete_first_bin_at_or_below_half"
MEASUREMENT_ID_STATE_RETURN = "state_return_probability_level0"
MEASUREMENT_ID_LOCAL_RECURRENCE = "local_recurrence_rate_level0"
MEASUREMENT_ID_REGION_SURVIVAL = "region_survival_probability_level1"
MEASUREMENT_ID_REGION_DWELL = "region_dwell_time_level1"
MEASUREMENT_ID_SELF_RETENTION = "transition_self_retention_level2"
MEASUREMENT_ID_ESCAPE_RATE = "transition_escape_rate_level2"
MEASUREMENT_ID_BASIN_ATTRACTOR = "basin_attractor_geometry_level4"
MEASUREMENT_ID_OBS_FUTURE_SPREAD = "observed_future_spread_level0"
MEASUREMENT_ID_OBS_FUTURE_DEFF = "observed_future_effective_dimension_level0"
MEASUREMENT_ID_OBS_FUTURE_LOGVOL = "observed_future_log_volume_level0"
MEASUREMENT_ID_OBS_FUTURE_ANISO = "observed_future_anisotropy_level0"
MEASUREMENT_ID_COND_FUTURE_SPREAD = "conditional_future_spread_level1"
MEASUREMENT_ID_TRANSITION_REACH_COV = "transition_reachability_covariance_level2"
MEASUREMENT_ID_TRANSITION_REACH_DEFF = "transition_reachability_effective_dimension_level2"
MEASUREMENT_ID_GENERATOR_SPREAD = "generator_predictive_spread_level3"
MEASUREMENT_ID_FINITE_TIME_REACH = "finite_time_reachability_level4"
MEASUREMENT_ID_RETURN_DISTANCE = "return_distance_level0"
MEASUREMENT_ID_MATCHED_RETURN_DISTANCE = "matched_return_distance_level1"
MEASUREMENT_ID_RECOVERY_TIME_L1 = "recovery_time_level1"
MEASUREMENT_ID_PATH_ASYMMETRY_L2 = "induction_recovery_path_asymmetry_level2"
MEASUREMENT_ID_PATH_ASYMMETRY_L1 = "induction_recovery_path_asymmetry_level1"
PHYSICAL_WQ_ROOT = "/stochastic_reachability/v1"
PHYSICAL_WQ_PRIMARY = f"{PHYSICAL_WQ_ROOT}/primary"
PHYSICAL_WQ_W = f"{PHYSICAL_WQ_PRIMARY}/w_q"
PHYSICAL_JACOBIAN_REACTIVITY_GAP = "/jacobian/derived_metrics/v1/series/reactivity_gap"
PHYSICAL_DIFFUSION_DEFF_ALIAS = f"{PHYSICAL_DIFFUSION_ROOT}/series/diffusion_effective_dimension"

VARIANT_POOLED = "pooled"
VARIANT_BLOCKED_CROSSFIT = "blocked_crossfit"
VARIANT_LAG_DIAGNOSTICS = "lag_diagnostics"
VARIANT_DECLARED_LAG_2 = "declared_lag_2"
VARIANT_DECLARED_LAG_STEPS_1 = "declared_lag_steps=1"
VARIANT_DECLARED_LAG_STEPS_2 = "declared_lag_steps=2"
VARIANT_HORIZON_STEPS_2 = "horizon_steps=2"
PHYSICAL_ONE_STEP_LAG2_ROOT = f"{PHYSICAL_ONE_STEP_ROOT}/{VARIANT_DECLARED_LAG_2}"
PHYSICAL_ONE_STEP_HORIZON_MAP = (
    f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_ITERATED_ONE_STEP_MAP}"
)
ONE_STEP_LAG2_MAP = f"{MEASUREMENT_ID_AFFINE_MAP}/{VARIANT_DECLARED_LAG_2}"
ONE_STEP_LAG2_MEAN_RATE = f"{MEASUREMENT_ID_AFFINE_MEAN_RATE}/{VARIANT_DECLARED_LAG_2}"
ONE_STEP_LAG2_INNOVATION = f"{MEASUREMENT_ID_INNOVATION_COVARIANCE}/{VARIANT_DECLARED_LAG_2}"
ONE_STEP_LAG2_SPECTRAL = f"{MEASUREMENT_ID_SPECTRAL_ABSCISSA}/{VARIANT_DECLARED_LAG_2}"
ONE_STEP_LAG2_NUMERICAL = f"{MEASUREMENT_ID_NUMERICAL_ABSCISSA}/{VARIANT_DECLARED_LAG_2}"
ONE_STEP_LAG2_DIVERGENCE = f"{MEASUREMENT_ID_DIVERGENCE}/{VARIANT_DECLARED_LAG_2}"
ONE_STEP_LAG2_ROTATION = f"{MEASUREMENT_ID_GENERATOR_ROTATION}/{VARIANT_DECLARED_LAG_2}"
ONE_STEP_LAG2_MAX_GAIN = f"{MEASUREMENT_ID_OPERATOR_MAX_GAIN}/{VARIANT_DECLARED_LAG_2}"
ONE_STEP_LAG2_VOLUME_GAIN = f"{MEASUREMENT_ID_OPERATOR_VOLUME_GAIN}/{VARIANT_DECLARED_LAG_2}"
ONE_STEP_LAG2_ROTATION_RATE = f"{MEASUREMENT_ID_OPERATOR_ROTATION_RATE}/{VARIANT_DECLARED_LAG_2}"
ALLOWED_ONE_STEP_DECLARED_LAGS = (1, 2)
DEFAULT_ONE_STEP_DECLARED_LAGS = (1,)

PHYSICAL_POOLED_SOURCE_IDX = (
    f"{PHYSICAL_DRIFT_ROOT}/{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}/"
    f"{VARIANT_POOLED}/series/source_idx"
)
PHYSICAL_CROSSFIT_SOURCE_IDX = (
    f"{PHYSICAL_DRIFT_ROOT}/{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}/"
    f"{VARIANT_BLOCKED_CROSSFIT}/series/source_idx"
)

CLAIM_DESCRIPTIVE = "descriptive_trajectory"
CLAIM_CONDITIONAL = "conditional_finite_lag"
CLAIM_DISCRETE_ONE_STEP = "discrete_one_step"
CLAIM_GENERATOR = "infinitesimal_generator"
CLAIM_CONDITIONAL_HORIZON = "conditional_finite_horizon"
CLAIM_RESTRICTED_1D_QUADRATURE = "restricted_1d_reaction_coordinate_quadrature"
CLAIM_HITTING_PROBABILITY = "hitting_probability"
CLAIM_FINITE_AMPLITUDE_RESPONSE = "finite_amplitude_response"
CLAIM_FINITE_HORIZON_PROPAGATION = "finite_horizon_propagation"

ESTIMAND_PER_STEP_DX_DT = "per_step_forward_difference_over_observed_dt"
ESTIMAND_MEAN_INCREMENT_NOMINAL_DT = "mean_increment_over_nominal_dt"
ESTIMAND_LAG_DIAGNOSTIC = "multi_lag_conditional_mean_rate_consistency"
ESTIMAND_UNCONDITIONAL_INCREMENT_COV = "unconditional_increment_covariance"
ESTIMAND_CENTERED_INCREMENT_COV_NOMINAL_DT = "centered_increment_covariance_over_nominal_dt"
ESTIMAND_INNOVATION_COV = "affine_residual_increment_covariance"
ESTIMAND_AFFINE_MAP = "recording_affine_next_state_map"
ESTIMAND_AFFINE_MEAN_RATE = "affine_one_step_predicted_increment_over_nominal_dt"
ESTIMAND_AFFINE_LAG2_MAP = "recording_affine_direct_lag2_next_state_map"
ESTIMAND_AFFINE_LAG2_MEAN_RATE = "affine_direct_lag2_predicted_increment_over_nominal_2dt"
ESTIMAND_INNOVATION_LAG2 = "affine_direct_lag2_residual_increment_covariance"
ESTIMAND_ITERATED_ONE_STEP_MAP = "composed_lag1_affine_map_applied_twice"
ESTIMAND_SPECTRAL_ABSCISSA = "max_real_eigenvalue_of_matrix_log_phi_over_dt"
ESTIMAND_NUMERICAL_ABSCISSA = "max_eigenvalue_of_symmetric_part_of_matrix_log_phi_over_dt"
ESTIMAND_DIVERGENCE = "trace_of_matrix_log_phi_over_dt"
ESTIMAND_GENERATOR_ROTATION = "frobenius_norm_of_skew_part_of_matrix_log_phi_over_dt"
ESTIMAND_OPERATOR_MAX_GAIN = "log_max_singular_value_of_phi_over_dt"
ESTIMAND_OPERATOR_VOLUME_GAIN = "log_abs_det_phi_over_dt"
ESTIMAND_OPERATOR_ROTATION_RATE = "frobenius_log_polar_rotation_over_sqrt2_dt"
ESTIMAND_NEIGHBOR_GAIN_Q90 = (
    "per_source_q90_of_same_pair_successor_distance_over_source_distance"
)
ESTIMAND_NEIGHBOR_SEPARATION = (
    "median_same_pair_log_distance_ratio_over_nominal_dt"
)
ESTIMAND_NEIGHBOR_GAIN_RATE = "log_per_source_q90_neighbor_gain_over_nominal_dt"
ESTIMAND_CLOUD_VOLUME = "same_pair_cloud_logdet_change_over_two_nominal_dt"
ESTIMAND_HISTORY_PREDICTIVE_GAIN = "oos_error_reduction_from_one_lag_of_history"
ESTIMAND_HISTORY_OPERATOR = "history_augmented_affine_next_state_map_oos_identified"
ESTIMAND_TURNING_RATE = "successive_increment_angle_over_observed_dt"
ESTIMAND_TURNING_ANGLE = "successive_increment_angle"
ESTIMAND_SAVGOL = "savgol_derivative_of_mnps_3d"
ESTIMAND_TRACE_OF_A = "trace_of_centered_increment_covariance_rate"
ESTIMAND_ANISOTROPY_OF_A = "eigenvalue_anisotropy_of_centered_increment_covariance_rate"
ESTIMAND_ALIGNMENT_ABD = "chart_velocity_alignment_to_increment_covariance"
ESTIMAND_RATIO_R = "dt_times_drift_norm_sq_over_trace_a"
ESTIMAND_RESOLVED_FIRST_HIT = "local_mean_of_resolved_first_hit_outcomes"
ESTIMAND_O2B_QUADRATURE = "one_d_constant_diffusion_quadrature_on_explicit_reaction_coordinate"
ESTIMAND_GENERATOR_COMMITTOR = "generator_pde_committor_on_chart"
ESTIMAND_FAR_RETURN_FRACTION = "observed_return_fraction_by_perturbation_amplitude"
ESTIMAND_R50_DISCRETE = "first_amplitude_bin_with_return_fraction_at_or_below_half"
ESTIMAND_FAR_THRESHOLD_SUP = "supremum_amplitude_with_recovery_at_least_p"
ESTIMAND_STATE_RETURN = "observed_return_to_same_quantized_state"
ESTIMAND_LOCAL_RECURRENCE = "epsilon_ball_recurrence_rate"
ESTIMAND_REGION_SURVIVAL = "probability_stay_in_frozen_region_for_horizon"
ESTIMAND_REGION_DWELL = "first_exit_time_from_frozen_region"
ESTIMAND_SELF_RETENTION = "one_step_self_transition_probability"
ESTIMAND_TRANSFORMED_RETENTION = "minus_log_self_retention_over_dt"
ESTIMAND_BASIN_ATTRACTOR = "validated_long_horizon_basin_geometry"
ESTIMAND_EMPIRICAL_FUTURE_COV = "state_matched_future_endpoint_covariance"
ESTIMAND_LYAPUNOV_W = "discrete_lyapunov_propagation_of_gate_e_q"
ESTIMAND_GENERATOR_SPREAD = "infinitesimal_generator_predictive_covariance"
ESTIMAND_RETURN_DISTANCE = "chart_distance_post_to_baseline"
ESTIMAND_MATCHED_RETURN = "baseline_state_matched_return_distance"
ESTIMAND_HYSTERESIS_RECOVERY_TIME = "baseline_state_matched_return_time"
ESTIMAND_PATH_ASYMMETRY = "aligned_up_down_path_integral"

RELATION_DOCUMENTED_IDENTITY = "documented_identity_of_existing"
RELATION_NEW_MEASURE = "new_measure_not_alias"
RELATION_EXISTING_DATASET = "existing_dataset_documented_identity"
RELATION_SAME_VARIANT = "same_measure_variant"
RELATION_DIAGNOSTICS = "diagnostics_not_level_upgrade"
RELATION_SERIES_ALIAS = "series_alias_same_array"
RELATION_DERIVED_SCALAR = "derived_scalar_of_a"
RELATION_ALIGNMENT = "alignment_scalar_not_testable"
RELATION_SUPPORT_OBJECT = "support_object_not_measurement"
RELATION_WITHHELD = "withheld_not_written"
RELATION_SUPERSEDED = "superseded_unreleased_name_no_alias"
RELATION_LOGICAL = "logical_path_not_written"

COMPATIBILITY_RELATIONS = frozenset(
    {
        RELATION_DOCUMENTED_IDENTITY,
        RELATION_NEW_MEASURE,
        RELATION_EXISTING_DATASET,
        RELATION_SAME_VARIANT,
        RELATION_DIAGNOSTICS,
        RELATION_SERIES_ALIAS,
        RELATION_DERIVED_SCALAR,
        RELATION_ALIGNMENT,
        RELATION_SUPPORT_OBJECT,
        RELATION_WITHHELD,
        RELATION_SUPERSEDED,
        RELATION_LOGICAL,
    }
)

LOGICAL_LOCAL_DYNAMICS_ROOT = "/local_dynamics"
LOGICAL_POSSIBLE_FUTURES_ROOT = "/possible_futures"
LOGICAL_POSSIBLE_FUTURES_DESTINATION = "/possible_futures/destination"
LOGICAL_PERTURBATION_ROOT = "/perturbation"
LOGICAL_POSSIBLE_FUTURES_PERTURBATION = "/possible_futures/perturbation"
LOGICAL_STATE_GEOMETRY_ROOT = "/state_geometry"
LOGICAL_POSSIBLE_FUTURES_PERSISTENCE = "/possible_futures/persistence"
LOGICAL_POSSIBLE_FUTURES_RECURRENCE = "/possible_futures/recurrence"
LOGICAL_POSSIBLE_FUTURES_SPREAD = "/possible_futures/spread"
LOGICAL_POSSIBLE_FUTURES_HYSTERESIS = "/possible_futures/hysteresis"

QUALIFICATION_NOT_ASSESSED = "not_assessed"
QUALIFICATION_ITO_NOT_QUALIFIED = "ito_not_qualified"
QUALIFICATION_ONE_STEP_IDENTIFIED = "one_step_identified"
QUALIFICATION_ONE_STEP_NOT_IDENTIFIED = "one_step_not_identified"
QUALIFICATION_GENERATOR_PROXY_NOT_ITO = "generator_proxy_from_qualified_one_step_not_ito"
QUALIFICATION_ONE_STEP_FUNCTIONAL = "one_step_functional_of_qualified_map_not_independent_oos"
QUALIFICATION_SAME_PAIR_GAIN = "same_pair_observed_gain_not_operator_max_gain"
QUALIFICATION_SAME_PAIR_SEPARATION = "same_pair_observed_separation_not_spectral_abscissa"
QUALIFICATION_SAME_PAIR_GAIN_RATE = "same_pair_observed_gain_rate_not_operator_max_gain"
QUALIFICATION_CLOUD_VOLUME = "same_pair_cloud_volume_not_operator_volume"
QUALIFICATION_INCREMENT_COV = "unconditional_increment_covariance_not_a_hat"
QUALIFICATION_HISTORY_GAIN = "history_error_reduction_not_markov_restoration"
QUALIFICATION_HISTORY_OPERATOR_IDENTIFIED = (
    "history_conditioned_operator_identified_not_markov_restoration"
)
QUALIFICATION_HISTORY_OPERATOR_NOT_IDENTIFIED = "history_conditioned_operator_not_identified"
QUALIFICATION_TURNING = "realized_turning_not_operator_rotation"
QUALIFICATION_HORIZON_PROPAGATION_IDENTIFIED = "horizon_propagation_identified"
QUALIFICATION_HORIZON_PROPAGATION_NOT_IDENTIFIED = "horizon_propagation_not_identified"
QUALIFICATION_NOT_GENERATOR_COMMITTOR = "not_generator_committor"
QUALIFICATION_NOT_FAR = "not_far"
QUALIFICATION_NOT_ATTRACTOR = "not_attractor"
QUALIFICATION_NOT_CONTROLLABILITY = "not_controllability"
SUPPORT_SUFFICIENT = "sufficient"
SUPPORT_INSUFFICIENT = "insufficient_local_support"
SUPPORT_NOT_APPLICABLE = "not_applicable"

FORBIDDEN_DIFFUSION_CONFIG_KEYS = (
    "increment_covariance_level0",
    "ito_diffusion_tensor_level3",
    "ito_qualified",
    "diffusion_effective_dimension_level1",
    "diffusion_condition_number_level1",
    "diffusion_directional_entropy_level1",
    "d_diff_as_diffusion_effective_dimension_level1",
)
FORBIDDEN_DRIFT_CONFIG_KEYS = (
    "finite_lag",
    "crossfit",
    "ito_candidate",
    "ito_drift_level3",
    "ito_qualified",
    "crossfit_local_chart_b",
    "source_crossfit_as_independent_b",
)
FORBIDDEN_ONE_STEP_CONFIG_KEYS = (
    "ito_drift_level3",
    "ito_qualified",
    "jacobian_expm",
    "jacobian_expm_as_fitted_phi",
    "multi_step_rollout",
    "one_step_iteration_as_level2",
    "abscissa_from_unidentified_operator",
    "innovation_as_diffusion_a_hat",
    "mnps_xdot_as_one_step_map",
    "phi_one_composed_as_lag2",
    "affine_two_step",
    "two_step",
    "affine_two_step_map_level2",
    "peak_gain_from_phi_powers",
    "finite_time_peak_gain_level4",
    "operator_gain_as_spectral_abscissa",
    "epsilon_rescued_volume_gain",
    "reactivity_gap_level3",
    "abscissa_difference_as_reactivity_gap",
    "jacobian_reactivity_gap_as_level3",
    "operator_gain_anisotropy_level2",
    "generator_symmetric_anisotropy_level3",
)
FORBIDDEN_AMPLIFICATION_CONFIG_KEYS = (
    "resample_neighbors_at_target",
    "neighbor_gain_as_operator_max_gain",
    "neighbor_gain_as_spectral_abscissa",
    "history_predictive_gain_level1",
    "neighbor_separation_rate_level1",
    "neighbor_gain_rate_q90_level1",
    "cloud_volume_change_rate_level1",
    "neighbor_separation_as_spectral_abscissa",
    "cloud_volume_as_operator_volume",
)
FORBIDDEN_HISTORY_CONFIG_KEYS = (
    "history_as_markov_restoration",
    "history_conditioned_operator_level2",
    "history_augmented_generator_level3",
    "history_augmented_propagator_level4",
    "mnps_xdot_as_history_model",
    "one_step_phi_as_history_m0",
    "logm_of_history_m1",
    "history_m1_as_ito_drift",
    "m1_iteration_as_history_propagator",
)
FORBIDDEN_TURNING_CONFIG_KEYS = (
    "operator_rotation_as_turning",
    "generator_rotation_as_turning",
    "mnps_xdot_as_turning",
    "zero_fill_undefined_direction",
    "cloud_volume_change_rate_level1",
)
FORBIDDEN_DESTINATION_CONFIG_KEYS = (
    "generator_committor_level3",
    "transition_model_first_hit_probability_level2",
    "transition_model_first_hit_probability_level4",
    "destination_first_hit_fraction_level0",
    "destination_hit_probability_level1",
    "destination_unresolved_fraction_level1",
    "committor",
    "resolved_q_as_destination_hit_probability_level1",
    "unresolved_outcomes_as_destination_unresolved_fraction_level1",
    "include_unresolved_in_first_hit_q",
)
FORBIDDEN_RESILIENCE_CONFIG_KEYS = (
    "spontaneous_return_fraction_level0",
    "excursion_recovery_probability_level1",
    "excursion_recovery_time_median_level1",
    "matched_perturbation_recovery_level2",
    "matched_perturbation_recovery_level1",
    "far_threshold_p50_level4",
    "far_threshold_p90_level4",
    "amplitude_curve_as_spontaneous_return",
    "r50_as_far_threshold_p50",
)
FORBIDDEN_PERSISTENCE_CONFIG_KEYS = (
    "attractor",
    "basin",
    "persistence",
    "recurrence",
    "state_geometry",
    "basin_attractor_geometry_level4",
    "state_return_probability_level0",
    "local_recurrence_rate_level0",
    "region_survival_probability_level1",
    "region_dwell_time_level1",
    "transition_self_retention_level2",
    "transition_escape_rate_level2",
    "state_recurrence_level0",
    "residence_persistence_level1",
    "transition_metastability_level2",
)
FORBIDDEN_SPREAD_CONFIG_KEYS = (
    "observed_future_spread_level0",
    "observed_future_effective_dimension_level0",
    "observed_future_log_volume_level0",
    "observed_future_anisotropy_level0",
    "conditional_future_spread_level1",
    "transition_reachability_covariance_level2",
    "transition_reachability_effective_dimension_level2",
    "transition_predictive_spread_level2",
    "generator_predictive_spread_level3",
    "finite_time_reachability_level4",
    "reachability_effective_dimension_level4",
    "reachability_anisotropy_level4",
    "d_eff_as_reachability_effective_dimension_level4",
    "w_q_as_conditional_future_spread_level1",
)
FORBIDDEN_LOGICAL_ROOT_KEYS = (
    "local_dynamics",
    "possible_futures",
    "perturbation",
    "/local_dynamics",
    "/possible_futures",
    "/perturbation",
)
FORBIDDEN_HYSTERESIS_CONFIG_KEYS = (
    "hysteresis",
    "recovery",
    "return_distance_level0",
    "matched_return_distance_level1",
    "recovery_time_level1",
    "induction_recovery_path_asymmetry_level2",
    "induction_recovery_path_asymmetry_level1",
)


def reject_forbidden_diffusion_config_keys(diffusion_cfg: Mapping[str, Any] | None) -> None:
    """Refuse nested increment covariance as a YAML toggle."""
    if not isinstance(diffusion_cfg, Mapping):
        return
    found = [key for key in FORBIDDEN_DIFFUSION_CONFIG_KEYS if key in diffusion_cfg]
    if found:
        raise ValueError(
            "dynamical_families.diffusion keys "
            f"{found} are refused; increment_covariance_level0 is a nested "
            "identity of the diffusion family, not a YAML toggle. "
            "ito_diffusion_tensor_level3 and ito_qualified are withheld. "
            "No YAML alias is provided."
        )


def reject_forbidden_drift_config_keys(drift_cfg: Mapping[str, Any] | None) -> None:
    """Refuse unreleased nested names. No silent alias is provided."""
    if not isinstance(drift_cfg, Mapping):
        return
    found = [key for key in FORBIDDEN_DRIFT_CONFIG_KEYS if key in drift_cfg]
    if found:
        raise ValueError(
            "dynamical_families.drift keys "
            f"{found} are refused; current identities are "
            "realized_velocity_level0 and "
            "conditional_mean_rate_level1/"
            "{pooled,blocked_crossfit,lag_diagnostics}. "
            "ito_drift_level3 and ito_qualified are withheld; there is no "
            "auto-promotion. crossfit_local_chart_b is closed before M3 and "
            "is not an independent b for A_bD. No YAML alias is provided."
        )


def reject_forbidden_one_step_config_keys(one_step_cfg: Mapping[str, Any] | None) -> None:
    """Refuse Itô auto-promotion and Jacobian-expm aliases. No silent alias."""
    if not isinstance(one_step_cfg, Mapping):
        return
    found = [key for key in FORBIDDEN_ONE_STEP_CONFIG_KEYS if key in one_step_cfg]
    if found:
        raise ValueError(
            "dynamical_families.one_step keys "
            f"{found} are refused; the writable identities are "
            "affine_one_step_map_level2, conditional_affine_mean_rate_level2, "
            "innovation_covariance_level2, spectral_abscissa_level3, "
            "numerical_abscissa_level3, divergence_level3, and "
            "generator_rotation_norm_level3. Direct lag-2 identities write "
            "under declared_lag_2/ when declared_lags includes 2. "
            "Phi_1 applied twice is iterated_one_step_horizon_map_level4, "
            "not the lag-2 identity. "
            "operator_max_gain_rate_level2 is SVD of Phi, not spectral "
            "abscissa and not finite_time_peak_gain_level4. Rank-deficient "
            "volume is not epsilon-rescued. "
            "ito_drift_level3 and ito_qualified are withheld. Phi composed "
            "twice is iterated_one_step_horizon_map_level4, not peak gain "
            "from Phi powers. No YAML alias is provided."
        )


def reject_forbidden_amplification_config_keys(
    amplification_cfg: Mapping[str, Any] | None,
) -> None:
    """Refuse operator-gain aliases and successor re-selection."""
    if not isinstance(amplification_cfg, Mapping):
        return
    found = [key for key in FORBIDDEN_AMPLIFICATION_CONFIG_KEYS if key in amplification_cfg]
    if found:
        raise ValueError(
            "dynamical_families.amplification keys "
            f"{found} are refused; neighbor_gain_q90_level1 writes when the "
            "family is enabled. Nested same-pair identities are not YAML "
            "toggles. Neighbors are not re-selected at the successor. "
            "history_predictive_gain_level1 is a separate family. No YAML "
            "alias is provided."
        )


def reject_forbidden_history_config_keys(history_cfg: Mapping[str, Any] | None) -> None:
    """Refuse Markov-restoration and operator/generator aliases."""
    if not isinstance(history_cfg, Mapping):
        return
    found = [key for key in FORBIDDEN_HISTORY_CONFIG_KEYS if key in history_cfg]
    if found:
        raise ValueError(
            "dynamical_families.history keys "
            f"{found} are refused; history_predictive_gain_level1 writes "
            "when the family is enabled. history_conditioned_operator_level2 "
            "is a nested identity of that family, not a YAML toggle. Frozen "
            "M0/M1 OOS error reduction is not Markov restoration. "
            "history_augmented_generator_level3 is not logm of the 3x6 M1 "
            "map and not ito_drift_level3. "
            "history_augmented_propagator_level4 is not iteration of M1. "
            "No YAML alias is provided."
        )


def reject_forbidden_turning_config_keys(turning_cfg: Mapping[str, Any] | None) -> None:
    """Refuse operator/generator aliases and cloud-volume expansion."""
    if not isinstance(turning_cfg, Mapping):
        return
    found = [key for key in FORBIDDEN_TURNING_CONFIG_KEYS if key in turning_cfg]
    if found:
        raise ValueError(
            "dynamical_families.turning keys "
            f"{found} are refused; the writable identities are "
            "turning_rate_level0 and turning_angle_level0. Insufficient "
            "displacement is undefined, not zero. Cloud-volume expansion "
            "is not written. No YAML alias is provided."
        )


def parse_one_step_declared_lags(one_step_cfg: Mapping[str, Any] | None) -> tuple[int, ...]:
    """Return implemented declared lags. Missing key means lag 1 only."""
    reject_unimplemented_one_step_declared_lags(one_step_cfg)
    if not isinstance(one_step_cfg, Mapping) or "declared_lags" not in one_step_cfg:
        return DEFAULT_ONE_STEP_DECLARED_LAGS
    raw = one_step_cfg.get("declared_lags")
    return tuple(sorted({int(value) for value in raw}))


def reject_unimplemented_one_step_declared_lags(
    one_step_cfg: Mapping[str, Any] | None,
) -> None:
    """Refuse declared_lags outside the implemented set {1, 2}."""
    if not isinstance(one_step_cfg, Mapping) or "declared_lags" not in one_step_cfg:
        return
    raw = one_step_cfg.get("declared_lags")
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "dynamical_families.one_step declared_lags must be a list of "
            "positive integers; implemented lags are [1] and [1, 2]. "
            "Composing the lag-1 map is not the lag-2 identity."
        )
    try:
        lags = tuple(int(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "dynamical_families.one_step declared_lags must be a list of "
            "positive integers; implemented lags are [1] and [1, 2]."
        ) from exc
    if any(type(value) is not int or int(value) != value for value in raw):
        raise ValueError(
            "dynamical_families.one_step declared_lags must be a list of "
            "positive integers; implemented lags are [1] and [1, 2]."
        )
    if not lags or any(lag < 1 for lag in lags):
        raise ValueError(
            "dynamical_families.one_step declared_lags must be a non-empty list of "
            "positive integers; implemented lags are [1] and [1, 2]."
        )
    if len(set(lags)) != len(lags):
        raise ValueError(
            "dynamical_families.one_step declared_lags must not contain duplicates."
        )
    unknown = [lag for lag in lags if lag not in ALLOWED_ONE_STEP_DECLARED_LAGS]
    if unknown:
        raise ValueError(
            "dynamical_families.one_step declared_lags "
            f"{list(unknown)} are not implemented. Direct lag-2 identities live "
            "under declared_lag_2/. Composing the lag-1 map is not that "
            "identity (phi_one_composed_as_lag2 / one_step_iteration_as_level2)."
        )


def reject_forbidden_destination_config_keys(destination_cfg: Mapping[str, Any] | None) -> None:
    """Refuse generator-committor YAML names. No silent alias is provided."""
    if not isinstance(destination_cfg, Mapping):
        return
    found = [key for key in FORBIDDEN_DESTINATION_CONFIG_KEYS if key in destination_cfg]
    if found:
        raise ValueError(
            "dynamical_families.destination keys "
            f"{found} are refused; ingest writes /dynamical_families/destination/v1 "
            "series/q_A_to_B under an estimator-specific identity "
            "(restricted_1d_local_law_quadrature_q or "
            "destination_first_hit_fraction_resolved_level1). "
            "generator_committor_level3 is withheld. "
            "destination_hit_probability_level1 (including unresolved) and "
            "destination_unresolved_fraction_level1 are withheld. "
            "003: x-conditioned first-hit is level1, not "
            "destination_first_hit_fraction_level0. No YAML alias is provided."
        )


def reject_forbidden_resilience_config_keys(resilience_cfg: Mapping[str, Any] | None) -> None:
    """Refuse observational names as FAR YAML. No silent alias is provided."""
    if not isinstance(resilience_cfg, Mapping):
        return
    found = [
        key
        for key in (*FORBIDDEN_RESILIENCE_CONFIG_KEYS, *FORBIDDEN_HYSTERESIS_CONFIG_KEYS)
        if key in resilience_cfg
    ]
    if found:
        raise ValueError(
            "dynamical_families.resilience keys "
            f"{found} are refused; ingest writes /dynamical_families/resilience/v1 "
            "amplitude_curve as far_recovery_probability_level4 when a perturbation "
            "protocol is present. Spontaneous return, excursion recovery, and "
            "hysteresis/recovery names are not FAR. far_threshold_p50_level4 and "
            "far_threshold_p90_level4 are withheld; existing r50 is "
            "discrete_first_bin_at_or_below_half. Amplitude_curve is not "
            "spontaneous_return_fraction_level0. No YAML alias is provided."
        )


def reject_forbidden_persistence_config_keys(family_root: Mapping[str, Any] | None) -> None:
    """Refuse attractor/basin/persistence YAML. No silent alias is provided."""
    if not isinstance(family_root, Mapping):
        return
    found = [key for key in FORBIDDEN_PERSISTENCE_CONFIG_KEYS if key in family_root]
    if found:
        raise ValueError(
            "dynamical_families keys "
            f"{found} are refused; ingest does not write attractor, basin, "
            "persistence, or recurrence. Recurrence is not an attractor. "
            "-log(P_RR)/dt is transformed retention, not an escape rate. "
            "Observational recurrence, region survival/dwell, and one-step "
            "retention names are also refused. No YAML alias is provided."
        )


def reject_forbidden_spread_config_keys(family_root: Mapping[str, Any] | None) -> None:
    """Refuse 002 future-spread YAML under dynamical_families. No alias."""
    if not isinstance(family_root, Mapping):
        return
    found = [key for key in FORBIDDEN_SPREAD_CONFIG_KEYS if key in family_root]
    if found:
        raise ValueError(
            "dynamical_families keys "
            f"{found} are refused; family YAML spread remains gate_closed. "
            "Existing W_Q is /stochastic_reachability/v1 via "
            "local_dynamics.stochastic_reachability. State-matched future "
            "spread is level1, not level0. Multi-step W_Q is not level2. "
            "W_Q is not conditional_future_spread_level1. "
            "No YAML alias is provided."
        )


def reject_forbidden_hysteresis_config_keys(family_root: Mapping[str, Any] | None) -> None:
    """Refuse 002 hysteresis/recovery YAML. No silent alias is provided."""
    if not isinstance(family_root, Mapping):
        return
    found = [key for key in FORBIDDEN_HYSTERESIS_CONFIG_KEYS if key in family_root]
    if found:
        raise ValueError(
            "dynamical_families keys "
            f"{found} are refused; ingest does not write hysteresis or "
            "observational recovery. Matching is not automatically level2. "
            "Hysteresis is not FAR. recovery_time_level1 is not "
            "excursion_recovery_time_median_level1. No YAML alias is provided."
        )


def reject_forbidden_logical_root_keys(family_root: Mapping[str, Any] | None) -> None:
    """Refuse 003 logical trees as dynamical_families YAML. No physical write."""
    if not isinstance(family_root, Mapping):
        return
    found = [key for key in FORBIDDEN_LOGICAL_ROOT_KEYS if key in family_root]
    if found:
        raise ValueError(
            "dynamical_families keys "
            f"{found} are logical 003 paths, not writable HDF5 roots. "
            "Physical writes stay under /dynamical_families and "
            "/stochastic_reachability/v1. No YAML alias is provided."
        )


MEASUREMENT_REGISTER: dict[str, dict[str, Any]] = {
    MEASUREMENT_ID_SMOOTHED_VELOCITY: {
        "measurement_id": MEASUREMENT_ID_SMOOTHED_VELOCITY,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": PHYSICAL_MNPS_DOT,
        "estimand": ESTIMAND_SAVGOL,
        "relation": RELATION_EXISTING_DATASET,
        "identity_relation": "existing_dataset_not_alias_of_realized_velocity",
        "not_sde_drift": True,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
        "written_by_drift_family": False,
    },
    MEASUREMENT_ID_REALIZED_VELOCITY: {
        "measurement_id": MEASUREMENT_ID_REALIZED_VELOCITY,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": f"{PHYSICAL_DRIFT_ROOT}/{MEASUREMENT_ID_REALIZED_VELOCITY}",
        "estimand": ESTIMAND_PER_STEP_DX_DT,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": RELATION_NEW_MEASURE,
        "not_sde_drift": True,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
        "written_by_drift_family": True,
    },
    MEASUREMENT_ID_CONDITIONAL_MEAN_RATE: {
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL,
        "physical_path": f"{PHYSICAL_DRIFT_ROOT}/{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}",
        "estimand": ESTIMAND_MEAN_INCREMENT_NOMINAL_DT,
        "relation": RELATION_NEW_MEASURE,
        "not_sde_drift": True,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
        "written_by_drift_family": True,
        "variants": (VARIANT_POOLED, VARIANT_BLOCKED_CROSSFIT, VARIANT_LAG_DIAGNOSTICS),
    },
    MEASUREMENT_ID_INCREMENT_COVARIANCE: {
        "measurement_id": MEASUREMENT_ID_INCREMENT_COVARIANCE,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": f"{PHYSICAL_DIFFUSION_ROOT}/{MEASUREMENT_ID_INCREMENT_COVARIANCE}",
        "estimand": ESTIMAND_UNCONDITIONAL_INCREMENT_COV,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "unconditional_increment_covariance_not_a_hat",
        "diffusion_convention": DIFFUSION_CONVENTION_A,
        "not_ito_diffusion_tensor": True,
        "not_sde_drift": True,
        "not_divided_by_dt": True,
        "not_local_knn": True,
        "default_qualification_status": QUALIFICATION_INCREMENT_COV,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
        "written_by_drift_family": False,
        "written_by_diffusion_family": True,
    },
    MEASUREMENT_ID_CONDITIONAL_COVARIANCE: {
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL,
        "physical_path": PHYSICAL_DIFFUSION_A_HAT,
        "estimand": ESTIMAND_CENTERED_INCREMENT_COV_NOMINAL_DT,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "identity_relation": "documented_identity_of_existing_a_hat_not_a_rename",
        "diffusion_convention": DIFFUSION_CONVENTION_A,
        "not_ito_diffusion_tensor": True,
        "not_sde_drift": True,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
        "written_by_drift_family": False,
        "written_by_diffusion_family": True,
    },
    MEASUREMENT_ID_INNOVATION_COVARIANCE: {
        "measurement_id": MEASUREMENT_ID_INNOVATION_COVARIANCE,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_INNOVATION_COVARIANCE}",
        "estimand": ESTIMAND_INNOVATION_COV,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "new_measure_affine_residual_not_diffusion_a_hat",
        "diffusion_convention": DIFFUSION_CONVENTION_A,
        "not_ito_diffusion_tensor": True,
        "not_sde_drift": True,
        "default_qualification_status": QUALIFICATION_ONE_STEP_NOT_IDENTIFIED,
        "written_by_drift_family": False,
        "written_by_diffusion_family": False,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_1,
        "declared_lag_steps": 1,
    },
    MEASUREMENT_ID_AFFINE_MAP: {
        "measurement_id": MEASUREMENT_ID_AFFINE_MAP,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_AFFINE_MAP}",
        "estimand": ESTIMAND_AFFINE_MAP,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "new_measure_not_expm_of_production_jacobian",
        "not_sde_drift": True,
        "default_qualification_status": QUALIFICATION_ONE_STEP_NOT_IDENTIFIED,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_1,
        "declared_lag_steps": 1,
    },
    MEASUREMENT_ID_AFFINE_MEAN_RATE: {
        "measurement_id": MEASUREMENT_ID_AFFINE_MEAN_RATE,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_AFFINE_MEAN_RATE}",
        "estimand": ESTIMAND_AFFINE_MEAN_RATE,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "new_measure_not_conditional_mean_rate_level1",
        "not_sde_drift": True,
        "default_qualification_status": QUALIFICATION_ONE_STEP_NOT_IDENTIFIED,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_1,
        "declared_lag_steps": 1,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    MEASUREMENT_ID_OPERATOR_MAX_GAIN: {
        "measurement_id": MEASUREMENT_ID_OPERATOR_MAX_GAIN,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_OPERATOR_MAX_GAIN}",
        "estimand": ESTIMAND_OPERATOR_MAX_GAIN,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "one_step_svd_gain_not_spectral_abscissa_and_not_peak_gain",
        "not_sde_drift": True,
        "not_independent_singular_oos": True,
        "not_spectral_abscissa": True,
        "not_peak_gain_level4": True,
        "default_qualification_status": QUALIFICATION_ONE_STEP_FUNCTIONAL,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_1,
        "declared_lag_steps": 1,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    MEASUREMENT_ID_OPERATOR_VOLUME_GAIN: {
        "measurement_id": MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_OPERATOR_VOLUME_GAIN}",
        "estimand": ESTIMAND_OPERATOR_VOLUME_GAIN,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "one_step_log_abs_det_not_generator_divergence",
        "not_sde_drift": True,
        "not_peak_gain_level4": True,
        "default_qualification_status": QUALIFICATION_ONE_STEP_FUNCTIONAL,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_1,
        "declared_lag_steps": 1,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    MEASUREMENT_ID_OPERATOR_ROTATION_RATE: {
        "measurement_id": MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_OPERATOR_ROTATION_RATE}",
        "estimand": ESTIMAND_OPERATOR_ROTATION_RATE,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "polar_rotation_of_phi_not_generator_skew",
        "not_sde_drift": True,
        "not_generator_rotation": True,
        "default_qualification_status": QUALIFICATION_ONE_STEP_FUNCTIONAL,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_1,
        "declared_lag_steps": 1,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    MEASUREMENT_ID_NEIGHBOR_GAIN: {
        "measurement_id": MEASUREMENT_ID_NEIGHBOR_GAIN,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL,
        "physical_path": f"{PHYSICAL_AMPLIFICATION_ROOT}/{MEASUREMENT_ID_NEIGHBOR_GAIN}",
        "estimand": ESTIMAND_NEIGHBOR_GAIN_Q90,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "same_pair_neighbor_gain_not_operator_max_gain",
        "not_sde_drift": True,
        "not_operator_max_gain": True,
        "not_spectral_abscissa": True,
        "not_peak_gain_level4": True,
        "same_pair_forward": True,
        "not_resampled_at_target": True,
        "default_qualification_status": QUALIFICATION_SAME_PAIR_GAIN,
        "written_by_amplification_family": True,
        "written_by_one_step_family": False,
        "declared_lag_steps": 1,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    MEASUREMENT_ID_NEIGHBOR_SEPARATION: {
        "measurement_id": MEASUREMENT_ID_NEIGHBOR_SEPARATION,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL,
        "physical_path": f"{PHYSICAL_AMPLIFICATION_ROOT}/{MEASUREMENT_ID_NEIGHBOR_SEPARATION}",
        "estimand": ESTIMAND_NEIGHBOR_SEPARATION,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "same_pair_log_distance_ratio_not_spectral_abscissa",
        "not_sde_drift": True,
        "not_spectral_abscissa": True,
        "same_pair_forward": True,
        "not_resampled_at_target": True,
        "default_qualification_status": QUALIFICATION_SAME_PAIR_SEPARATION,
        "written_by_amplification_family": True,
        "written_by_one_step_family": False,
        "declared_lag_steps": 1,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    MEASUREMENT_ID_NEIGHBOR_GAIN_RATE: {
        "measurement_id": MEASUREMENT_ID_NEIGHBOR_GAIN_RATE,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL,
        "physical_path": f"{PHYSICAL_AMPLIFICATION_ROOT}/{MEASUREMENT_ID_NEIGHBOR_GAIN_RATE}",
        "estimand": ESTIMAND_NEIGHBOR_GAIN_RATE,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "log_q90_gain_over_dt_not_operator_max_gain",
        "not_sde_drift": True,
        "not_operator_max_gain": True,
        "not_spectral_abscissa": True,
        "not_peak_gain_level4": True,
        "same_pair_forward": True,
        "not_resampled_at_target": True,
        "default_qualification_status": QUALIFICATION_SAME_PAIR_GAIN_RATE,
        "written_by_amplification_family": True,
        "written_by_one_step_family": False,
        "declared_lag_steps": 1,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN: {
        "measurement_id": MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL,
        "physical_path": f"{PHYSICAL_HISTORY_ROOT}/{MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN}",
        "estimand": ESTIMAND_HISTORY_PREDICTIVE_GAIN,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "oos_mse_reduction_m0_minus_m1_not_markov_restoration",
        "not_sde_drift": True,
        "not_markov_restoration": True,
        "not_one_step_operator": True,
        "default_qualification_status": QUALIFICATION_HISTORY_GAIN,
        "written_by_history_family": True,
        "written_by_amplification_family": False,
        "written_by_one_step_family": False,
        "declared_lag_steps": 1,
        "embargo_semantics": "index_steps",
        "min_embargo_steps": 4,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    MEASUREMENT_ID_HISTORY_OPERATOR: {
        "measurement_id": MEASUREMENT_ID_HISTORY_OPERATOR,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": f"{PHYSICAL_HISTORY_ROOT}/{MEASUREMENT_ID_HISTORY_OPERATOR}",
        "estimand": ESTIMAND_HISTORY_OPERATOR,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "history_augmented_affine_oos_not_markov_restoration",
        "not_sde_drift": True,
        "not_markov_restoration": True,
        "history_m1_not_lag1_phi": True,
        "default_qualification_status": QUALIFICATION_HISTORY_OPERATOR_NOT_IDENTIFIED,
        "written_by_history_family": True,
        "written_by_one_step_family": False,
        "declared_lag_steps": 1,
        "embargo_semantics": "index_steps",
        "min_embargo_steps": 4,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    MEASUREMENT_ID_HISTORY_GENERATOR: {
        "measurement_id": MEASUREMENT_ID_HISTORY_GENERATOR,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": None,
        "estimand": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_history_augmented_generator_not_logm_m1_not_ito",
        "not_sde_drift": True,
        "not_logm_of_m1": True,
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
        "written_by_history_family": False,
        "written_by_one_step_family": False,
        "written_by_drift_family": False,
    },
    MEASUREMENT_ID_HISTORY_PROPAGATOR: {
        "measurement_id": MEASUREMENT_ID_HISTORY_PROPAGATOR,
        "interpretation_level": 4,
        "claim_class": CLAIM_FINITE_HORIZON_PROPAGATION,
        "physical_path": None,
        "estimand": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_history_augmented_propagator_not_m1_iteration",
        "not_sde_drift": True,
        "not_m1_iteration": True,
        "not_composed_one_step": True,
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
        "written_by_history_family": False,
        "written_by_one_step_family": False,
    },
    MEASUREMENT_ID_TURNING_RATE: {
        "measurement_id": MEASUREMENT_ID_TURNING_RATE,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": f"{PHYSICAL_TURNING_ROOT}/{MEASUREMENT_ID_TURNING_RATE}",
        "estimand": ESTIMAND_TURNING_RATE,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "successive_increment_turning_rate_not_operator_rotation",
        "not_sde_drift": True,
        "not_operator_rotation": True,
        "not_generator_rotation": True,
        "undefined_not_zero": True,
        "default_qualification_status": QUALIFICATION_TURNING,
        "written_by_turning_family": True,
        "written_by_one_step_family": False,
        "declared_lag_steps": 1,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    MEASUREMENT_ID_TURNING_ANGLE: {
        "measurement_id": MEASUREMENT_ID_TURNING_ANGLE,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": f"{PHYSICAL_TURNING_ROOT}/{MEASUREMENT_ID_TURNING_ANGLE}",
        "estimand": ESTIMAND_TURNING_ANGLE,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "successive_increment_turning_angle_not_operator_rotation",
        "not_sde_drift": True,
        "not_operator_rotation": True,
        "not_generator_rotation": True,
        "undefined_not_zero": True,
        "default_qualification_status": QUALIFICATION_TURNING,
        "written_by_turning_family": True,
        "written_by_one_step_family": False,
        "declared_lag_steps": 1,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    MEASUREMENT_ID_CLOUD_VOLUME: {
        "measurement_id": MEASUREMENT_ID_CLOUD_VOLUME,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL,
        "physical_path": f"{PHYSICAL_AMPLIFICATION_ROOT}/{MEASUREMENT_ID_CLOUD_VOLUME}",
        "estimand": ESTIMAND_CLOUD_VOLUME,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "same_pair_epsilon_logdet_cloud_not_operator_volume",
        "not_sde_drift": True,
        "not_operator_volume": True,
        "epsilon_logdet_regularized": True,
        "same_pair_forward": True,
        "not_resampled_at_target": True,
        "default_qualification_status": QUALIFICATION_CLOUD_VOLUME,
        "written_by_amplification_family": True,
        "written_by_turning_family": False,
        "written_by_one_step_family": False,
        "declared_lag_steps": 1,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    ONE_STEP_LAG2_MAP: {
        "measurement_id": MEASUREMENT_ID_AFFINE_MAP,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_AFFINE_MAP}",
        "estimand": ESTIMAND_AFFINE_LAG2_MAP,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "new_measure_direct_lag2_not_composed_one_step",
        "not_sde_drift": True,
        "not_composed_one_step": True,
        "default_qualification_status": QUALIFICATION_ONE_STEP_NOT_IDENTIFIED,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "declared_lag_steps": 2,
        "embargo_semantics": "index_steps",
        "min_embargo_steps": 2,
    },
    ONE_STEP_LAG2_MEAN_RATE: {
        "measurement_id": MEASUREMENT_ID_AFFINE_MEAN_RATE,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_AFFINE_MEAN_RATE}",
        "estimand": ESTIMAND_AFFINE_LAG2_MEAN_RATE,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "new_measure_direct_lag2_not_composed_one_step",
        "not_sde_drift": True,
        "not_composed_one_step": True,
        "default_qualification_status": QUALIFICATION_ONE_STEP_NOT_IDENTIFIED,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "declared_lag_steps": 2,
        "embargo_semantics": "index_steps",
        "min_embargo_steps": 2,
    },
    ONE_STEP_LAG2_INNOVATION: {
        "measurement_id": MEASUREMENT_ID_INNOVATION_COVARIANCE,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_INNOVATION_COVARIANCE}",
        "estimand": ESTIMAND_INNOVATION_LAG2,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "new_measure_direct_lag2_not_composed_one_step",
        "not_ito_diffusion_tensor": True,
        "not_sde_drift": True,
        "not_composed_one_step": True,
        "default_qualification_status": QUALIFICATION_ONE_STEP_NOT_IDENTIFIED,
        "written_by_drift_family": False,
        "written_by_diffusion_family": False,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "declared_lag_steps": 2,
        "embargo_semantics": "index_steps",
        "min_embargo_steps": 2,
    },
    MEASUREMENT_ID_ITERATED_ONE_STEP_MAP: {
        "measurement_id": MEASUREMENT_ID_ITERATED_ONE_STEP_MAP,
        "interpretation_level": 4,
        "claim_class": CLAIM_FINITE_HORIZON_PROPAGATION,
        "physical_path": PHYSICAL_ONE_STEP_HORIZON_MAP,
        "estimand": ESTIMAND_ITERATED_ONE_STEP_MAP,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "new_measure_composed_lag1_not_direct_lag2",
        "not_sde_drift": True,
        "not_direct_lag2_map": True,
        "not_composed_one_step": False,
        "default_qualification_status": QUALIFICATION_HORIZON_PROPAGATION_NOT_IDENTIFIED,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_HORIZON_STEPS_2,
        "horizon_steps": 2,
        "embargo_semantics": "index_steps",
        "min_embargo_steps": 2,
    },
    f"{MEASUREMENT_ID_SPECTRAL_ABSCISSA}/{VARIANT_DECLARED_LAG_2}": {
        "measurement_id": MEASUREMENT_ID_SPECTRAL_ABSCISSA,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_SPECTRAL_ABSCISSA}",
        "estimand": ESTIMAND_SPECTRAL_ABSCISSA,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "generator_proxy_from_direct_lag2_map_not_ito",
        "not_sde_drift": True,
        "not_composed_one_step": True,
        "default_qualification_status": QUALIFICATION_GENERATOR_PROXY_NOT_ITO,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "declared_lag_steps": 2,
        "embargo_semantics": "index_steps",
        "min_embargo_steps": 2,
    },
    f"{MEASUREMENT_ID_NUMERICAL_ABSCISSA}/{VARIANT_DECLARED_LAG_2}": {
        "measurement_id": MEASUREMENT_ID_NUMERICAL_ABSCISSA,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_NUMERICAL_ABSCISSA}",
        "estimand": ESTIMAND_NUMERICAL_ABSCISSA,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "generator_proxy_from_direct_lag2_map_not_ito",
        "not_sde_drift": True,
        "not_composed_one_step": True,
        "default_qualification_status": QUALIFICATION_GENERATOR_PROXY_NOT_ITO,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "declared_lag_steps": 2,
        "embargo_semantics": "index_steps",
        "min_embargo_steps": 2,
    },
    f"{MEASUREMENT_ID_DIVERGENCE}/{VARIANT_DECLARED_LAG_2}": {
        "measurement_id": MEASUREMENT_ID_DIVERGENCE,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_DIVERGENCE}",
        "estimand": ESTIMAND_DIVERGENCE,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "generator_proxy_from_direct_lag2_map_not_ito",
        "not_sde_drift": True,
        "not_composed_one_step": True,
        "default_qualification_status": QUALIFICATION_GENERATOR_PROXY_NOT_ITO,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "declared_lag_steps": 2,
        "embargo_semantics": "index_steps",
        "min_embargo_steps": 2,
    },
    f"{MEASUREMENT_ID_GENERATOR_ROTATION}/{VARIANT_DECLARED_LAG_2}": {
        "measurement_id": MEASUREMENT_ID_GENERATOR_ROTATION,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_GENERATOR_ROTATION}",
        "estimand": ESTIMAND_GENERATOR_ROTATION,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "generator_proxy_from_direct_lag2_map_not_ito",
        "not_sde_drift": True,
        "not_composed_one_step": True,
        "default_qualification_status": QUALIFICATION_GENERATOR_PROXY_NOT_ITO,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "declared_lag_steps": 2,
        "embargo_semantics": "index_steps",
        "min_embargo_steps": 2,
    },
    ONE_STEP_LAG2_MAX_GAIN: {
        "measurement_id": MEASUREMENT_ID_OPERATOR_MAX_GAIN,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_OPERATOR_MAX_GAIN}",
        "estimand": ESTIMAND_OPERATOR_MAX_GAIN,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "one_step_svd_gain_from_direct_lag2_not_peak_gain",
        "not_sde_drift": True,
        "not_composed_one_step": True,
        "not_independent_singular_oos": True,
        "not_spectral_abscissa": True,
        "not_peak_gain_level4": True,
        "default_qualification_status": QUALIFICATION_ONE_STEP_FUNCTIONAL,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "declared_lag_steps": 2,
        "embargo_semantics": "index_steps",
        "min_embargo_steps": 2,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    ONE_STEP_LAG2_VOLUME_GAIN: {
        "measurement_id": MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_OPERATOR_VOLUME_GAIN}",
        "estimand": ESTIMAND_OPERATOR_VOLUME_GAIN,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "one_step_log_abs_det_from_direct_lag2_not_divergence",
        "not_sde_drift": True,
        "not_composed_one_step": True,
        "not_peak_gain_level4": True,
        "default_qualification_status": QUALIFICATION_ONE_STEP_FUNCTIONAL,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "declared_lag_steps": 2,
        "embargo_semantics": "index_steps",
        "min_embargo_steps": 2,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    ONE_STEP_LAG2_ROTATION_RATE: {
        "measurement_id": MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_OPERATOR_ROTATION_RATE}",
        "estimand": ESTIMAND_OPERATOR_ROTATION_RATE,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "polar_rotation_of_phi2_not_generator_skew",
        "not_sde_drift": True,
        "not_composed_one_step": True,
        "not_generator_rotation": True,
        "default_qualification_status": QUALIFICATION_ONE_STEP_FUNCTIONAL,
        "written_by_one_step_family": True,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "declared_lag_steps": 2,
        "embargo_semantics": "index_steps",
        "min_embargo_steps": 2,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
    },
    MEASUREMENT_ID_SPECTRAL_ABSCISSA: {
        "measurement_id": MEASUREMENT_ID_SPECTRAL_ABSCISSA,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_SPECTRAL_ABSCISSA}",
        "estimand": ESTIMAND_SPECTRAL_ABSCISSA,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "generator_proxy_not_jacobian_metrics_and_not_ito",
        "not_sde_drift": True,
        "default_qualification_status": QUALIFICATION_GENERATOR_PROXY_NOT_ITO,
        "written_by_one_step_family": True,
    },
    MEASUREMENT_ID_NUMERICAL_ABSCISSA: {
        "measurement_id": MEASUREMENT_ID_NUMERICAL_ABSCISSA,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_NUMERICAL_ABSCISSA}",
        "estimand": ESTIMAND_NUMERICAL_ABSCISSA,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "generator_proxy_not_jacobian_metrics_and_not_ito",
        "not_sde_drift": True,
        "default_qualification_status": QUALIFICATION_GENERATOR_PROXY_NOT_ITO,
        "written_by_one_step_family": True,
    },
    MEASUREMENT_ID_DIVERGENCE: {
        "measurement_id": MEASUREMENT_ID_DIVERGENCE,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_DIVERGENCE}",
        "estimand": ESTIMAND_DIVERGENCE,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "generator_proxy_not_ito",
        "not_sde_drift": True,
        "default_qualification_status": QUALIFICATION_GENERATOR_PROXY_NOT_ITO,
        "written_by_one_step_family": True,
    },
    MEASUREMENT_ID_GENERATOR_ROTATION: {
        "measurement_id": MEASUREMENT_ID_GENERATOR_ROTATION,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_GENERATOR_ROTATION}",
        "estimand": ESTIMAND_GENERATOR_ROTATION,
        "relation": RELATION_NEW_MEASURE,
        "identity_relation": "generator_proxy_not_ito",
        "not_sde_drift": True,
        "default_qualification_status": QUALIFICATION_GENERATOR_PROXY_NOT_ITO,
        "written_by_one_step_family": True,
    },
    MEASUREMENT_ID_ITO_DRIFT: {
        "measurement_id": MEASUREMENT_ID_ITO_DRIFT,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_until_independent_qualification",
        "not_sde_drift": True,
        "written_by_drift_family": False,
        "written_by_diffusion_family": False,
        "written_by_one_step_family": False,
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
    },
    MEASUREMENT_ID_ITO_DIFFUSION: {
        "measurement_id": MEASUREMENT_ID_ITO_DIFFUSION,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_until_independent_qualification",
        "diffusion_convention": DIFFUSION_CONVENTION_A,
        "not_ito_diffusion_tensor": True,
        "not_sde_drift": True,
        "written_by_drift_family": False,
        "written_by_diffusion_family": False,
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
    },
    MEASUREMENT_ID_PEAK_GAIN: {
        "measurement_id": MEASUREMENT_ID_PEAK_GAIN,
        "interpretation_level": 4,
        "claim_class": CLAIM_FINITE_HORIZON_PROPAGATION,
        "physical_path": None,
        "estimand": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_peak_gain_not_phi_powers_not_iterated_one_step",
        "not_sde_drift": True,
        "not_composed_one_step": True,
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
        "written_by_one_step_family": False,
    },
    MEASUREMENT_ID_REACTIVITY_GAP: {
        "measurement_id": MEASUREMENT_ID_REACTIVITY_GAP,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": None,
        "estimand": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_omega_minus_alpha_not_own_leaf_not_jacobian_gap",
        "not_sde_drift": True,
        "not_jacobian_metrics": True,
        "written_by_one_step_family": False,
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
    },
    MEASUREMENT_ID_DIFFUSION_DEFF: {
        "measurement_id": MEASUREMENT_ID_DIFFUSION_DEFF,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL,
        "physical_path": None,
        "estimand": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_001_name_not_existing_d_diff",
        "not_ito_diffusion_tensor": True,
        "written_by_diffusion_family": False,
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
    },
    MEASUREMENT_ID_DIFFUSION_COND: {
        "measurement_id": MEASUREMENT_ID_DIFFUSION_COND,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL,
        "physical_path": None,
        "estimand": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_001_name_not_existing_c_diff",
        "written_by_diffusion_family": False,
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
    },
    MEASUREMENT_ID_DIFFUSION_ENTROPY: {
        "measurement_id": MEASUREMENT_ID_DIFFUSION_ENTROPY,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL,
        "physical_path": None,
        "estimand": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_until_licensed_directional_entropy",
        "written_by_diffusion_family": False,
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
    },
    MEASUREMENT_ID_OPERATOR_GAIN_ANISO: {
        "measurement_id": MEASUREMENT_ID_OPERATOR_GAIN_ANISO,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": None,
        "estimand": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_not_operator_max_gain",
        "written_by_one_step_family": False,
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
    },
    MEASUREMENT_ID_GENERATOR_SYM_ANISO: {
        "measurement_id": MEASUREMENT_ID_GENERATOR_SYM_ANISO,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": None,
        "estimand": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_not_numerical_abscissa",
        "written_by_one_step_family": False,
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
    },
    MEASUREMENT_ID_REACH_DEFF_L4: {
        "measurement_id": MEASUREMENT_ID_REACH_DEFF_L4,
        "interpretation_level": 4,
        "claim_class": CLAIM_FINITE_HORIZON_PROPAGATION,
        "physical_path": None,
        "estimand": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_001_name_not_existing_d_eff",
        "not_controllability": True,
        "written_by_spread_family": False,
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
    },
    MEASUREMENT_ID_REACH_ANISO_L4: {
        "measurement_id": MEASUREMENT_ID_REACH_ANISO_L4,
        "interpretation_level": 4,
        "claim_class": CLAIM_FINITE_HORIZON_PROPAGATION,
        "physical_path": None,
        "estimand": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_001_name_not_existing_c_1_q",
        "not_controllability": True,
        "written_by_spread_family": False,
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
    },
    MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0: {
        "measurement_id": MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_state_conditioned_first_hit_is_not_level0",
        "default_qualification_status": QUALIFICATION_NOT_GENERATOR_COMMITTOR,
        "not_generator_committor": True,
        "not_resolved_first_hit": True,
        "written_by_destination_family": False,
    },
    MEASUREMENT_ID_DESTINATION_HIT_L1: {
        "measurement_id": MEASUREMENT_ID_DESTINATION_HIT_L1,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL_HORIZON,
        "physical_path": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_unresolved_included_hit_probability_not_written",
        "default_qualification_status": QUALIFICATION_NOT_GENERATOR_COMMITTOR,
        "not_generator_committor": True,
        "not_resolved_first_hit": True,
        "written_by_destination_family": False,
    },
    MEASUREMENT_ID_DESTINATION_RESOLVED: {
        "measurement_id": MEASUREMENT_ID_DESTINATION_RESOLVED,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL_HORIZON,
        "physical_path": PHYSICAL_DESTINATION_Q,
        "estimand": ESTIMAND_RESOLVED_FIRST_HIT,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "identity_relation": "documented_identity_of_first_hit_q_A_to_B_not_a_rename",
        "default_qualification_status": QUALIFICATION_NOT_GENERATOR_COMMITTOR,
        "not_generator_committor": True,
        "not_a_3d_mnps_committor": True,
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
        "written_by_destination_family": True,
        "ingest_estimator": "local_first_hit_outcome_average",
    },
    MEASUREMENT_ID_DESTINATION_UNRESOLVED: {
        "measurement_id": MEASUREMENT_ID_DESTINATION_UNRESOLVED,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL_HORIZON,
        "physical_path": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_unresolved_fraction_not_written",
        "default_qualification_status": QUALIFICATION_NOT_GENERATOR_COMMITTOR,
        "not_generator_committor": True,
        "not_resolved_outcome_encoding": True,
        "written_by_destination_family": False,
    },
    MEASUREMENT_ID_O2B_QUADRATURE: {
        "measurement_id": MEASUREMENT_ID_O2B_QUADRATURE,
        "interpretation_level": None,
        "claim_class": CLAIM_RESTRICTED_1D_QUADRATURE,
        "physical_path": PHYSICAL_DESTINATION_Q,
        "estimand": ESTIMAND_O2B_QUADRATURE,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "identity_relation": "documented_identity_of_o2b_q_A_to_B_not_generator_committor",
        "default_qualification_status": QUALIFICATION_NOT_GENERATOR_COMMITTOR,
        "not_generator_committor": True,
        "not_a_3d_mnps_committor": True,
        "interpretation_level_token": "not_numbered",
        "written_by_destination_family": True,
        "ingest_estimator": "local_law_dense_grid_o2b",
    },
    MEASUREMENT_ID_TRANSITION_HIT_L2: {
        "measurement_id": MEASUREMENT_ID_TRANSITION_HIT_L2,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": None,
        "relation": RELATION_SUPERSEDED,
        "identity_relation": "superseded_by_level4_when_horizon_composes_steps",
        "default_qualification_status": QUALIFICATION_NOT_GENERATOR_COMMITTOR,
        "not_generator_committor": True,
        "written_by_destination_family": False,
    },
    MEASUREMENT_ID_TRANSITION_HIT_L4: {
        "measurement_id": MEASUREMENT_ID_TRANSITION_HIT_L4,
        "interpretation_level": 4,
        "claim_class": "finite_horizon_propagation",
        "physical_path": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_until_validated_multi_step_transition_model",
        "default_qualification_status": QUALIFICATION_NOT_GENERATOR_COMMITTOR,
        "not_generator_committor": True,
        "written_by_destination_family": False,
    },
    MEASUREMENT_ID_GENERATOR_COMMITTOR: {
        "measurement_id": MEASUREMENT_ID_GENERATOR_COMMITTOR,
        "interpretation_level": 3,
        "claim_class": CLAIM_HITTING_PROBABILITY,
        "physical_path": None,
        "estimand": ESTIMAND_GENERATOR_COMMITTOR,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_until_independent_generator_qualification",
        "default_qualification_status": QUALIFICATION_NOT_GENERATOR_COMMITTOR,
        "not_generator_committor": True,
        "not_a_3d_mnps_committor": True,
        "written_by_destination_family": False,
    },
    MEASUREMENT_ID_SPONTANEOUS_RETURN: {
        "measurement_id": MEASUREMENT_ID_SPONTANEOUS_RETURN,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_observational_return_is_not_far",
        "default_qualification_status": QUALIFICATION_NOT_FAR,
        "not_far": True,
        "written_by_resilience_family": False,
    },
    MEASUREMENT_ID_EXCURSION_RECOVERY: {
        "measurement_id": MEASUREMENT_ID_EXCURSION_RECOVERY,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL_HORIZON,
        "physical_path": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_observational_resilience_curve_is_not_far",
        "default_qualification_status": QUALIFICATION_NOT_FAR,
        "not_far": True,
        "written_by_resilience_family": False,
    },
    MEASUREMENT_ID_EXCURSION_RECOVERY_TIME: {
        "measurement_id": MEASUREMENT_ID_EXCURSION_RECOVERY_TIME,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL_HORIZON,
        "physical_path": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_observational_recovery_time_is_not_far",
        "default_qualification_status": QUALIFICATION_NOT_FAR,
        "not_far": True,
        "written_by_resilience_family": False,
    },
    MEASUREMENT_ID_MATCHED_RECOVERY_L2: {
        "measurement_id": MEASUREMENT_ID_MATCHED_RECOVERY_L2,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": None,
        "relation": RELATION_SUPERSEDED,
        "identity_relation": "superseded_matching_is_not_automatically_level2",
        "default_qualification_status": QUALIFICATION_NOT_FAR,
        "not_far": True,
        "written_by_resilience_family": False,
    },
    MEASUREMENT_ID_MATCHED_DISPLACEMENT_L2: {
        "measurement_id": MEASUREMENT_ID_MATCHED_DISPLACEMENT_L2,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": None,
        "relation": RELATION_SUPERSEDED,
        "identity_relation": "superseded_matching_is_not_automatically_level2",
        "default_qualification_status": QUALIFICATION_NOT_FAR,
        "not_far": True,
        "written_by_resilience_family": False,
    },
    MEASUREMENT_ID_MATCHED_RECOVERY_L1: {
        "measurement_id": MEASUREMENT_ID_MATCHED_RECOVERY_L1,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL_HORIZON,
        "physical_path": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_003_matched_observational_comparison_not_written",
        "default_qualification_status": QUALIFICATION_NOT_FAR,
        "not_far": True,
        "written_by_resilience_family": False,
    },
    MEASUREMENT_ID_FAR_RECOVERY: {
        "measurement_id": MEASUREMENT_ID_FAR_RECOVERY,
        "interpretation_level": 4,
        "claim_class": CLAIM_FINITE_AMPLITUDE_RESPONSE,
        "physical_path": PHYSICAL_RESILIENCE_CURVE,
        "estimand": ESTIMAND_FAR_RETURN_FRACTION,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "identity_relation": "documented_identity_of_existing_amplitude_curve_not_a_rename",
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
        "not_inferred_from_spontaneous_trajectory": True,
        "written_by_resilience_family": True,
        "ingest_estimator": "observed_perturbation_outcome_summary",
    },
    MEASUREMENT_ID_FAR_P50: {
        "measurement_id": MEASUREMENT_ID_FAR_P50,
        "interpretation_level": 4,
        "claim_class": CLAIM_FINITE_AMPLITUDE_RESPONSE,
        "physical_path": None,
        "estimand": ESTIMAND_FAR_THRESHOLD_SUP,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_002_supremum_threshold_not_existing_r50",
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
        "not_existing_r50": True,
        "written_by_resilience_family": False,
    },
    MEASUREMENT_ID_FAR_P90: {
        "measurement_id": MEASUREMENT_ID_FAR_P90,
        "interpretation_level": 4,
        "claim_class": CLAIM_FINITE_AMPLITUDE_RESPONSE,
        "physical_path": None,
        "estimand": ESTIMAND_FAR_THRESHOLD_SUP,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_002_supremum_threshold_not_written",
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
        "written_by_resilience_family": False,
    },
    MEASUREMENT_ID_R50_DISCRETE: {
        "measurement_id": MEASUREMENT_ID_R50_DISCRETE,
        "interpretation_level": None,
        "claim_class": CLAIM_FINITE_AMPLITUDE_RESPONSE,
        "physical_path": f"{PHYSICAL_RESILIENCE_ROOT}/summary/r50_discrete_first_bin_at_or_below_half",
        "estimand": ESTIMAND_R50_DISCRETE,
        "relation": RELATION_DIAGNOSTICS,
        "identity_relation": "diagnostics_not_far_threshold_p50",
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
        "interpretation_level_token": "not_numbered",
        "written_by_resilience_family": True,
    },
    MEASUREMENT_ID_STATE_RETURN: {
        "measurement_id": MEASUREMENT_ID_STATE_RETURN,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": None,
        "estimand": ESTIMAND_STATE_RETURN,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_observational_return_is_not_attractor",
        "default_qualification_status": QUALIFICATION_NOT_ATTRACTOR,
        "not_attractor": True,
        "written_by_persistence_family": False,
    },
    MEASUREMENT_ID_LOCAL_RECURRENCE: {
        "measurement_id": MEASUREMENT_ID_LOCAL_RECURRENCE,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": None,
        "estimand": ESTIMAND_LOCAL_RECURRENCE,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_epsilon_recurrence_is_not_attractor",
        "default_qualification_status": QUALIFICATION_NOT_ATTRACTOR,
        "not_attractor": True,
        "written_by_persistence_family": False,
    },
    MEASUREMENT_ID_REGION_SURVIVAL: {
        "measurement_id": MEASUREMENT_ID_REGION_SURVIVAL,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL_HORIZON,
        "physical_path": None,
        "estimand": ESTIMAND_REGION_SURVIVAL,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_region_survival_is_not_attractor",
        "default_qualification_status": QUALIFICATION_NOT_ATTRACTOR,
        "not_attractor": True,
        "written_by_persistence_family": False,
    },
    MEASUREMENT_ID_REGION_DWELL: {
        "measurement_id": MEASUREMENT_ID_REGION_DWELL,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL_HORIZON,
        "physical_path": None,
        "estimand": ESTIMAND_REGION_DWELL,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_region_dwell_is_not_attractor",
        "default_qualification_status": QUALIFICATION_NOT_ATTRACTOR,
        "not_attractor": True,
        "written_by_persistence_family": False,
    },
    MEASUREMENT_ID_SELF_RETENTION: {
        "measurement_id": MEASUREMENT_ID_SELF_RETENTION,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": None,
        "estimand": ESTIMAND_SELF_RETENTION,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_until_qualified_one_step_retention_model",
        "default_qualification_status": QUALIFICATION_NOT_ATTRACTOR,
        "not_attractor": True,
        "written_by_persistence_family": False,
    },
    MEASUREMENT_ID_ESCAPE_RATE: {
        "measurement_id": MEASUREMENT_ID_ESCAPE_RATE,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": None,
        "estimand": ESTIMAND_TRANSFORMED_RETENTION,
        "relation": RELATION_WITHHELD,
        "identity_relation": "transformed_retention_not_exit_rate",
        "default_qualification_status": QUALIFICATION_NOT_ATTRACTOR,
        "not_attractor": True,
        "not_escape_rate": True,
        "written_by_persistence_family": False,
    },
    MEASUREMENT_ID_BASIN_ATTRACTOR: {
        "measurement_id": MEASUREMENT_ID_BASIN_ATTRACTOR,
        "interpretation_level": 4,
        "claim_class": CLAIM_FINITE_HORIZON_PROPAGATION,
        "physical_path": None,
        "estimand": ESTIMAND_BASIN_ATTRACTOR,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_until_validated_long_horizon_attractor",
        "default_qualification_status": QUALIFICATION_NOT_ATTRACTOR,
        "not_attractor": True,
        "written_by_persistence_family": False,
    },
    MEASUREMENT_ID_OBS_FUTURE_SPREAD: {
        "measurement_id": MEASUREMENT_ID_OBS_FUTURE_SPREAD,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": None,
        "estimand": ESTIMAND_EMPIRICAL_FUTURE_COV,
        "relation": RELATION_SUPERSEDED,
        "identity_relation": "superseded_state_matched_spread_is_level1",
        "default_qualification_status": QUALIFICATION_NOT_CONTROLLABILITY,
        "not_controllability": True,
        "not_occupancy": True,
        "not_empirical_written": True,
        "written_by_spread_family": False,
    },
    MEASUREMENT_ID_OBS_FUTURE_DEFF: {
        "measurement_id": MEASUREMENT_ID_OBS_FUTURE_DEFF,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": None,
        "relation": RELATION_SUPERSEDED,
        "identity_relation": "superseded_state_matched_spread_scalar_is_level1",
        "default_qualification_status": QUALIFICATION_NOT_CONTROLLABILITY,
        "not_controllability": True,
        "written_by_spread_family": False,
    },
    MEASUREMENT_ID_OBS_FUTURE_LOGVOL: {
        "measurement_id": MEASUREMENT_ID_OBS_FUTURE_LOGVOL,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": None,
        "relation": RELATION_SUPERSEDED,
        "identity_relation": "superseded_state_matched_spread_scalar_is_level1",
        "default_qualification_status": QUALIFICATION_NOT_CONTROLLABILITY,
        "not_controllability": True,
        "written_by_spread_family": False,
    },
    MEASUREMENT_ID_OBS_FUTURE_ANISO: {
        "measurement_id": MEASUREMENT_ID_OBS_FUTURE_ANISO,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": None,
        "relation": RELATION_SUPERSEDED,
        "identity_relation": "superseded_state_matched_spread_scalar_is_level1",
        "default_qualification_status": QUALIFICATION_NOT_CONTROLLABILITY,
        "not_controllability": True,
        "written_by_spread_family": False,
    },
    MEASUREMENT_ID_COND_FUTURE_SPREAD: {
        "measurement_id": MEASUREMENT_ID_COND_FUTURE_SPREAD,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL_HORIZON,
        "physical_path": None,
        "estimand": ESTIMAND_EMPIRICAL_FUTURE_COV,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_empirical_future_covariance_not_w_q",
        "default_qualification_status": QUALIFICATION_NOT_CONTROLLABILITY,
        "not_controllability": True,
        "not_occupancy": True,
        "written_by_spread_family": False,
    },
    MEASUREMENT_ID_TRANSITION_REACH_COV: {
        "measurement_id": MEASUREMENT_ID_TRANSITION_REACH_COV,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": PHYSICAL_WQ_W,
        "estimand": ESTIMAND_LYAPUNOV_W,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "identity_relation": "documented_identity_of_one_step_w_q_not_empirical_spread",
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
        "not_controllability": True,
        "not_occupancy": True,
        "not_empirical_future_covariance": True,
        "not_generator_spread": True,
        "written_by_local_dynamics_reachability": True,
        "ingest_estimator": "compute_stochastic_reachability",
    },
    MEASUREMENT_ID_TRANSITION_REACH_DEFF: {
        "measurement_id": MEASUREMENT_ID_TRANSITION_REACH_DEFF,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": None,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_one_step_d_eff_name_not_existing_d_eff",
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
        "not_controllability": True,
        "written_by_spread_family": False,
    },
    MEASUREMENT_ID_GENERATOR_SPREAD: {
        "measurement_id": MEASUREMENT_ID_GENERATOR_SPREAD,
        "interpretation_level": 3,
        "claim_class": CLAIM_GENERATOR,
        "physical_path": None,
        "estimand": ESTIMAND_GENERATOR_SPREAD,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_until_qualified_generator_spread",
        "default_qualification_status": QUALIFICATION_NOT_CONTROLLABILITY,
        "not_controllability": True,
        "not_generator_spread": True,
        "written_by_spread_family": False,
    },
    MEASUREMENT_ID_FINITE_TIME_REACH: {
        "measurement_id": MEASUREMENT_ID_FINITE_TIME_REACH,
        "interpretation_level": 4,
        "claim_class": CLAIM_FINITE_HORIZON_PROPAGATION,
        "physical_path": PHYSICAL_WQ_W,
        "estimand": ESTIMAND_LYAPUNOV_W,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "identity_relation": "documented_identity_of_existing_w_q_not_a_rename",
        "default_qualification_status": QUALIFICATION_NOT_ASSESSED,
        "not_controllability": True,
        "not_occupancy": True,
        "not_empirical_future_covariance": True,
        "not_generator_spread": True,
        "written_by_local_dynamics_reachability": True,
        "ingest_estimator": "compute_stochastic_reachability_from_gate_e",
    },
    MEASUREMENT_ID_RETURN_DISTANCE: {
        "measurement_id": MEASUREMENT_ID_RETURN_DISTANCE,
        "interpretation_level": 0,
        "claim_class": CLAIM_DESCRIPTIVE,
        "physical_path": None,
        "estimand": ESTIMAND_RETURN_DISTANCE,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_descriptive_return_distance_is_not_far",
        "default_qualification_status": QUALIFICATION_NOT_FAR,
        "not_far": True,
        "written_by_hysteresis_family": False,
    },
    MEASUREMENT_ID_MATCHED_RETURN_DISTANCE: {
        "measurement_id": MEASUREMENT_ID_MATCHED_RETURN_DISTANCE,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL_HORIZON,
        "physical_path": None,
        "estimand": ESTIMAND_MATCHED_RETURN,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_matched_return_is_observational_not_level2",
        "default_qualification_status": QUALIFICATION_NOT_FAR,
        "not_far": True,
        "written_by_hysteresis_family": False,
    },
    MEASUREMENT_ID_RECOVERY_TIME_L1: {
        "measurement_id": MEASUREMENT_ID_RECOVERY_TIME_L1,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL_HORIZON,
        "physical_path": None,
        "estimand": ESTIMAND_HYSTERESIS_RECOVERY_TIME,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_hysteresis_recovery_time_is_not_far",
        "default_qualification_status": QUALIFICATION_NOT_FAR,
        "not_far": True,
        "written_by_hysteresis_family": False,
    },
    MEASUREMENT_ID_PATH_ASYMMETRY_L2: {
        "measurement_id": MEASUREMENT_ID_PATH_ASYMMETRY_L2,
        "interpretation_level": 2,
        "claim_class": CLAIM_DISCRETE_ONE_STEP,
        "physical_path": None,
        "estimand": ESTIMAND_PATH_ASYMMETRY,
        "relation": RELATION_SUPERSEDED,
        "identity_relation": "superseded_matching_is_not_automatically_level2",
        "default_qualification_status": QUALIFICATION_NOT_FAR,
        "not_far": True,
        "written_by_hysteresis_family": False,
    },
    MEASUREMENT_ID_PATH_ASYMMETRY_L1: {
        "measurement_id": MEASUREMENT_ID_PATH_ASYMMETRY_L1,
        "interpretation_level": 1,
        "claim_class": CLAIM_CONDITIONAL_HORIZON,
        "physical_path": None,
        "estimand": ESTIMAND_PATH_ASYMMETRY,
        "relation": RELATION_WITHHELD,
        "identity_relation": "withheld_003_path_comparison_not_written",
        "default_qualification_status": QUALIFICATION_NOT_FAR,
        "not_far": True,
        "written_by_hysteresis_family": False,
    },
}

SUPPORT_OBJECT_LAG1 = "lag1_source_transitions"
SUPPORT_OBJECTS: dict[str, dict[str, Any]] = {
    SUPPORT_OBJECT_LAG1: {
        "support_object_id": SUPPORT_OBJECT_LAG1,
        "object_kind": "transition_support",
        "interpretation_level": None,
        "not_a_measurement_level": True,
        "relation": RELATION_SUPPORT_OBJECT,
        "physical_path": PHYSICAL_POOLED_SOURCE_IDX,
        "also_written_at": (PHYSICAL_DIFFUSION_SOURCE_IDX,),
        "variant_subset_paths": (PHYSICAL_CROSSFIT_SOURCE_IDX,),
        "distance_metric_id": DISTANCE_EUCLIDEAN_CHART,
        "embargo_semantics": "index_steps",
        "written_by_drift_family": True,
        "written_by_diffusion_family": True,
    },
}

PHYSICAL_POOLED_B_HAT = (
    f"{PHYSICAL_DRIFT_ROOT}/{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}/"
    f"{VARIANT_POOLED}/series/b_hat"
)
PHYSICAL_DIFFUSION_TENSOR_ALIAS = f"{PHYSICAL_DIFFUSION_ROOT}/series/diffusion_tensor"
PHYSICAL_D_TOTAL = f"{PHYSICAL_DIFFUSION_ROOT}/series/D_total"
PHYSICAL_A_BD = f"{PHYSICAL_DIFFUSION_ROOT}/series/A_bD"
PHYSICAL_R_B_OVER_A = f"{PHYSICAL_DIFFUSION_ROOT}/series/R_b_over_a"

COMPATIBILITY_ROWS: dict[str, dict[str, Any]] = {
    "/mnps_3d_dot": {
        "historical_or_export_name": "/mnps_3d_dot",
        "physical_path": PHYSICAL_MNPS_DOT,
        "estimand": ESTIMAND_SAVGOL,
        "measurement_id": MEASUREMENT_ID_SMOOTHED_VELOCITY,
        "relation": RELATION_EXISTING_DATASET,
        "notes": (
            "Savitzky-Golay derivative of /mnps_3d. Documented identity of "
            "smoothed_velocity_savgol_level0. Not an alias of realized_velocity_level0. "
            "Diaries that called this Level 0 velocity keep that historical wording."
        ),
    },
    MEASUREMENT_ID_SMOOTHED_VELOCITY: {
        "historical_or_export_name": MEASUREMENT_ID_SMOOTHED_VELOCITY,
        "physical_path": PHYSICAL_MNPS_DOT,
        "estimand": ESTIMAND_SAVGOL,
        "measurement_id": MEASUREMENT_ID_SMOOTHED_VELOCITY,
        "relation": RELATION_EXISTING_DATASET,
        "notes": "Register name for existing /mnps_3d_dot. Not written by the drift family.",
    },
    MEASUREMENT_ID_REALIZED_VELOCITY: {
        "historical_or_export_name": MEASUREMENT_ID_REALIZED_VELOCITY,
        "physical_path": f"{PHYSICAL_DRIFT_ROOT}/{MEASUREMENT_ID_REALIZED_VELOCITY}",
        "estimand": ESTIMAND_PER_STEP_DX_DT,
        "measurement_id": MEASUREMENT_ID_REALIZED_VELOCITY,
        "relation": RELATION_NEW_MEASURE,
        "notes": "Per-step (x_{t+1}-x_t)/observed_dt. New measure, not an alias of /mnps_3d_dot.",
    },
    f"{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}/{VARIANT_POOLED}": {
        "historical_or_export_name": f"{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}/{VARIANT_POOLED}",
        "physical_path": f"{PHYSICAL_DRIFT_ROOT}/{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}/{VARIANT_POOLED}",
        "estimand": ESTIMAND_MEAN_INCREMENT_NOMINAL_DT,
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
        "variant_id": VARIANT_POOLED,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "Weighted mean increment over nominal_dt. Not ito_drift_level3 and not "
            "an alias of diary-397 finite_lag."
        ),
    },
    "b_hat": {
        "historical_or_export_name": "b_hat",
        "physical_path": PHYSICAL_POOLED_B_HAT,
        "estimand": ESTIMAND_MEAN_INCREMENT_NOMINAL_DT,
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
        "variant_id": VARIANT_POOLED,
        "relation": RELATION_NEW_MEASURE,
        "notes": "Physical series of pooled conditional_mean_rate_level1. Series name is not renamed.",
    },
    f"{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}/{VARIANT_BLOCKED_CROSSFIT}": {
        "historical_or_export_name": (
            f"{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}/{VARIANT_BLOCKED_CROSSFIT}"
        ),
        "physical_path": (
            f"{PHYSICAL_DRIFT_ROOT}/{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}/"
            f"{VARIANT_BLOCKED_CROSSFIT}"
        ),
        "estimand": ESTIMAND_MEAN_INCREMENT_NOMINAL_DT,
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
        "variant_id": VARIANT_BLOCKED_CROSSFIT,
        "relation": RELATION_SAME_VARIANT,
        "notes": "Same measurement_id as pooled. Index-embargoed subset. Not a level upgrade.",
    },
    f"{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}/{VARIANT_LAG_DIAGNOSTICS}": {
        "historical_or_export_name": (
            f"{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}/{VARIANT_LAG_DIAGNOSTICS}"
        ),
        "physical_path": (
            f"{PHYSICAL_DRIFT_ROOT}/{MEASUREMENT_ID_CONDITIONAL_MEAN_RATE}/"
            f"{VARIANT_LAG_DIAGNOSTICS}"
        ),
        "estimand": ESTIMAND_LAG_DIAGNOSTIC,
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_MEAN_RATE,
        "variant_id": VARIANT_LAG_DIAGNOSTICS,
        "relation": RELATION_DIAGNOSTICS,
        "notes": "Multi-lag consistency. Not ito_drift_level3. lag_inconsistent does not erase pooled.",
    },
    "a_hat": {
        "historical_or_export_name": "a_hat",
        "physical_path": PHYSICAL_DIFFUSION_A_HAT,
        "estimand": ESTIMAND_CENTERED_INCREMENT_COV_NOMINAL_DT,
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": (
            "Centered np.cov(..., ddof=1)/nominal_dt. Documented identity of "
            "conditional_covariance_rate_level1, not a rename, not E[dX dX^T]/dt, "
            "not D=a/2."
        ),
    },
    MEASUREMENT_ID_CONDITIONAL_COVARIANCE: {
        "historical_or_export_name": MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
        "physical_path": PHYSICAL_DIFFUSION_A_HAT,
        "estimand": ESTIMAND_CENTERED_INCREMENT_COV_NOMINAL_DT,
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": "Semantic name of existing series/a_hat. No new HDF5 group.",
    },
    "diffusion_tensor": {
        "historical_or_export_name": "diffusion_tensor",
        "physical_path": PHYSICAL_DIFFUSION_TENSOR_ALIAS,
        "estimand": ESTIMAND_CENTERED_INCREMENT_COV_NOMINAL_DT,
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
        "relation": RELATION_SERIES_ALIAS,
        "notes": "v1 series alias of a_hat (same array). Not ito_diffusion_tensor_level3.",
    },
    "D_total": {
        "historical_or_export_name": "D_total",
        "physical_path": PHYSICAL_D_TOTAL,
        "estimand": ESTIMAND_TRACE_OF_A,
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
        "relation": RELATION_DERIVED_SCALAR,
        "notes": "Trace of a_hat. Not a separate diffusion tensor.",
    },
    "d_diff": {
        "historical_or_export_name": "d_diff",
        "physical_path": f"{PHYSICAL_DIFFUSION_ROOT}/series/d_diff",
        "estimand": ESTIMAND_ANISOTROPY_OF_A,
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
        "relation": RELATION_DERIVED_SCALAR,
        "notes": (
            "Effective dimension scalar of a_hat. Not "
            "diffusion_effective_dimension_level1."
        ),
    },
    "diffusion_effective_dimension": {
        "historical_or_export_name": "diffusion_effective_dimension",
        "physical_path": PHYSICAL_DIFFUSION_DEFF_ALIAS,
        "estimand": ESTIMAND_ANISOTROPY_OF_A,
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
        "relation": RELATION_SERIES_ALIAS,
        "notes": (
            "v1 series alias of d_diff (same array). Not "
            "diffusion_effective_dimension_level1."
        ),
    },
    "c_diff": {
        "historical_or_export_name": "c_diff",
        "physical_path": f"{PHYSICAL_DIFFUSION_ROOT}/series/c_diff",
        "estimand": ESTIMAND_ANISOTROPY_OF_A,
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
        "relation": RELATION_DERIVED_SCALAR,
        "notes": "Concentration scalar of a_hat. Not diffusion_condition_number_level1.",
    },
    MEASUREMENT_ID_DIFFUSION_DEFF: {
        "historical_or_export_name": MEASUREMENT_ID_DIFFUSION_DEFF,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_DIFFUSION_DEFF,
        "relation": RELATION_WITHHELD,
        "notes": "Not an alias of existing d_diff. d_diff is a derived scalar of a_hat.",
    },
    MEASUREMENT_ID_DIFFUSION_COND: {
        "historical_or_export_name": MEASUREMENT_ID_DIFFUSION_COND,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_DIFFUSION_COND,
        "relation": RELATION_WITHHELD,
        "notes": "Not an alias of existing c_diff. Not written.",
    },
    MEASUREMENT_ID_DIFFUSION_ENTROPY: {
        "historical_or_export_name": MEASUREMENT_ID_DIFFUSION_ENTROPY,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_DIFFUSION_ENTROPY,
        "relation": RELATION_WITHHELD,
        "notes": "001 anisotropy name. Not written.",
    },
    "A_bD": {
        "historical_or_export_name": "A_bD",
        "physical_path": PHYSICAL_A_BD,
        "estimand": ESTIMAND_ALIGNMENT_ABD,
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
        "relation": RELATION_ALIGNMENT,
        "notes": "Ingest C1 leaves this not_testable. Not a drift_source license.",
    },
    "R_b_over_a": {
        "historical_or_export_name": "R_b_over_a",
        "physical_path": PHYSICAL_R_B_OVER_A,
        "estimand": ESTIMAND_RATIO_R,
        "measurement_id": MEASUREMENT_ID_CONDITIONAL_COVARIANCE,
        "relation": RELATION_ALIGNMENT,
        "notes": "Ingest C1 leaves this not_testable. NaN is not zero alignment.",
    },
    SUPPORT_OBJECT_LAG1: {
        "historical_or_export_name": SUPPORT_OBJECT_LAG1,
        "physical_path": PHYSICAL_POOLED_SOURCE_IDX,
        "estimand": None,
        "measurement_id": None,
        "support_object_id": SUPPORT_OBJECT_LAG1,
        "relation": RELATION_SUPPORT_OBJECT,
        "notes": "Source-index identity, not a _levelN measure and not kNN neighborhoods.",
    },
    MEASUREMENT_ID_INCREMENT_COVARIANCE: {
        "historical_or_export_name": MEASUREMENT_ID_INCREMENT_COVARIANCE,
        "physical_path": f"{PHYSICAL_DIFFUSION_ROOT}/{MEASUREMENT_ID_INCREMENT_COVARIANCE}",
        "estimand": ESTIMAND_UNCONDITIONAL_INCREMENT_COV,
        "measurement_id": MEASUREMENT_ID_INCREMENT_COVARIANCE,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "Unconditional centered increment covariance, not divided by dt. "
            "Not diffusion a_hat and not C2 residualization."
        ),
    },
    MEASUREMENT_ID_INNOVATION_COVARIANCE: {
        "historical_or_export_name": MEASUREMENT_ID_INNOVATION_COVARIANCE,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_INNOVATION_COVARIANCE}",
        "estimand": ESTIMAND_INNOVATION_COV,
        "measurement_id": MEASUREMENT_ID_INNOVATION_COVARIANCE,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "Affine residual covariance of the qualified one-step map over "
            "nominal dt. Not diffusion a_hat and not C2 residualization."
        ),
    },
    MEASUREMENT_ID_AFFINE_MAP: {
        "historical_or_export_name": MEASUREMENT_ID_AFFINE_MAP,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_AFFINE_MAP}",
        "estimand": ESTIMAND_AFFINE_MAP,
        "measurement_id": MEASUREMENT_ID_AFFINE_MAP,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "Recording-level affine x_{t+dt} map with blocked holdout "
            "scoring. Variant declared_lag_steps=1 at this path. Not "
            "expm(J_hat dt), not a local kNN Jacobian, not the production "
            "Jacobian, and not the direct lag-2 map."
        ),
    },
    MEASUREMENT_ID_AFFINE_MEAN_RATE: {
        "historical_or_export_name": MEASUREMENT_ID_AFFINE_MEAN_RATE,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_AFFINE_MEAN_RATE}",
        "estimand": ESTIMAND_AFFINE_MEAN_RATE,
        "measurement_id": MEASUREMENT_ID_AFFINE_MEAN_RATE,
        "relation": RELATION_NEW_MEASURE,
        "notes": "One-step functional of Phi. Not conditional_mean_rate_level1.",
    },
    ONE_STEP_LAG2_MAP: {
        "historical_or_export_name": ONE_STEP_LAG2_MAP,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_AFFINE_MAP}",
        "estimand": ESTIMAND_AFFINE_LAG2_MAP,
        "measurement_id": MEASUREMENT_ID_AFFINE_MAP,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "Direct lag-2 affine map on pairs (x_t, x_{t+2 dt}). Same object "
            "class as lag 1. Not Phi_1 composed twice."
        ),
    },
    ONE_STEP_LAG2_MEAN_RATE: {
        "historical_or_export_name": ONE_STEP_LAG2_MEAN_RATE,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_AFFINE_MEAN_RATE}",
        "estimand": ESTIMAND_AFFINE_LAG2_MEAN_RATE,
        "measurement_id": MEASUREMENT_ID_AFFINE_MEAN_RATE,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "relation": RELATION_NEW_MEASURE,
        "notes": "Direct lag-2 functional of Phi_2. Not a composed one-step rollout.",
    },
    ONE_STEP_LAG2_INNOVATION: {
        "historical_or_export_name": ONE_STEP_LAG2_INNOVATION,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_INNOVATION_COVARIANCE}",
        "estimand": ESTIMAND_INNOVATION_LAG2,
        "measurement_id": MEASUREMENT_ID_INNOVATION_COVARIANCE,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "relation": RELATION_NEW_MEASURE,
        "notes": "Affine residual of the direct lag-2 map. Not diffusion a_hat.",
    },
    ONE_STEP_LAG2_SPECTRAL: {
        "historical_or_export_name": ONE_STEP_LAG2_SPECTRAL,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_SPECTRAL_ABSCISSA}",
        "estimand": ESTIMAND_SPECTRAL_ABSCISSA,
        "measurement_id": MEASUREMENT_ID_SPECTRAL_ABSCISSA,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "logm(Phi_2)/nominal_dt from a qualified direct lag-2 map. "
            "nominal_dt is the median lag-2 span. Not ito_drift_level3."
        ),
    },
    ONE_STEP_LAG2_NUMERICAL: {
        "historical_or_export_name": ONE_STEP_LAG2_NUMERICAL,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_NUMERICAL_ABSCISSA}",
        "estimand": ESTIMAND_NUMERICAL_ABSCISSA,
        "measurement_id": MEASUREMENT_ID_NUMERICAL_ABSCISSA,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "relation": RELATION_NEW_MEASURE,
        "notes": "Symmetric-part abscissa of logm(Phi_2)/nominal_dt. Not jacobian_metrics.",
    },
    ONE_STEP_LAG2_DIVERGENCE: {
        "historical_or_export_name": ONE_STEP_LAG2_DIVERGENCE,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_DIVERGENCE}",
        "estimand": ESTIMAND_DIVERGENCE,
        "measurement_id": MEASUREMENT_ID_DIVERGENCE,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "relation": RELATION_NEW_MEASURE,
        "notes": "trace(logm(Phi_2)/nominal_dt). Generator proxy from Phi_2, not Ito.",
    },
    ONE_STEP_LAG2_ROTATION: {
        "historical_or_export_name": ONE_STEP_LAG2_ROTATION,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_GENERATOR_ROTATION}",
        "estimand": ESTIMAND_GENERATOR_ROTATION,
        "measurement_id": MEASUREMENT_ID_GENERATOR_ROTATION,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "relation": RELATION_NEW_MEASURE,
        "notes": "Skew Frobenius of logm(Phi_2)/nominal_dt. Generator proxy, not Ito.",
    },
    ONE_STEP_LAG2_MAX_GAIN: {
        "historical_or_export_name": ONE_STEP_LAG2_MAX_GAIN,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_OPERATOR_MAX_GAIN}",
        "estimand": ESTIMAND_OPERATOR_MAX_GAIN,
        "measurement_id": MEASUREMENT_ID_OPERATOR_MAX_GAIN,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "relation": RELATION_NEW_MEASURE,
        "notes": "log sigma_max(Phi_2)/nominal_dt. Not spectral abscissa and not peak gain.",
    },
    ONE_STEP_LAG2_VOLUME_GAIN: {
        "historical_or_export_name": ONE_STEP_LAG2_VOLUME_GAIN,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_OPERATOR_VOLUME_GAIN}",
        "estimand": ESTIMAND_OPERATOR_VOLUME_GAIN,
        "measurement_id": MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "relation": RELATION_NEW_MEASURE,
        "notes": "log|det Phi_2|/nominal_dt. Rank-deficient maps fail closed. Not divergence_level3.",
    },
    ONE_STEP_LAG2_ROTATION_RATE: {
        "historical_or_export_name": ONE_STEP_LAG2_ROTATION_RATE,
        "physical_path": f"{PHYSICAL_ONE_STEP_LAG2_ROOT}/{MEASUREMENT_ID_OPERATOR_ROTATION_RATE}",
        "estimand": ESTIMAND_OPERATOR_ROTATION_RATE,
        "measurement_id": MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
        "variant_id": VARIANT_DECLARED_LAG_STEPS_2,
        "relation": RELATION_NEW_MEASURE,
        "notes": "Polar rotation of Phi_2. Not generator_rotation_norm_level3.",
    },
    MEASUREMENT_ID_NEIGHBOR_GAIN: {
        "historical_or_export_name": MEASUREMENT_ID_NEIGHBOR_GAIN,
        "physical_path": f"{PHYSICAL_AMPLIFICATION_ROOT}/{MEASUREMENT_ID_NEIGHBOR_GAIN}",
        "estimand": ESTIMAND_NEIGHBOR_GAIN_Q90,
        "measurement_id": MEASUREMENT_ID_NEIGHBOR_GAIN,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "Per-source q90 of same-pair (d1+eps)/(d0+eps); not a pooled "
            "pair-level Q90. Not operator_max_gain_rate_level2."
        ),
    },
    MEASUREMENT_ID_NEIGHBOR_SEPARATION: {
        "historical_or_export_name": MEASUREMENT_ID_NEIGHBOR_SEPARATION,
        "physical_path": f"{PHYSICAL_AMPLIFICATION_ROOT}/{MEASUREMENT_ID_NEIGHBOR_SEPARATION}",
        "estimand": ESTIMAND_NEIGHBOR_SEPARATION,
        "measurement_id": MEASUREMENT_ID_NEIGHBOR_SEPARATION,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "Per-source median of log((d1+eps)/(d0+eps))/nominal_dt on the "
            "same pairs as neighbor_gain_q90_level1. Not spectral_abscissa_level3."
        ),
    },
    MEASUREMENT_ID_NEIGHBOR_GAIN_RATE: {
        "historical_or_export_name": MEASUREMENT_ID_NEIGHBOR_GAIN_RATE,
        "physical_path": f"{PHYSICAL_AMPLIFICATION_ROOT}/{MEASUREMENT_ID_NEIGHBOR_GAIN_RATE}",
        "estimand": ESTIMAND_NEIGHBOR_GAIN_RATE,
        "measurement_id": MEASUREMENT_ID_NEIGHBOR_GAIN_RATE,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "log(neighbor_gain_q90)/nominal_dt of the already written q90. "
            "Not a new pair set. Not operator_max_gain_rate_level2."
        ),
    },
    MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN: {
        "historical_or_export_name": MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN,
        "physical_path": f"{PHYSICAL_HISTORY_ROOT}/{MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN}",
        "estimand": ESTIMAND_HISTORY_PREDICTIVE_GAIN,
        "measurement_id": MEASUREMENT_ID_HISTORY_PREDICTIVE_GAIN,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "OOS MSE(M0)-MSE(M1) on the same triples. Frozen affine M0/M1. "
            "Not Markov restoration. Not neighbor_gain_q90_level1."
        ),
    },
    MEASUREMENT_ID_HISTORY_OPERATOR: {
        "historical_or_export_name": MEASUREMENT_ID_HISTORY_OPERATOR,
        "physical_path": f"{PHYSICAL_HISTORY_ROOT}/{MEASUREMENT_ID_HISTORY_OPERATOR}",
        "estimand": ESTIMAND_HISTORY_OPERATOR,
        "measurement_id": MEASUREMENT_ID_HISTORY_OPERATOR,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "M1 affine (x_t, x_{t-1}) -> x_{t+1} identified by the frozen "
            "0.9 one-step OOS gate on the same triples. Not Markov "
            "restoration and not lag-1 Phi."
        ),
    },
    MEASUREMENT_ID_HISTORY_GENERATOR: {
        "historical_or_export_name": MEASUREMENT_ID_HISTORY_GENERATOR,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_HISTORY_GENERATOR,
        "relation": RELATION_WITHHELD,
        "notes": "Withheld. Not logm of the 3x6 M1 map and not ito_drift_level3.",
    },
    MEASUREMENT_ID_HISTORY_PROPAGATOR: {
        "historical_or_export_name": MEASUREMENT_ID_HISTORY_PROPAGATOR,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_HISTORY_PROPAGATOR,
        "relation": RELATION_WITHHELD,
        "notes": "Withheld. Not iteration of M1 and not iterated_one_step_horizon_map_level4.",
    },
    MEASUREMENT_ID_TURNING_RATE: {
        "historical_or_export_name": MEASUREMENT_ID_TURNING_RATE,
        "physical_path": f"{PHYSICAL_TURNING_ROOT}/{MEASUREMENT_ID_TURNING_RATE}",
        "estimand": ESTIMAND_TURNING_RATE,
        "measurement_id": MEASUREMENT_ID_TURNING_RATE,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "Angle between consecutive lag-1 displacements over observed dt. "
            "Undefined displacement is NaN, not zero. Not operator_rotation_rate_level2."
        ),
    },
    MEASUREMENT_ID_TURNING_ANGLE: {
        "historical_or_export_name": MEASUREMENT_ID_TURNING_ANGLE,
        "physical_path": f"{PHYSICAL_TURNING_ROOT}/{MEASUREMENT_ID_TURNING_ANGLE}",
        "estimand": ESTIMAND_TURNING_ANGLE,
        "measurement_id": MEASUREMENT_ID_TURNING_ANGLE,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "Angle between consecutive lag-1 displacements. Undefined is NaN, "
            "not zero. Not operator_rotation_rate_level2."
        ),
    },
    MEASUREMENT_ID_CLOUD_VOLUME: {
        "historical_or_export_name": MEASUREMENT_ID_CLOUD_VOLUME,
        "physical_path": f"{PHYSICAL_AMPLIFICATION_ROOT}/{MEASUREMENT_ID_CLOUD_VOLUME}",
        "estimand": ESTIMAND_CLOUD_VOLUME,
        "measurement_id": MEASUREMENT_ID_CLOUD_VOLUME,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "Same-pair cloud (logdet(C1+eps I)-logdet(C0+eps I))/(2 nominal_dt). "
            "Epsilon is a documented logdet floor, not operator-volume rescue."
        ),
    },
    "affine_two_step_map_level2": {
        "historical_or_export_name": "affine_two_step_map_level2",
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_SUPERSEDED,
        "notes": (
            "Name refused: two_step reads as composition. The level-2 object "
            "is affine_one_step_map_level2/declared_lag_2."
        ),
    },
    MEASUREMENT_ID_ITERATED_ONE_STEP_MAP: {
        "historical_or_export_name": MEASUREMENT_ID_ITERATED_ONE_STEP_MAP,
        "physical_path": PHYSICAL_ONE_STEP_HORIZON_MAP,
        "estimand": ESTIMAND_ITERATED_ONE_STEP_MAP,
        "measurement_id": MEASUREMENT_ID_ITERATED_ONE_STEP_MAP,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "Phi_1 applied twice with horizon holdout at 2 dt. Level 4 "
            "finite_horizon_propagation. Not the direct lag-2 map, not "
            "finite_time_peak_gain_level4, and not a level-2 write."
        ),
    },
    MEASUREMENT_ID_SPECTRAL_ABSCISSA: {
        "historical_or_export_name": MEASUREMENT_ID_SPECTRAL_ABSCISSA,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_SPECTRAL_ABSCISSA}",
        "estimand": ESTIMAND_SPECTRAL_ABSCISSA,
        "measurement_id": MEASUREMENT_ID_SPECTRAL_ABSCISSA,
        "relation": RELATION_NEW_MEASURE,
        "notes": "max Re eig(logm(Phi)/dt) after one-step identification. Not jacobian_metrics and not ito_drift_level3.",
    },
    MEASUREMENT_ID_NUMERICAL_ABSCISSA: {
        "historical_or_export_name": MEASUREMENT_ID_NUMERICAL_ABSCISSA,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_NUMERICAL_ABSCISSA}",
        "estimand": ESTIMAND_NUMERICAL_ABSCISSA,
        "measurement_id": MEASUREMENT_ID_NUMERICAL_ABSCISSA,
        "relation": RELATION_NEW_MEASURE,
        "notes": (
            "max eig of symmetric part of logm(Phi)/dt. Euclidean chart metric. "
            "Not jacobian_metrics. The difference omega-alpha is not "
            "reactivity_gap_level3."
        ),
    },
    MEASUREMENT_ID_REACTIVITY_GAP: {
        "historical_or_export_name": MEASUREMENT_ID_REACTIVITY_GAP,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_REACTIVITY_GAP,
        "relation": RELATION_WITHHELD,
        "notes": (
            "Withheld. Abscissa proxies exist; omega-alpha is not its own leaf. "
            "Not jacobian_metrics series/reactivity_gap."
        ),
    },
    "reactivity_gap": {
        "historical_or_export_name": "reactivity_gap",
        "physical_path": PHYSICAL_JACOBIAN_REACTIVITY_GAP,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_EXISTING_DATASET,
        "notes": (
            "Jacobian-metrics omega-alpha of J_hat. Not reactivity_gap_level3 "
            "and not a one-step generator-proxy leaf."
        ),
    },
    MEASUREMENT_ID_DIVERGENCE: {
        "historical_or_export_name": MEASUREMENT_ID_DIVERGENCE,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_DIVERGENCE}",
        "estimand": ESTIMAND_DIVERGENCE,
        "measurement_id": MEASUREMENT_ID_DIVERGENCE,
        "relation": RELATION_NEW_MEASURE,
        "notes": "trace(logm(Phi)/dt). Generator proxy, not Ito.",
    },
    MEASUREMENT_ID_GENERATOR_ROTATION: {
        "historical_or_export_name": MEASUREMENT_ID_GENERATOR_ROTATION,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_GENERATOR_ROTATION}",
        "estimand": ESTIMAND_GENERATOR_ROTATION,
        "measurement_id": MEASUREMENT_ID_GENERATOR_ROTATION,
        "relation": RELATION_NEW_MEASURE,
        "notes": "Frobenius norm of skew(logm(Phi)/dt). Generator proxy, not Ito.",
    },
    MEASUREMENT_ID_GENERATOR_SYM_ANISO: {
        "historical_or_export_name": MEASUREMENT_ID_GENERATOR_SYM_ANISO,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_GENERATOR_SYM_ANISO,
        "relation": RELATION_WITHHELD,
        "notes": "Not an alias of numerical_abscissa_level3. Not written.",
    },
    MEASUREMENT_ID_OPERATOR_MAX_GAIN: {
        "historical_or_export_name": MEASUREMENT_ID_OPERATOR_MAX_GAIN,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_OPERATOR_MAX_GAIN}",
        "estimand": ESTIMAND_OPERATOR_MAX_GAIN,
        "measurement_id": MEASUREMENT_ID_OPERATOR_MAX_GAIN,
        "relation": RELATION_NEW_MEASURE,
        "notes": "log sigma_max(Phi)/dt of a qualified one-step map. Not spectral_abscissa_level3 and not finite_time_peak_gain_level4.",
    },
    MEASUREMENT_ID_OPERATOR_GAIN_ANISO: {
        "historical_or_export_name": MEASUREMENT_ID_OPERATOR_GAIN_ANISO,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_OPERATOR_GAIN_ANISO,
        "relation": RELATION_WITHHELD,
        "notes": "Not an alias of operator_max_gain_rate_level2. Not written.",
    },
    MEASUREMENT_ID_OPERATOR_VOLUME_GAIN: {
        "historical_or_export_name": MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_OPERATOR_VOLUME_GAIN}",
        "estimand": ESTIMAND_OPERATOR_VOLUME_GAIN,
        "measurement_id": MEASUREMENT_ID_OPERATOR_VOLUME_GAIN,
        "relation": RELATION_NEW_MEASURE,
        "notes": "log|det Phi|/dt. Rank-deficient Phi is insufficient_support, not epsilon-rescued. Not divergence_level3.",
    },
    MEASUREMENT_ID_OPERATOR_ROTATION_RATE: {
        "historical_or_export_name": MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
        "physical_path": f"{PHYSICAL_ONE_STEP_ROOT}/{MEASUREMENT_ID_OPERATOR_ROTATION_RATE}",
        "estimand": ESTIMAND_OPERATOR_ROTATION_RATE,
        "measurement_id": MEASUREMENT_ID_OPERATOR_ROTATION_RATE,
        "relation": RELATION_NEW_MEASURE,
        "notes": "||log R||_F/(sqrt(2) dt) from polar Phi=R P. Reflection or non-real log fails closed. Not generator_rotation_norm_level3.",
    },
    MEASUREMENT_ID_ITO_DRIFT: {
        "historical_or_export_name": MEASUREMENT_ID_ITO_DRIFT,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_ITO_DRIFT,
        "relation": RELATION_WITHHELD,
        "notes": "Not written. lag_diagnostics is not this identity. No auto-promotion.",
    },
    MEASUREMENT_ID_ITO_DIFFUSION: {
        "historical_or_export_name": MEASUREMENT_ID_ITO_DIFFUSION,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_ITO_DIFFUSION,
        "relation": RELATION_WITHHELD,
        "notes": "Not written. a_hat is not this identity. No auto-promotion.",
    },
    "ito_qualified": {
        "historical_or_export_name": "ito_qualified",
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_WITHHELD,
        "notes": "Status token is never written. Not a measurement_id and not auto-promotion.",
    },
    MEASUREMENT_ID_PEAK_GAIN: {
        "historical_or_export_name": MEASUREMENT_ID_PEAK_GAIN,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_PEAK_GAIN,
        "relation": RELATION_WITHHELD,
        "notes": (
            "Withheld. Not Phi powers, not operator_max_gain_rate_level2, "
            "and not iterated_one_step_horizon_map_level4."
        ),
    },
    f"local_first_hit_outcome_average/q_A_to_B": {
        "historical_or_export_name": "local_first_hit_outcome_average/q_A_to_B",
        "physical_path": PHYSICAL_DESTINATION_Q,
        "estimand": ESTIMAND_RESOLVED_FIRST_HIT,
        "measurement_id": MEASUREMENT_ID_DESTINATION_RESOLVED,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": (
            "Existing first-hit series/q_A_to_B. State-conditioned mean of resolved "
            "A/B hits. Not destination_first_hit_fraction_level0, not "
            "destination_hit_probability_level1 (including unresolved), and not "
            "generator_committor_level3. Series name is not renamed."
        ),
    },
    MEASUREMENT_ID_DESTINATION_RESOLVED: {
        "historical_or_export_name": MEASUREMENT_ID_DESTINATION_RESOLVED,
        "physical_path": PHYSICAL_DESTINATION_Q,
        "estimand": ESTIMAND_RESOLVED_FIRST_HIT,
        "measurement_id": MEASUREMENT_ID_DESTINATION_RESOLVED,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": "Register name for first-hit q_A_to_B. Not a new HDF5 group.",
    },
    "local_law_dense_grid_o2b/q_A_to_B": {
        "historical_or_export_name": "local_law_dense_grid_o2b/q_A_to_B",
        "physical_path": PHYSICAL_DESTINATION_Q,
        "estimand": ESTIMAND_O2B_QUADRATURE,
        "measurement_id": MEASUREMENT_ID_O2B_QUADRATURE,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": (
            "Ingest destination series/q_A_to_B under O2b. Restricted 1-D "
            "quadrature on an explicit reaction coordinate. Not "
            "generator_committor_level3 and not a 3D/9D MNPS committor."
        ),
    },
    MEASUREMENT_ID_O2B_QUADRATURE: {
        "historical_or_export_name": MEASUREMENT_ID_O2B_QUADRATURE,
        "physical_path": PHYSICAL_DESTINATION_Q,
        "estimand": ESTIMAND_O2B_QUADRATURE,
        "measurement_id": MEASUREMENT_ID_O2B_QUADRATURE,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": "Register name for ingest O2b q_A_to_B. Not a rename of the series.",
    },
    "q_hat": {
        "historical_or_export_name": "q_hat",
        "physical_path": f"{PHYSICAL_DESTINATION_ROOT}/series/q_hat",
        "estimand": ESTIMAND_O2B_QUADRATURE,
        "measurement_id": MEASUREMENT_ID_O2B_QUADRATURE,
        "relation": RELATION_SERIES_ALIAS,
        "notes": "O2b series alias of q_A_to_B. Same array.",
    },
    MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0: {
        "historical_or_export_name": MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_DESTINATION_FIRST_HIT_L0,
        "relation": RELATION_WITHHELD,
        "notes": "003: x-conditioned first-hit is level1, not level0. Not written. Not an alias of destination_first_hit_fraction_resolved_level1.",
    },
    MEASUREMENT_ID_DESTINATION_HIT_L1: {
        "historical_or_export_name": MEASUREMENT_ID_DESTINATION_HIT_L1,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_DESTINATION_HIT_L1,
        "relation": RELATION_WITHHELD,
        "notes": (
            "Unresolved-included hit probability is not the current first-hit "
            "estimand. Not an alias of destination_first_hit_fraction_resolved_level1."
        ),
    },
    MEASUREMENT_ID_DESTINATION_UNRESOLVED: {
        "historical_or_export_name": MEASUREMENT_ID_DESTINATION_UNRESOLVED,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_DESTINATION_UNRESOLVED,
        "relation": RELATION_WITHHELD,
        "notes": (
            "Unresolved fraction is not serialized as its own series. "
            "series/resolved_first_hit_outcome NaNs are an encoding, not this leaf."
        ),
    },
    "resolved_first_hit_outcome": {
        "historical_or_export_name": "resolved_first_hit_outcome",
        "physical_path": PHYSICAL_DESTINATION_OUTCOME,
        "estimand": ESTIMAND_RESOLVED_FIRST_HIT,
        "measurement_id": None,
        "relation": RELATION_EXISTING_DATASET,
        "notes": (
            "Per-row 0/1/NaN first-hit encoding under destination. Not "
            "destination_unresolved_fraction_level1 and not "
            "destination_hit_probability_level1."
        ),
    },
    "n_resolved_first_hit_outcomes": {
        "historical_or_export_name": "n_resolved_first_hit_outcomes",
        "physical_path": PHYSICAL_DESTINATION_N_RESOLVED,
        "estimand": ESTIMAND_RESOLVED_FIRST_HIT,
        "measurement_id": None,
        "relation": RELATION_DIAGNOSTICS,
        "notes": (
            "Count of finite A/B first-hit outcomes. Not "
            "destination_unresolved_fraction_level1."
        ),
    },
    MEASUREMENT_ID_GENERATOR_COMMITTOR: {
        "historical_or_export_name": MEASUREMENT_ID_GENERATOR_COMMITTOR,
        "physical_path": None,
        "estimand": ESTIMAND_GENERATOR_COMMITTOR,
        "measurement_id": MEASUREMENT_ID_GENERATOR_COMMITTOR,
        "relation": RELATION_WITHHELD,
        "notes": "Not written. O2b q and first-hit q are not this identity.",
    },
    MEASUREMENT_ID_TRANSITION_HIT_L2: {
        "historical_or_export_name": MEASUREMENT_ID_TRANSITION_HIT_L2,
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_SUPERSEDED,
        "notes": (
            "002 name. 003 moves multi-step model first-hit to level4. "
            "No YAML alias. Not written."
        ),
    },
    MEASUREMENT_ID_TRANSITION_HIT_L4: {
        "historical_or_export_name": MEASUREMENT_ID_TRANSITION_HIT_L4,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_TRANSITION_HIT_L4,
        "relation": RELATION_WITHHELD,
        "notes": (
            "Withheld until a validated multi-step transition model exists. "
            "Composing a qualified one-step map is not this identity. YAML nested "
            "name is refused."
        ),
    },
    "finite_lag": {
        "historical_or_export_name": "finite_lag",
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_SUPERSEDED,
        "notes": (
            "Diary 397 unreleased nested name. YAML/H5 refused. Not an alias of "
            "conditional_mean_rate_level1/pooled. Historical diaries keep original meaning."
        ),
    },
    "crossfit": {
        "historical_or_export_name": "crossfit",
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_SUPERSEDED,
        "notes": (
            "Diary 397 unreleased nested name. YAML/H5 refused. Not an alias of "
            "blocked_crossfit. Historical diaries keep original meaning. "
            "The C1 source token crossfit_local_chart_b is a separate withheld name."
        ),
    },
    "crossfit_local_chart_b": {
        "historical_or_export_name": "crossfit_local_chart_b",
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_WITHHELD,
        "notes": (
            "C1 SOURCE_CROSSFIT token. Closed before M3. Nested YAML key is "
            "refused. Using it as drift.source stays not_testable "
            "(crossfit_not_authorized_before_m3). Not an independent b for A_bD."
        ),
    },
    "ito_candidate": {
        "historical_or_export_name": "ito_candidate",
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_SUPERSEDED,
        "notes": (
            "Diary 397 unreleased nested name. YAML/H5 refused. Not an alias of "
            "lag_diagnostics or ito_drift_level3."
        ),
    },
    LOGICAL_LOCAL_DYNAMICS_ROOT: {
        "historical_or_export_name": LOGICAL_LOCAL_DYNAMICS_ROOT,
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_LOGICAL,
        "notes": (
            "SL-LEV-MES-003 §9 proposed logical tree. Not a writable HDF5 root. "
            "Physical paths stay under /dynamical_families and /mnps_3d_dot."
        ),
    },
    LOGICAL_POSSIBLE_FUTURES_ROOT: {
        "historical_or_export_name": LOGICAL_POSSIBLE_FUTURES_ROOT,
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_LOGICAL,
        "notes": (
            "002/003 proposed logical tree for destination, persistence, and "
            "spread. Not a writable HDF5 root. Physical destination stays "
            "under /dynamical_families/destination/v1."
        ),
    },
    LOGICAL_POSSIBLE_FUTURES_DESTINATION: {
        "historical_or_export_name": LOGICAL_POSSIBLE_FUTURES_DESTINATION,
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_LOGICAL,
        "notes": "Logical 002 path. Not written. Destination family root is unchanged.",
    },
    "basin_return_probability": {
        "historical_or_export_name": "basin_return_probability",
        "physical_path": PHYSICAL_RESILIENCE_CURVE,
        "estimand": ESTIMAND_FAR_RETURN_FRACTION,
        "measurement_id": MEASUREMENT_ID_FAR_RECOVERY,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": (
            "Existing amplitude_curve field. Documented identity of "
            "far_recovery_probability_level4. Not spontaneous return. "
            "Requires an explicit perturbation protocol."
        ),
    },
    "amplitude_curve": {
        "historical_or_export_name": "amplitude_curve",
        "physical_path": PHYSICAL_RESILIENCE_CURVE,
        "estimand": ESTIMAND_FAR_RETURN_FRACTION,
        "measurement_id": MEASUREMENT_ID_FAR_RECOVERY,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": "Physical FAR curve. Series/group name is not renamed.",
    },
    MEASUREMENT_ID_FAR_RECOVERY: {
        "historical_or_export_name": MEASUREMENT_ID_FAR_RECOVERY,
        "physical_path": PHYSICAL_RESILIENCE_CURVE,
        "estimand": ESTIMAND_FAR_RETURN_FRACTION,
        "measurement_id": MEASUREMENT_ID_FAR_RECOVERY,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": "Register name for existing protocol-gated return fractions.",
    },
    "finite_amplitude_resilience_level4": {
        "historical_or_export_name": "finite_amplitude_resilience_level4",
        "physical_path": PHYSICAL_RESILIENCE_CURVE,
        "estimand": ESTIMAND_FAR_RETURN_FRACTION,
        "measurement_id": MEASUREMENT_ID_FAR_RECOVERY,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": "002 family heading. Same identity as far_recovery_probability_level4.",
    },
    "return_fraction": {
        "historical_or_export_name": "return_fraction",
        "physical_path": PHYSICAL_RESILIENCE_CURVE,
        "estimand": ESTIMAND_FAR_RETURN_FRACTION,
        "measurement_id": MEASUREMENT_ID_FAR_RECOVERY,
        "relation": RELATION_SERIES_ALIAS,
        "notes": "Same numeric field as basin_return_probability in each amplitude row.",
    },
    "r50_discrete_first_bin_at_or_below_half": {
        "historical_or_export_name": "r50_discrete_first_bin_at_or_below_half",
        "physical_path": f"{PHYSICAL_RESILIENCE_ROOT}/summary/r50_discrete_first_bin_at_or_below_half",
        "estimand": ESTIMAND_R50_DISCRETE,
        "measurement_id": MEASUREMENT_ID_R50_DISCRETE,
        "relation": RELATION_DIAGNOSTICS,
        "notes": "Not far_threshold_p50_level4. Not the 002 supremum {a: R(a,H) >= p}.",
    },
    MEASUREMENT_ID_R50_DISCRETE: {
        "historical_or_export_name": MEASUREMENT_ID_R50_DISCRETE,
        "physical_path": f"{PHYSICAL_RESILIENCE_ROOT}/summary/r50_discrete_first_bin_at_or_below_half",
        "estimand": ESTIMAND_R50_DISCRETE,
        "measurement_id": MEASUREMENT_ID_R50_DISCRETE,
        "relation": RELATION_DIAGNOSTICS,
        "notes": "Register name for existing r50 discrete diagnostic.",
    },
    MEASUREMENT_ID_SPONTANEOUS_RETURN: {
        "historical_or_export_name": MEASUREMENT_ID_SPONTANEOUS_RETURN,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_SPONTANEOUS_RETURN,
        "relation": RELATION_WITHHELD,
        "notes": "Observational return after a natural excursion. Not FAR. Not amplitude_curve. Not written.",
    },
    MEASUREMENT_ID_EXCURSION_RECOVERY: {
        "historical_or_export_name": MEASUREMENT_ID_EXCURSION_RECOVERY,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_EXCURSION_RECOVERY,
        "relation": RELATION_WITHHELD,
        "notes": "Finite-amplitude observational curve. Still not FAR. Not written.",
    },
    MEASUREMENT_ID_EXCURSION_RECOVERY_TIME: {
        "historical_or_export_name": MEASUREMENT_ID_EXCURSION_RECOVERY_TIME,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_EXCURSION_RECOVERY_TIME,
        "relation": RELATION_WITHHELD,
        "notes": "Observational recovery-time median. Not FAR. Not written.",
    },
    MEASUREMENT_ID_MATCHED_RECOVERY_L2: {
        "historical_or_export_name": MEASUREMENT_ID_MATCHED_RECOVERY_L2,
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_SUPERSEDED,
        "notes": "002 name. 003: matching is not automatically level2. No YAML alias.",
    },
    MEASUREMENT_ID_MATCHED_DISPLACEMENT_L2: {
        "historical_or_export_name": MEASUREMENT_ID_MATCHED_DISPLACEMENT_L2,
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_SUPERSEDED,
        "notes": "002 name. 003: matching is not automatically level2. No YAML alias.",
    },
    MEASUREMENT_ID_MATCHED_RECOVERY_L1: {
        "historical_or_export_name": MEASUREMENT_ID_MATCHED_RECOVERY_L1,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_MATCHED_RECOVERY_L1,
        "relation": RELATION_WITHHELD,
        "notes": "003 preferred observational matched comparison. Not written. Not FAR.",
    },
    MEASUREMENT_ID_FAR_P50: {
        "historical_or_export_name": MEASUREMENT_ID_FAR_P50,
        "physical_path": None,
        "estimand": ESTIMAND_FAR_THRESHOLD_SUP,
        "measurement_id": MEASUREMENT_ID_FAR_P50,
        "relation": RELATION_WITHHELD,
        "notes": "002 supremum threshold. Existing r50 is a different estimand.",
    },
    MEASUREMENT_ID_FAR_P90: {
        "historical_or_export_name": MEASUREMENT_ID_FAR_P90,
        "physical_path": None,
        "estimand": ESTIMAND_FAR_THRESHOLD_SUP,
        "measurement_id": MEASUREMENT_ID_FAR_P90,
        "relation": RELATION_WITHHELD,
        "notes": "002 supremum threshold. Not written.",
    },
    LOGICAL_PERTURBATION_ROOT: {
        "historical_or_export_name": LOGICAL_PERTURBATION_ROOT,
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_LOGICAL,
        "notes": "003 sole logical home for perturbation measures. Not a writable HDF5 root.",
    },
    LOGICAL_POSSIBLE_FUTURES_PERTURBATION: {
        "historical_or_export_name": LOGICAL_POSSIBLE_FUTURES_PERTURBATION,
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_LOGICAL,
        "notes": "002 alternate path. 003 prefers /perturbation/. Not written.",
    },
    MEASUREMENT_ID_STATE_RETURN: {
        "historical_or_export_name": MEASUREMENT_ID_STATE_RETURN,
        "physical_path": None,
        "estimand": ESTIMAND_STATE_RETURN,
        "measurement_id": MEASUREMENT_ID_STATE_RETURN,
        "relation": RELATION_WITHHELD,
        "notes": "Observational quantized return. Not an attractor. Not written.",
    },
    "state_recurrence_level0": {
        "historical_or_export_name": "state_recurrence_level0",
        "physical_path": None,
        "estimand": ESTIMAND_STATE_RETURN,
        "measurement_id": MEASUREMENT_ID_STATE_RETURN,
        "relation": RELATION_WITHHELD,
        "notes": "002 heading. Same withheld identity as state_return_probability_level0.",
    },
    MEASUREMENT_ID_LOCAL_RECURRENCE: {
        "historical_or_export_name": MEASUREMENT_ID_LOCAL_RECURRENCE,
        "physical_path": None,
        "estimand": ESTIMAND_LOCAL_RECURRENCE,
        "measurement_id": MEASUREMENT_ID_LOCAL_RECURRENCE,
        "relation": RELATION_WITHHELD,
        "notes": "Epsilon-ball recurrence rate. Not an attractor. Not written.",
    },
    MEASUREMENT_ID_REGION_SURVIVAL: {
        "historical_or_export_name": MEASUREMENT_ID_REGION_SURVIVAL,
        "physical_path": None,
        "estimand": ESTIMAND_REGION_SURVIVAL,
        "measurement_id": MEASUREMENT_ID_REGION_SURVIVAL,
        "relation": RELATION_WITHHELD,
        "notes": "Stay-in-R for horizon H. Persistence, not an attractor. Not written.",
    },
    "residence_persistence_level1": {
        "historical_or_export_name": "residence_persistence_level1",
        "physical_path": None,
        "estimand": ESTIMAND_REGION_SURVIVAL,
        "measurement_id": MEASUREMENT_ID_REGION_SURVIVAL,
        "relation": RELATION_WITHHELD,
        "notes": "002 heading. Same withheld identity as region_survival_probability_level1.",
    },
    MEASUREMENT_ID_REGION_DWELL: {
        "historical_or_export_name": MEASUREMENT_ID_REGION_DWELL,
        "physical_path": None,
        "estimand": ESTIMAND_REGION_DWELL,
        "measurement_id": MEASUREMENT_ID_REGION_DWELL,
        "relation": RELATION_WITHHELD,
        "notes": "First-exit time from a frozen region. Not an attractor. Not written.",
    },
    MEASUREMENT_ID_SELF_RETENTION: {
        "historical_or_export_name": MEASUREMENT_ID_SELF_RETENTION,
        "physical_path": None,
        "estimand": ESTIMAND_SELF_RETENTION,
        "measurement_id": MEASUREMENT_ID_SELF_RETENTION,
        "relation": RELATION_WITHHELD,
        "notes": "One-step P(R|R). Withheld until a qualified discrete model exists.",
    },
    "transition_metastability_level2": {
        "historical_or_export_name": "transition_metastability_level2",
        "physical_path": None,
        "estimand": ESTIMAND_SELF_RETENTION,
        "measurement_id": MEASUREMENT_ID_SELF_RETENTION,
        "relation": RELATION_WITHHELD,
        "notes": "002 heading. Same withheld identity as transition_self_retention_level2.",
    },
    MEASUREMENT_ID_ESCAPE_RATE: {
        "historical_or_export_name": MEASUREMENT_ID_ESCAPE_RATE,
        "physical_path": None,
        "estimand": ESTIMAND_TRANSFORMED_RETENTION,
        "measurement_id": MEASUREMENT_ID_ESCAPE_RATE,
        "relation": RELATION_WITHHELD,
        "notes": (
            "003: -log(P_RR)/dt is transformed retention, not an exit rate. "
            "Being back in R at the next sample is not staying in R the whole interval."
        ),
    },
    MEASUREMENT_ID_BASIN_ATTRACTOR: {
        "historical_or_export_name": MEASUREMENT_ID_BASIN_ATTRACTOR,
        "physical_path": None,
        "estimand": ESTIMAND_BASIN_ATTRACTOR,
        "measurement_id": MEASUREMENT_ID_BASIN_ATTRACTOR,
        "relation": RELATION_WITHHELD,
        "notes": "Attractor/basin name reserved for validated long-horizon structure. Not written.",
    },
    LOGICAL_STATE_GEOMETRY_ROOT: {
        "historical_or_export_name": LOGICAL_STATE_GEOMETRY_ROOT,
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_LOGICAL,
        "notes": "003 proposed tree for occupancy and chart geometry. Not a writable HDF5 root.",
    },
    LOGICAL_POSSIBLE_FUTURES_PERSISTENCE: {
        "historical_or_export_name": LOGICAL_POSSIBLE_FUTURES_PERSISTENCE,
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_LOGICAL,
        "notes": "003 logical home for persistence. Not written.",
    },
    LOGICAL_POSSIBLE_FUTURES_RECURRENCE: {
        "historical_or_export_name": LOGICAL_POSSIBLE_FUTURES_RECURRENCE,
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_LOGICAL,
        "notes": "002/003 logical home for recurrence. Not written. Recurrence is not an attractor.",
    },
    MEASUREMENT_ID_OBS_FUTURE_SPREAD: {
        "historical_or_export_name": MEASUREMENT_ID_OBS_FUTURE_SPREAD,
        "physical_path": None,
        "estimand": ESTIMAND_EMPIRICAL_FUTURE_COV,
        "measurement_id": MEASUREMENT_ID_COND_FUTURE_SPREAD,
        "relation": RELATION_SUPERSEDED,
        "notes": "003: state-matched future covariance is level1. No YAML alias. Not written.",
    },
    MEASUREMENT_ID_OBS_FUTURE_DEFF: {
        "historical_or_export_name": MEASUREMENT_ID_OBS_FUTURE_DEFF,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_COND_FUTURE_SPREAD,
        "relation": RELATION_SUPERSEDED,
        "notes": "Not existing d_eff of W_Q. Empirical future-dimension name is unwritten.",
    },
    MEASUREMENT_ID_OBS_FUTURE_LOGVOL: {
        "historical_or_export_name": MEASUREMENT_ID_OBS_FUTURE_LOGVOL,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_COND_FUTURE_SPREAD,
        "relation": RELATION_SUPERSEDED,
        "notes": "Not existing v_norm of W_Q. Unwritten.",
    },
    MEASUREMENT_ID_OBS_FUTURE_ANISO: {
        "historical_or_export_name": MEASUREMENT_ID_OBS_FUTURE_ANISO,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_COND_FUTURE_SPREAD,
        "relation": RELATION_SUPERSEDED,
        "notes": "Not existing c_1_q of W_Q. Unwritten.",
    },
    MEASUREMENT_ID_COND_FUTURE_SPREAD: {
        "historical_or_export_name": MEASUREMENT_ID_COND_FUTURE_SPREAD,
        "physical_path": None,
        "estimand": ESTIMAND_EMPIRICAL_FUTURE_COV,
        "measurement_id": MEASUREMENT_ID_COND_FUTURE_SPREAD,
        "relation": RELATION_WITHHELD,
        "notes": "Empirical x-conditioned future covariance. Not W_Q and not d_eff. Not written.",
    },
    "transition_predictive_spread_level2": {
        "historical_or_export_name": "transition_predictive_spread_level2",
        "physical_path": PHYSICAL_WQ_W,
        "estimand": ESTIMAND_LYAPUNOV_W,
        "measurement_id": MEASUREMENT_ID_TRANSITION_REACH_COV,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": "002 heading. One-step W_Q only. Multi-step composition is level4.",
    },
    MEASUREMENT_ID_TRANSITION_REACH_COV: {
        "historical_or_export_name": MEASUREMENT_ID_TRANSITION_REACH_COV,
        "physical_path": PHYSICAL_WQ_W,
        "estimand": ESTIMAND_LYAPUNOV_W,
        "measurement_id": MEASUREMENT_ID_TRANSITION_REACH_COV,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": "Existing w_q when n_propagator_steps==1. Predictive spread, not controllability.",
    },
    MEASUREMENT_ID_TRANSITION_REACH_DEFF: {
        "historical_or_export_name": MEASUREMENT_ID_TRANSITION_REACH_DEFF,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_TRANSITION_REACH_DEFF,
        "relation": RELATION_WITHHELD,
        "notes": "Not an alias of existing d_eff. d_eff is a derived scalar of W_Q.",
    },
    MEASUREMENT_ID_GENERATOR_SPREAD: {
        "historical_or_export_name": MEASUREMENT_ID_GENERATOR_SPREAD,
        "physical_path": None,
        "estimand": ESTIMAND_GENERATOR_SPREAD,
        "measurement_id": MEASUREMENT_ID_GENERATOR_SPREAD,
        "relation": RELATION_WITHHELD,
        "notes": "W_Q is not generator spread. Not written.",
    },
    "generator_predictive_spread_level3": {
        "historical_or_export_name": "generator_predictive_spread_level3",
        "physical_path": None,
        "estimand": ESTIMAND_GENERATOR_SPREAD,
        "measurement_id": MEASUREMENT_ID_GENERATOR_SPREAD,
        "relation": RELATION_WITHHELD,
        "notes": "002 heading. Same withheld identity.",
    },
    MEASUREMENT_ID_FINITE_TIME_REACH: {
        "historical_or_export_name": MEASUREMENT_ID_FINITE_TIME_REACH,
        "physical_path": PHYSICAL_WQ_W,
        "estimand": ESTIMAND_LYAPUNOV_W,
        "measurement_id": MEASUREMENT_ID_FINITE_TIME_REACH,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": "Existing w_q when n_propagator_steps>1, and ingest refusals. Not a rename.",
    },
    PHYSICAL_WQ_ROOT: {
        "historical_or_export_name": PHYSICAL_WQ_ROOT,
        "physical_path": PHYSICAL_WQ_ROOT,
        "estimand": ESTIMAND_LYAPUNOV_W,
        "measurement_id": MEASUREMENT_ID_FINITE_TIME_REACH,
        "relation": RELATION_EXISTING_DATASET,
        "notes": "Opt-in Gate F W_Q. Not /dynamical_families/spread. Predictive spread, not controllability.",
    },
    "w_q": {
        "historical_or_export_name": "w_q",
        "physical_path": PHYSICAL_WQ_W,
        "estimand": ESTIMAND_LYAPUNOV_W,
        "measurement_id": MEASUREMENT_ID_FINITE_TIME_REACH,
        "relation": RELATION_DOCUMENTED_IDENTITY,
        "notes": "Physical Lyapunov matrix. One-step computed objects use the level2 identity.",
    },
    "d_eff": {
        "historical_or_export_name": "d_eff",
        "physical_path": f"{PHYSICAL_WQ_PRIMARY}/d_eff",
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_FINITE_TIME_REACH,
        "relation": RELATION_DERIVED_SCALAR,
        "notes": (
            "Derived scalar of W_Q. Not observed_future_effective_dimension_level0 "
            "and not reachability_effective_dimension_level4."
        ),
    },
    "c_1_q": {
        "historical_or_export_name": "c_1_q",
        "physical_path": f"{PHYSICAL_WQ_PRIMARY}/c_1_q",
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_FINITE_TIME_REACH,
        "relation": RELATION_DERIVED_SCALAR,
        "notes": "Derived scalar of W_Q. Not reachability_anisotropy_level4.",
    },
    MEASUREMENT_ID_REACH_DEFF_L4: {
        "historical_or_export_name": MEASUREMENT_ID_REACH_DEFF_L4,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_REACH_DEFF_L4,
        "relation": RELATION_WITHHELD,
        "notes": "Not an alias of existing d_eff. d_eff is a derived scalar of W_Q.",
    },
    MEASUREMENT_ID_REACH_ANISO_L4: {
        "historical_or_export_name": MEASUREMENT_ID_REACH_ANISO_L4,
        "physical_path": None,
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_REACH_ANISO_L4,
        "relation": RELATION_WITHHELD,
        "notes": "Not an alias of existing c_1_q. Not written.",
    },
    "v_norm": {
        "historical_or_export_name": "v_norm",
        "physical_path": f"{PHYSICAL_WQ_PRIMARY}/v_norm",
        "estimand": None,
        "measurement_id": MEASUREMENT_ID_FINITE_TIME_REACH,
        "relation": RELATION_DERIVED_SCALAR,
        "notes": "Derived scalar of W_Q. Not observed_future_log_volume_level0.",
    },
    LOGICAL_POSSIBLE_FUTURES_SPREAD: {
        "historical_or_export_name": LOGICAL_POSSIBLE_FUTURES_SPREAD,
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_LOGICAL,
        "notes": "003 logical home for predictive spread. Physical W_Q stays /stochastic_reachability/v1.",
    },
    MEASUREMENT_ID_RETURN_DISTANCE: {
        "historical_or_export_name": MEASUREMENT_ID_RETURN_DISTANCE,
        "physical_path": None,
        "estimand": ESTIMAND_RETURN_DISTANCE,
        "measurement_id": MEASUREMENT_ID_RETURN_DISTANCE,
        "relation": RELATION_WITHHELD,
        "notes": "Descriptive d(x_post, x_baseline). Not FAR. Not written.",
    },
    MEASUREMENT_ID_MATCHED_RETURN_DISTANCE: {
        "historical_or_export_name": MEASUREMENT_ID_MATCHED_RETURN_DISTANCE,
        "physical_path": None,
        "estimand": ESTIMAND_MATCHED_RETURN,
        "measurement_id": MEASUREMENT_ID_MATCHED_RETURN_DISTANCE,
        "relation": RELATION_WITHHELD,
        "notes": "Baseline/state-matched observational comparison. Not level2. Not FAR. Not written.",
    },
    MEASUREMENT_ID_RECOVERY_TIME_L1: {
        "historical_or_export_name": MEASUREMENT_ID_RECOVERY_TIME_L1,
        "physical_path": None,
        "estimand": ESTIMAND_HYSTERESIS_RECOVERY_TIME,
        "measurement_id": MEASUREMENT_ID_RECOVERY_TIME_L1,
        "relation": RELATION_WITHHELD,
        "notes": "Observational hysteresis return time. Not excursion_recovery_time_median_level1. Not FAR. Not written.",
    },
    MEASUREMENT_ID_PATH_ASYMMETRY_L2: {
        "historical_or_export_name": MEASUREMENT_ID_PATH_ASYMMETRY_L2,
        "physical_path": None,
        "estimand": ESTIMAND_PATH_ASYMMETRY,
        "measurement_id": None,
        "relation": RELATION_SUPERSEDED,
        "notes": "002 name. 003: matching is not automatically level2. No YAML alias.",
    },
    MEASUREMENT_ID_PATH_ASYMMETRY_L1: {
        "historical_or_export_name": MEASUREMENT_ID_PATH_ASYMMETRY_L1,
        "physical_path": None,
        "estimand": ESTIMAND_PATH_ASYMMETRY,
        "measurement_id": MEASUREMENT_ID_PATH_ASYMMETRY_L1,
        "relation": RELATION_WITHHELD,
        "notes": "003 preferred observational path comparison. Not a discrete model. Not written.",
    },
    LOGICAL_POSSIBLE_FUTURES_HYSTERESIS: {
        "historical_or_export_name": LOGICAL_POSSIBLE_FUTURES_HYSTERESIS,
        "physical_path": None,
        "estimand": None,
        "measurement_id": None,
        "relation": RELATION_LOGICAL,
        "notes": "003 observational recovery without a perturbation protocol. /perturbation/ remains FAR-only. Not written.",
    },
}


def support_object_entry(support_object_id: str) -> dict[str, Any]:
    """Return a copy of one support-object row. These are not measurement levels."""
    try:
        return dict(SUPPORT_OBJECTS[support_object_id])
    except KeyError as exc:
        raise KeyError(f"Unknown support_object_id: {support_object_id}") from exc


def register_entry(measurement_id: str) -> dict[str, Any]:
    """Return a copy of one register row."""
    try:
        return dict(MEASUREMENT_REGISTER[measurement_id])
    except KeyError as exc:
        raise KeyError(f"Unknown measurement_id: {measurement_id}") from exc


def compatibility_entry(historical_or_export_name: str) -> dict[str, Any]:
    """Return one compatibility-table row. These do not rename physical paths."""
    try:
        return dict(COMPATIBILITY_ROWS[historical_or_export_name])
    except KeyError as exc:
        raise KeyError(
            f"Unknown compatibility name: {historical_or_export_name}"
        ) from exc


def relation_for(historical_or_export_name: str) -> str:
    """Return the closed relation token for one mapped name."""
    relation = compatibility_entry(historical_or_export_name)["relation"]
    if relation not in COMPATIBILITY_RELATIONS:
        raise ValueError(f"relation {relation!r} is not in COMPATIBILITY_RELATIONS")
    return str(relation)


def support_status_for(computation_status: str, *, failure_reason: str | None) -> str:
    """Map computation outcome to a separate support_status token."""
    status = str(computation_status or "").strip()
    reason = str(failure_reason or "").strip()
    if status == "computed":
        return SUPPORT_SUFFICIENT
    if status == "insufficient_support" or "insufficient" in reason:
        return SUPPORT_INSUFFICIENT
    return SUPPORT_NOT_APPLICABLE


def qualification_status_for(
    *,
    interpretation_level: int | None,
    computed: bool,
    default: str | None = None,
) -> str:
    """Return a qualification token. Destination uses an explicit default."""
    if default:
        return str(default)
    if interpretation_level is None:
        return QUALIFICATION_NOT_ASSESSED
    if int(interpretation_level) >= 3:
        return QUALIFICATION_ITO_NOT_QUALIFIED
    if int(interpretation_level) == 1:
        return QUALIFICATION_ITO_NOT_QUALIFIED
    if computed:
        return QUALIFICATION_NOT_ASSESSED
    return QUALIFICATION_NOT_ASSESSED


def stamp_register_fields(
    result: Mapping[str, Any],
    measurement_id: str,
    *,
    variant_id: str | None = None,
) -> dict[str, Any]:
    """Copy register identity onto a family result without renaming physical roots."""
    entry = register_entry(measurement_id)
    out = dict(result)
    status = str(out.get("computation_status") or "").strip()
    failure_reason = out.get("failure_reason")
    computed = status == "computed"
    raw_level = entry.get("interpretation_level")
    interpretation_level = None if raw_level is None else int(raw_level)
    out["measurement_id"] = entry["measurement_id"]
    out["interpretation_level"] = interpretation_level
    out["variant_id"] = str(variant_id) if variant_id else None
    out["claim_class"] = entry["claim_class"]
    out["support_status"] = support_status_for(status, failure_reason=str(failure_reason) if failure_reason else None)
    out["qualification_status"] = qualification_status_for(
        interpretation_level=interpretation_level,
        computed=computed,
        default=entry.get("default_qualification_status"),
    )
    summary = dict(out.get("summary") or {})
    summary["measurement_id"] = out["measurement_id"]
    summary["interpretation_level"] = interpretation_level
    summary["variant_id"] = out["variant_id"]
    summary["claim_class"] = out["claim_class"]
    summary["support_status"] = out["support_status"]
    summary["qualification_status"] = out["qualification_status"]
    if "not_sde_drift" in entry:
        summary["not_sde_drift"] = entry["not_sde_drift"]
    if entry.get("estimand"):
        summary["estimand"] = entry["estimand"]
    for extra_key in (
        "identity_relation",
        "relation",
        "diffusion_convention",
        "not_ito_diffusion_tensor",
        "not_generator_committor",
        "not_a_3d_mnps_committor",
        "interpretation_level_token",
        "not_far",
        "not_inferred_from_spontaneous_trajectory",
        "not_attractor",
        "not_escape_rate",
        "not_controllability",
        "not_occupancy",
        "not_empirical_future_covariance",
        "not_generator_spread",
        "not_composed_one_step",
        "not_direct_lag2_map",
        "not_independent_singular_oos",
        "not_spectral_abscissa",
        "not_peak_gain_level4",
        "not_generator_rotation",
        "not_operator_rotation",
        "undefined_not_zero",
        "not_operator_max_gain",
        "not_operator_volume",
        "epsilon_logdet_regularized",
        "not_divided_by_dt",
        "not_local_knn",
        "same_pair_forward",
        "not_resampled_at_target",
        "not_markov_restoration",
        "not_one_step_operator",
        "history_m1_not_lag1_phi",
        "not_logm_of_m1",
        "not_m1_iteration",
        "declared_lag_steps",
        "horizon_steps",
        "embargo_semantics",
        "min_embargo_steps",
    ):
        if extra_key in entry:
            summary[extra_key] = entry[extra_key]
    out["summary"] = summary
    provenance = dict(out.get("provenance") or {})
    settings = dict(provenance.get("settings") or {})
    settings["measurement_id"] = out["measurement_id"]
    settings["interpretation_level"] = interpretation_level
    settings["variant_id"] = out["variant_id"]
    settings["claim_class"] = out["claim_class"]
    settings["physical_path"] = entry.get("physical_path")
    settings["distance_metric_id"] = entry.get("distance_metric_id")
    if "not_sde_drift" in entry:
        settings["not_sde_drift"] = entry["not_sde_drift"]
    if entry.get("estimand"):
        settings["estimand"] = entry["estimand"]
    for extra_key in (
        "identity_relation",
        "relation",
        "diffusion_convention",
        "not_ito_diffusion_tensor",
        "not_generator_committor",
        "not_a_3d_mnps_committor",
        "interpretation_level_token",
        "ingest_estimator",
        "not_far",
        "not_inferred_from_spontaneous_trajectory",
        "not_attractor",
        "not_escape_rate",
        "not_controllability",
        "not_occupancy",
        "not_empirical_future_covariance",
        "not_generator_spread",
        "not_composed_one_step",
        "not_direct_lag2_map",
        "not_independent_singular_oos",
        "not_spectral_abscissa",
        "not_peak_gain_level4",
        "not_generator_rotation",
        "not_operator_rotation",
        "undefined_not_zero",
        "not_operator_max_gain",
        "not_operator_volume",
        "epsilon_logdet_regularized",
        "not_divided_by_dt",
        "not_local_knn",
        "same_pair_forward",
        "not_resampled_at_target",
        "not_markov_restoration",
        "not_one_step_operator",
        "history_m1_not_lag1_phi",
        "not_logm_of_m1",
        "not_m1_iteration",
        "declared_lag_steps",
        "horizon_steps",
        "embargo_semantics",
        "min_embargo_steps",
    ):
        if extra_key in entry:
            settings[extra_key] = entry[extra_key]
    if variant_id == VARIANT_LAG_DIAGNOSTICS:
        settings["estimand"] = ESTIMAND_LAG_DIAGNOSTIC
        summary["estimand"] = ESTIMAND_LAG_DIAGNOSTIC
    provenance["settings"] = settings
    out["provenance"] = provenance
    return out


def stamp_reachability_identity(result: Mapping[str, Any]) -> dict[str, Any]:
    """Stamp existing W_Q. One-step is level2; composed steps and refusals are level4."""
    status = str(result.get("computation_status") or "").strip()
    n_steps = result.get("n_propagator_steps")
    if status == "computed" and int(n_steps or 0) == 1:
        return stamp_register_fields(result, MEASUREMENT_ID_TRANSITION_REACH_COV)
    return stamp_register_fields(result, MEASUREMENT_ID_FINITE_TIME_REACH)
