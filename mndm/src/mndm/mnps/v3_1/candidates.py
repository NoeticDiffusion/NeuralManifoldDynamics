"""Versioned registry and E1 estimator for MNPS v3.1.

EEG sensor-space prototypes are implemented in :mod:`eeg_metrics`. E1 is a
separate fixed-protocol candidate. Executability does not ratify any estimator
or assign a coordinate; every result remains exploratory and unqualified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class MNPS31Candidate:
    """Registry entry for one proposed candidate family."""

    candidate_id: str
    family: str
    description: str
    implementation_status: Literal[
        "unresolved", "prototype_available", "core_probe_available"
    ]
    construct_name: str
    short_name: str
    public_id: str
    standard_availability: Literal["standard_when_supported", "experimental_only"]
    coordinate_assignment: None = None
    qualified: bool = False
    estimator_ids: tuple[str, ...] = ()
    historical_aliases: tuple[str, ...] = ()


MNPS31_CANDIDATES: tuple[MNPS31Candidate, ...] = (
    MNPS31Candidate(
        "M1",
        "M",
        "Collective phase coordination variability (sensor-space sidecar)",
        "prototype_available",
        "Phase Coordination Variability",
        "PCV",
        "phase_coordination_variability",
        "standard_when_supported",
        estimator_ids=(
            "M1.sensor_phase_order_parameter_sd.v0",
            "PCV.pipeline_native_window_alpha_phase_order_parameter_sd.v1",
        ),
    ),
    MNPS31Candidate(
        "M2",
        "M",
        "Temporal persistence or mobility",
        "prototype_available",
        "Temporal Persistence",
        "TP",
        "temporal_persistence",
        "standard_when_supported",
        estimator_ids=(
            "M2.sensor_lag1_decorrelation.v0",
            "M2.surface_channel_median_lag1_decorrelation.v0",
            "TP.pipeline_native_window_lag1_decorrelation.v1",
        ),
    ),
    MNPS31Candidate(
        "M3",
        "M",
        "Proposed regional reconfiguration family (current sensor-only activity/connectivity prototypes)",
        "prototype_available",
        "Activity Reconfiguration",
        "AR",
        "activity_reconfiguration",
        "standard_when_supported",
        estimator_ids=(
            "M3.sensor_activity_adjacent_pearson_change.v0",
            "M3.sensor_connectivity_adjacent_pearson_change.v0",
            "AR.pipeline_native_window_adjacent_block_rms_change.v1",
        ),
    ),
    MNPS31Candidate(
        "D1",
        "D",
        "Graph integration or segregation (sensor graph sidecar)",
        "prototype_available",
        "Network Integration Efficiency",
        "NIE",
        "network_integration_efficiency",
        "standard_when_supported",
        estimator_ids=(
            "D1.sensor_weighted_global_efficiency_invweight.v0",
            "NIE.pipeline_native_window_abs_pearson_weighted_efficiency.v1",
        ),
    ),
    MNPS31Candidate(
        "D2",
        "D",
        "Spatial gradient of alpha-band mean debiased squared wPLI over sensor distance; historical local/distributed balance retained as a sidecar",
        "prototype_available",
        "Spatial Coupling Gradient",
        "SCG",
        "spatial_coupling_gradient",
        "experimental_only",
        estimator_ids=(
            "spatial_dwpli_distance_spearman_alpha.v1",
            "dwpli2_debiased_per_frequency_alpha_8_13hz_nonoverlapping_2sec.v1",
            "local_distributed_coupling_contrast_distance_quartiles_dwpli2_alpha.v1",
            "D2.sensor_near_far_coupling_difference.v0",
            "D2.sensor_within_between_coupling_difference.v0",
            "D2.sensor_label_family_local_distributed_coupling_difference.v0",
        ),
        historical_aliases=("D2", "LDCB"),
    ),
    MNPS31Candidate(
        "D3",
        "D",
        "Representational dispersion (sensor-pattern sidecar)",
        "prototype_available",
        "Representational Dispersion",
        "RD",
        "representational_dispersion",
        "standard_when_supported",
        estimator_ids=(
            "D3.sensor_rdm_mean_euclidean.v0",
            "RD.pipeline_native_window_centered_log_psd_dispersion.v1",
        ),
    ),
    MNPS31Candidate(
        "E1",
        "E",
        "Fixed-set channel-median normalized ordinal permutation entropy",
        "prototype_available",
        "Ordinal Irregularity",
        "OI",
        "ordinal_irregularity",
        "standard_when_supported",
        estimator_ids=(
            "mndm.mnps_v3_1.e1_channel_median_pe.v0",
            "OI.pipeline_native_window_permutation_entropy_order5_delay1.v1",
        ),
    ),
    MNPS31Candidate(
        "E2",
        "E",
        "Temporal-order compressibility relative to exact-multiset shuffled references (unqualified fixed-set technical candidate)",
        "core_probe_available",
        "Sequence Complexity",
        "SC",
        "sequence_complexity",
        "standard_when_supported",
        estimator_ids=(
            "E2.sensor_median_lz78_exact_multiset_ratio_83.v1",
            "E2.sensor_lz78_phrase_density_unadjusted.v0",
            "E2.sensor_lz78_permutation_reference_ratio.v0",
            "SC.pipeline_native_window_lz78_exact_multiset_ratio.v1",
        ),
    ),
    MNPS31Candidate(
        "E3",
        "E",
        "Proposed effective regional dimension family (current sensor covariance prototype is sensor-only)",
        "prototype_available",
        "Effective Dimensionality",
        "ED",
        "effective_dimensionality",
        "standard_when_supported",
        estimator_ids=(
            "E3.sensor_covariance_participation_ratio.v0",
            "ED.pipeline_native_window_covariance_participation_ratio.v1",
        ),
    ),
)

MNPS31_CANDIDATES_BY_ID: dict[str, MNPS31Candidate] = {
    candidate.candidate_id: candidate for candidate in MNPS31_CANDIDATES
}

MNPS31_STANDARD_CANDIDATE_IDS: tuple[str, ...] = tuple(
    candidate.candidate_id
    for candidate in MNPS31_CANDIDATES
    if candidate.standard_availability == "standard_when_supported"
)


def construct_metadata(candidate_id: str) -> dict[str, object]:
    """Return public construct vocabulary while retaining the legacy M/D/E slot."""
    token = str(candidate_id)
    candidate = MNPS31_CANDIDATES_BY_ID.get(token)
    if candidate is None:
        return {
            "construct_name": f"Unregistered candidate {token}",
            "short_name": token,
            "public_id": token.casefold(),
            "historical_slot": token,
            "historical_family": token[:1] if token else None,
            "historical_aliases": [token] if token else [],
            "standard_availability": "experimental_only",
            "coordinate_assignment": None,
        }
    return {
        "construct_name": candidate.construct_name,
        "short_name": candidate.short_name,
        "public_id": candidate.public_id,
        "historical_slot": candidate.candidate_id,
        "historical_family": candidate.family,
        "historical_aliases": list(candidate.historical_aliases),
        "standard_availability": candidate.standard_availability,
        "coordinate_assignment": None,
    }

E1_CANDIDATE_ID = "e_irregularity_channel_median_pe_v0"
E1_METHOD_VERSION = "mndm.mnps_v3_1.e1_channel_median_pe.v0"
E1_CHANNEL_COUNT = 83
E1_BANDPASS_HZ = (1.0, 45.0)
E1_SAMPLE_RATE_HZ = 250.0
E1_WINDOW_SECONDS = 20.0
E1_SAMPLE_COUNT = int(E1_SAMPLE_RATE_HZ * E1_WINDOW_SECONDS)
E1_ORDER = 5
E1_DELAY_SAMPLES = 1


@dataclass(frozen=True)
class MNPS31WindowSupport:
    """Audited support required before a v3.1 metric computes a window value.

    All support flags must be explicitly true. ``None`` means that support
    was not audited and therefore cannot be treated as acceptable.
    Times describe the actual half-open sample interval ``[start, end)``.
    """

    window_start_sec: float
    window_end_sec: float
    sample_rate_hz: float
    sample_count: int
    segment_id: str
    source_is_continuous: bool | None = None
    timestamps_uniform: bool | None = None
    gap_free: bool | None = None
    bad_interval_free: bool | None = None
    within_recording_bounds: bool | None = None
    effective_filter_support_valid: bool | None = None


# Backward-compatible name from the first E1-only implementation tranche.
MNPS31E1WindowSupport = MNPS31WindowSupport


@dataclass(frozen=True)
class MNPS31E1Measurement:
    """One attempted E1 value and its independent status fields."""

    candidate_id: str
    method_version: str
    value: float | None
    units: str
    computed: bool
    valid: bool
    qualified: bool
    status: Literal["computed_exploratory", "unsupported"]
    reason: str | None
    channel_count: int
    sample_count_per_channel: int
    channel_order_sha256: str
    support: MNPS31WindowSupport

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record; scientific qualification stays false."""
        payload = asdict(self)
        payload["support"] = asdict(self.support)
        payload["claim_status"] = "no_biological_claim"
        return payload


def _labels_hash(labels: tuple[str, ...]) -> str:
    canonical = "\n".join(label.strip().casefold() for label in labels)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _unsupported(
    *,
    support: MNPS31WindowSupport,
    reason: str,
    channel_count: int = 0,
    sample_count: int = 0,
    channel_order_sha256: str = "",
) -> MNPS31E1Measurement:
    return MNPS31E1Measurement(
        candidate_id=E1_CANDIDATE_ID,
        method_version=E1_METHOD_VERSION,
        value=None,
        units="1",
        computed=False,
        valid=False,
        qualified=False,
        status="unsupported",
        reason=reason,
        channel_count=channel_count,
        sample_count_per_channel=sample_count,
        channel_order_sha256=channel_order_sha256,
        support=support,
    )


def compute_e1_channel_median_permutation_entropy(
    signals: np.ndarray,
    channel_names: list[str] | tuple[str, ...],
    fixed_channel_names: list[str] | tuple[str, ...],
    support: MNPS31WindowSupport,
) -> MNPS31E1Measurement:
    """Compute the exploratory ANPHY E1 candidate for one audited window.

    The accepted protocol is fixed at 83 named channels, 20 seconds sampled
    at 250 Hz, normalized order-5/delay-1 permutation entropy, and the median
    over channels. Input rows may be in any order; they are aligned to the
    declared fixed channel order. Missing channels, non-finite or constant
    channels, and unsupported timebase/support return an explicit unsupported
    result with no numeric substitute.

    A finite result is marked computed and valid for this implementation's
    protocol checks, while ``qualified`` remains false by construction. Full
    source/config/runtime provenance is owned by the calling pipeline and is
    required before a candidate export is considered reviewable.
    """
    try:
        values = np.asarray(signals, dtype=np.float64)
    except (TypeError, ValueError):
        return _unsupported(support=support, reason="invalid_signal_array")

    channel_labels = tuple(str(name).strip() for name in channel_names)
    fixed_labels = tuple(str(name).strip() for name in fixed_channel_names)
    input_keys = tuple(name.casefold() for name in channel_labels)
    fixed_keys = tuple(name.casefold() for name in fixed_labels)
    fixed_hash = _labels_hash(fixed_labels)

    if values.ndim != 2:
        return _unsupported(
            support=support,
            reason="signal_must_be_channels_by_samples",
            channel_count=len(channel_labels),
            channel_order_sha256=fixed_hash,
        )
    if len(set(input_keys)) != len(input_keys) or any(not key for key in input_keys):
        return _unsupported(
            support=support,
            reason="duplicate_or_empty_input_channel_label",
            channel_count=len(channel_labels),
            sample_count=int(values.shape[1]),
            channel_order_sha256=fixed_hash,
        )
    if (
        len(fixed_labels) != E1_CHANNEL_COUNT
        or len(set(fixed_keys)) != E1_CHANNEL_COUNT
        or any(not key for key in fixed_keys)
    ):
        return _unsupported(
            support=support,
            reason="fixed_channel_set_must_contain_83_unique_labels",
            channel_count=len(channel_labels),
            sample_count=int(values.shape[1]),
            channel_order_sha256=fixed_hash,
        )
    if len(input_keys) != E1_CHANNEL_COUNT or set(input_keys) != set(fixed_keys):
        return _unsupported(
            support=support,
            reason="fixed_channel_set_mismatch",
            channel_count=len(channel_labels),
            sample_count=int(values.shape[1]),
            channel_order_sha256=fixed_hash,
        )

    audit_fields = (
        support.source_is_continuous,
        support.timestamps_uniform,
        support.gap_free,
        support.bad_interval_free,
        support.within_recording_bounds,
        support.effective_filter_support_valid,
    )
    if any(field is None for field in audit_fields):
        return _unsupported(
            support=support,
            reason="support_audit_incomplete",
            channel_count=len(channel_labels),
            sample_count=int(values.shape[1]),
            channel_order_sha256=fixed_hash,
        )
    if any(not isinstance(field, (bool, np.bool_)) for field in audit_fields):
        return _unsupported(
            support=support,
            reason="support_audit_flag_not_boolean",
            channel_count=len(channel_labels),
            sample_count=int(values.shape[1]),
            channel_order_sha256=fixed_hash,
        )
    failed_support_reasons = (
        (support.source_is_continuous, "non_continuous_source"),
        (support.timestamps_uniform, "irregular_sample_times"),
        (support.gap_free, "gap_or_segment_boundary"),
        (support.bad_interval_free, "bad_interval_overlap"),
        (support.within_recording_bounds, "recording_boundary_overrun"),
        (support.effective_filter_support_valid, "insufficient_filter_edge_support"),
    )
    for passed, reason in failed_support_reasons:
        if not bool(passed):
            return _unsupported(
                support=support,
                reason=reason,
                channel_count=len(channel_labels),
                sample_count=int(values.shape[1]),
                channel_order_sha256=fixed_hash,
            )

    if support.segment_id.strip() == "":
        return _unsupported(
            support=support,
            reason="missing_segment_id",
            channel_count=len(channel_labels),
            sample_count=int(values.shape[1]),
            channel_order_sha256=fixed_hash,
        )
    if not np.isfinite(support.sample_rate_hz) or not np.isclose(
        support.sample_rate_hz, E1_SAMPLE_RATE_HZ, rtol=0.0, atol=1e-9
    ):
        return _unsupported(
            support=support,
            reason="unsupported_sampling_rate",
            channel_count=len(channel_labels),
            sample_count=int(values.shape[1]),
            channel_order_sha256=fixed_hash,
        )
    if values.shape != (E1_CHANNEL_COUNT, E1_SAMPLE_COUNT) or support.sample_count != E1_SAMPLE_COUNT:
        return _unsupported(
            support=support,
            reason="unsupported_window_sample_count",
            channel_count=len(channel_labels),
            sample_count=int(values.shape[1]),
            channel_order_sha256=fixed_hash,
        )
    duration = support.window_end_sec - support.window_start_sec
    if (
        not np.isfinite(support.window_start_sec)
        or not np.isfinite(support.window_end_sec)
        or not np.isclose(duration, E1_WINDOW_SECONDS, rtol=0.0, atol=1e-6)
    ):
        return _unsupported(
            support=support,
            reason="window_time_support_mismatch",
            channel_count=len(channel_labels),
            sample_count=int(values.shape[1]),
            channel_order_sha256=fixed_hash,
        )
    if not np.isfinite(values).all():
        return _unsupported(
            support=support,
            reason="non_finite_sample",
            channel_count=len(channel_labels),
            sample_count=int(values.shape[1]),
            channel_order_sha256=fixed_hash,
        )

    lookup = {key: index for index, key in enumerate(input_keys)}
    aligned = values[[lookup[key] for key in fixed_keys], :]
    if np.any(np.ptp(aligned, axis=1) == 0.0):
        return _unsupported(
            support=support,
            reason="zero_variance_channel",
            channel_count=len(channel_labels),
            sample_count=int(values.shape[1]),
            channel_order_sha256=fixed_hash,
        )

    try:
        # E1 needs only this NumPy helper, not the full EEG feature module.
        from ...features.eeg_entropy import compute_permutation_entropy as _compute_permutation_entropy

        per_channel = np.asarray(
            [
                _compute_permutation_entropy(
                    row,
                    order=E1_ORDER,
                    delay=E1_DELAY_SAMPLES,
                    normalize=True,
                )
                for row in aligned
            ],
            dtype=np.float64,
        )
    except Exception:
        return _unsupported(
            support=support,
            reason="permutation_entropy_failed",
            channel_count=len(channel_labels),
            sample_count=int(values.shape[1]),
            channel_order_sha256=fixed_hash,
        )
    if per_channel.shape != (E1_CHANNEL_COUNT,) or not np.isfinite(per_channel).all():
        return _unsupported(
            support=support,
            reason="permutation_entropy_unsupported",
            channel_count=len(channel_labels),
            sample_count=int(values.shape[1]),
            channel_order_sha256=fixed_hash,
        )

    return MNPS31E1Measurement(
        candidate_id=E1_CANDIDATE_ID,
        method_version=E1_METHOD_VERSION,
        value=float(np.median(per_channel)),
        units="1",
        computed=True,
        valid=True,
        qualified=False,
        status="computed_exploratory",
        reason=None,
        channel_count=E1_CHANNEL_COUNT,
        sample_count_per_channel=E1_SAMPLE_COUNT,
        channel_order_sha256=fixed_hash,
        support=support,
    )
