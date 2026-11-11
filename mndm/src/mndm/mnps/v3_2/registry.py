"""Registry for the MNPS v3.2 candidate measurement collection.

The registry names estimator families and experimental realizations without
assigning them to the canonical MNPS chart or claiming scientific qualification.
The full-vector experiment preserves scalar, vector, and multiscale outputs as
separate measurement records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ImplementationStatus = Literal["kernel_available", "new_experimental_estimator", "optional_existing_estimator"]


@dataclass(frozen=True)
class MeasurementFamily:
    family_id: str
    name: str
    primary_estimator_id: str
    implementation_status: ImplementationStatus
    experiment_role: str
    primary_output_shape: str
    nominal_support_seconds: float | None
    qualified: bool = False
    coordinate_assignment: None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FULL_VECTOR_CHALLENGE_001_REGISTRY: tuple[MeasurementFamily, ...] = (
    MeasurementFamily("PCV", "Phase coordination variability",
                      "PCV.pipeline_native_window_alpha_phase_order_parameter_sd.v1",
                      "kernel_available", "primary_candidate", "scalar", 20.0),
    MeasurementFamily("TP", "Temporal persistence",
                      "TP.alpha_log_rms_envelope_pearson_persistence.v1",
                      "new_experimental_estimator", "primary_candidate", "scalar", 20.0),
    MeasurementFamily("AR", "Activity reconfiguration",
                      "AR.sensor_activity_adjacent_pearson_change_vector.v1",
                      "kernel_available", "primary_candidate", "ordered_vector_9", 20.0),
    MeasurementFamily("NIE", "Network integration efficiency",
                      "D1.sensor_weighted_global_efficiency_invweight.v0",
                      "kernel_available", "primary_candidate", "scalar", 20.0),
    MeasurementFamily("RD", "Representational dispersion",
                      "RD.pipeline_native_window_centered_log_psd_dispersion.v1",
                      "new_experimental_estimator", "primary_candidate", "scalar", 20.0),
    MeasurementFamily("OI", "Ordinal irregularity",
                      "OI.matrix_input_channel_median_normalized_PE5_delay1.v1",
                      "kernel_available", "primary_candidate", "scalar", 20.0),
    MeasurementFamily("SC", "Sequence complexity",
                      "SC.pipeline_native_window_lz78_exact_multiset_ratio.v1",
                      "new_experimental_estimator", "primary_candidate", "scalar", 20.0),
    MeasurementFamily("ED", "Effective dimensionality",
                      "ED.sensor_covariance_participation_ratio_raw_matrix.v1",
                      "kernel_available", "primary_candidate", "scalar", 20.0),
    MeasurementFamily("SCG", "Spatial coupling gradient",
                      "spatial_dwpli_distance_spearman_alpha.v1",
                      "optional_existing_estimator", "optional_candidate", "scalar_long_support", 60.0),
)

FULL_VECTOR_CHALLENGE_001_BY_FAMILY: dict[str, MeasurementFamily] = {
    item.family_id: item for item in FULL_VECTOR_CHALLENGE_001_REGISTRY
}


def get_full_vector_registry() -> tuple[dict[str, object], ...]:
    """Return all eight families and optional SCG, even when support is absent."""
    return tuple(item.to_dict() for item in FULL_VECTOR_CHALLENGE_001_REGISTRY)


__all__ = [
    "FULL_VECTOR_CHALLENGE_001_BY_FAMILY", "FULL_VECTOR_CHALLENGE_001_REGISTRY",
    "MeasurementFamily", "get_full_vector_registry",
]
