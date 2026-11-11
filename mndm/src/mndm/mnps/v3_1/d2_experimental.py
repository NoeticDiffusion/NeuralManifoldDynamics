"""Experimental D2 local/distributed coupling proxy in EEG sensor space.

This module adds one fixed pair rule and an end-to-end in-memory signal
prototype beside the generic D2 mask arithmetic. It reads no source files or
electrode-position sidecars. The signal prototype uses signed zero-lag Pearson
r with a caller-declared reference and the rule uses only the canonical letter
prefix in supported 10-20/10-10 sensor labels:

* local proxy: both labels have the same exact prefix (for example ``Fp1``
  and ``Fp2``);
* distributed proxy: labels have different prefixes (for example ``Fp1``
  and ``P3``).

The terms local and distributed describe this label-family partition only.
They do not assert physical distance, cortical location, anatomical regions,
or functional networks. Position-based near/far remains unsupported until its
coordinate units and frame are declared and verified.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import re
from typing import Literal, Sequence

import numpy as np

from .candidates import MNPS31WindowSupport
from .eeg_metrics import (
    EEGCandidateMeasurement,
    _support_reason,
    _unsupported,
    compute_d2_sensor_pairclass_difference,
)

D2_LABEL_FAMILY_ESTIMATOR_ID = (
    "D2.sensor_label_family_local_distributed_coupling_difference.v0"
)
D2_LABEL_FAMILY_PAIR_DEFINITION_ID = (
    "D2.sensor_10_20_label_prefix_same_vs_different.v0"
)
D2_SIGNED_ZERO_LAG_PEARSON_ESTIMATOR_ID = (
    "D2.sensor_signed_zero_lag_pearson_matrix.v0"
)
D2_LABEL_PREFIX_RULE = (
    "local=same canonical sensor-label letter prefix; "
    "distributed=different canonical sensor-label letter prefixes"
)

# Longest-first parsing keeps FC, FT, CP, TP, PO, AF and Fp distinct from
# their one-letter prefixes. The spelling is canonicalized case-insensitively.
_LABEL_PREFIXES = ("FP", "AF", "FC", "FT", "CP", "TP", "PO", "F", "C", "T", "P", "O")
_LABEL_SUFFIX = re.compile(r"^(?:Z|[0-9]{1,2})$")


@dataclass(frozen=True)
class D2LabelFamilyPairMasks:
    """Fixed pair masks derived only from an ordered set of sensor labels."""

    channel_names: tuple[str, ...]
    label_families: tuple[str, ...] | None
    local_mask: np.ndarray | None
    distributed_mask: np.ndarray | None
    local_pair_count: int
    distributed_pair_count: int
    supported: bool
    reason: str | None
    pair_definition_id: str = D2_LABEL_FAMILY_PAIR_DEFINITION_ID


def _canonical_prefix(label: str) -> str | None:
    normalized = label.strip().upper()
    for prefix in _LABEL_PREFIXES:
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix) :]
            if _LABEL_SUFFIX.fullmatch(suffix):
                return prefix
    return None


def _ordered_label_hash(labels: Sequence[str]) -> str:
    normalized = "\n".join(label.strip().casefold() for label in labels)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_d2_sensor_label_family_pair_masks(
    channel_names: Sequence[str],
) -> D2LabelFamilyPairMasks:
    """Build local-proxy and distributed-proxy masks from fixed labels.

    Every unordered off-diagonal sensor pair is assigned to exactly one mask:
    pairs sharing an exact supported label prefix go to ``local_mask``; pairs
    with different prefixes go to ``distributed_mask``. Channel order affects
    mask indexing and the node-order hash, but not the label-based rule.

    This rule requires at least one pair in each class. It does not set a
    global channel-count floor or silently drop unknown labels.
    """
    empty_names: tuple[str, ...] = ()
    try:
        if isinstance(channel_names, (str, bytes)):
            raise TypeError
        labels = tuple(channel_names)
    except (TypeError, ValueError):
        return D2LabelFamilyPairMasks(
            channel_names=empty_names,
            label_families=None,
            local_mask=None,
            distributed_mask=None,
            local_pair_count=0,
            distributed_pair_count=0,
            supported=False,
            reason="invalid_channel_label_sequence",
        )

    if len(labels) < 2:
        return D2LabelFamilyPairMasks(
            channel_names=tuple(str(label) for label in labels),
            label_families=None,
            local_mask=None,
            distributed_mask=None,
            local_pair_count=0,
            distributed_pair_count=0,
            supported=False,
            reason="insufficient_channels_for_pair_classes",
        )
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        return D2LabelFamilyPairMasks(
            channel_names=tuple(str(label) for label in labels),
            label_families=None,
            local_mask=None,
            distributed_mask=None,
            local_pair_count=0,
            distributed_pair_count=0,
            supported=False,
            reason="invalid_channel_label",
        )

    normalized_labels = tuple(label.strip().casefold() for label in labels)
    if len(set(normalized_labels)) != len(normalized_labels):
        return D2LabelFamilyPairMasks(
            channel_names=tuple(label.strip() for label in labels),
            label_families=None,
            local_mask=None,
            distributed_mask=None,
            local_pair_count=0,
            distributed_pair_count=0,
            supported=False,
            reason="duplicate_channel_label",
        )

    families = tuple(_canonical_prefix(label) for label in labels)
    if any(family is None for family in families):
        return D2LabelFamilyPairMasks(
            channel_names=tuple(label.strip() for label in labels),
            label_families=tuple(family or "" for family in families),
            local_mask=None,
            distributed_mask=None,
            local_pair_count=0,
            distributed_pair_count=0,
            supported=False,
            reason="unsupported_sensor_label_for_fixed_pair_rule",
        )

    local = np.zeros((len(labels), len(labels)), dtype=bool)
    distributed = np.zeros_like(local)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            target = local if families[i] == families[j] else distributed
            target[i, j] = target[j, i] = True

    local_count = int(np.count_nonzero(np.triu(local, k=1)))
    distributed_count = int(np.count_nonzero(np.triu(distributed, k=1)))
    reason = None
    if local_count == 0 or distributed_count == 0:
        reason = "insufficient_pair_class_support"
    return D2LabelFamilyPairMasks(
        channel_names=tuple(label.strip() for label in labels),
        label_families=tuple(str(family) for family in families),
        local_mask=local,
        distributed_mask=distributed,
        local_pair_count=local_count,
        distributed_pair_count=distributed_count,
        supported=reason is None,
        reason=reason,
    )


def compute_d2_sensor_label_family_local_distributed(
    coupling: np.ndarray,
    channel_names: Sequence[str],
    support: MNPS31WindowSupport,
    *,
    coupling_estimator_id: str,
) -> EEGCandidateMeasurement:
    """Compute mean(local-proxy coupling) minus mean(distributed-proxy coupling).

    ``coupling`` must already be a finite symmetric signed-unit-interval matrix
    in exactly ``channel_names`` order. Its EEG estimator, reference/leakage
    policy, and construction remain explicit caller responsibilities.
    """
    try:
        if isinstance(channel_names, (str, bytes)):
            raise TypeError
        submitted_labels = tuple(str(label).strip() for label in channel_names)
    except (TypeError, ValueError):
        submitted_labels = ()
    label_provenance: dict[str, object] = {
        "channel_names": list(submitted_labels),
        "channel_count": len(submitted_labels),
        "channel_order_sha256": _ordered_label_hash(submitted_labels),
        "channel_set_sha256": _label_set_hash(submitted_labels),
    }
    support_error = _support_reason(support)
    if support_error:
        return _unsupported(
            candidate_id="D2",
            estimator_id=D2_LABEL_FAMILY_ESTIMATOR_ID,
            support=support,
            reason=support_error,
            details={
                "pair_definition_id": D2_LABEL_FAMILY_PAIR_DEFINITION_ID,
                **label_provenance,
            },
        )

    pair_masks = build_d2_sensor_label_family_pair_masks(channel_names)
    if not pair_masks.supported:
        return _unsupported(
            candidate_id="D2",
            estimator_id=D2_LABEL_FAMILY_ESTIMATOR_ID,
            support=support,
            reason=pair_masks.reason or "pair_class_support_missing",
            details={
                "pair_definition_id": D2_LABEL_FAMILY_PAIR_DEFINITION_ID,
                "pair_rule": D2_LABEL_PREFIX_RULE,
                "channel_names": list(pair_masks.channel_names),
                "channel_count": len(pair_masks.channel_names),
                "channel_order_sha256": _ordered_label_hash(pair_masks.channel_names),
                "channel_set_sha256": _label_set_hash(pair_masks.channel_names),
                "label_families": (
                    list(pair_masks.label_families)
                    if pair_masks.label_families is not None
                    else None
                ),
                "local_pair_count": pair_masks.local_pair_count,
                "distributed_pair_count": pair_masks.distributed_pair_count,
                "position_metadata_used": False,
                "coordinate_units": None,
                "coordinate_frame": None,
                "space_claim": "sensor_label_family_proxy_only",
            },
        )

    assert pair_masks.local_mask is not None
    assert pair_masks.distributed_mask is not None
    base = compute_d2_sensor_pairclass_difference(
        coupling,
        pair_masks.local_mask,
        pair_masks.distributed_mask,
        support,
        variant="within_between",
        pair_definition_id=D2_LABEL_FAMILY_PAIR_DEFINITION_ID,
        coupling_estimator_id=coupling_estimator_id,
        node_order_sha256=_ordered_label_hash(pair_masks.channel_names),
    )
    details = {
        **base.details,
        "estimator_definition": D2_LABEL_PREFIX_RULE,
        "pair_rule": D2_LABEL_PREFIX_RULE,
        "pair_definition_id": D2_LABEL_FAMILY_PAIR_DEFINITION_ID,
        **label_provenance,
        "label_families": list(pair_masks.label_families or ()),
        "local_pair_count": pair_masks.local_pair_count,
        "distributed_pair_count": pair_masks.distributed_pair_count,
        "position_metadata_used": False,
        "coordinate_units": None,
        "coordinate_frame": None,
        "space_claim": "sensor_label_family_proxy_only",
        "contrast_order": "local_minus_distributed",
    }
    return replace(base, estimator_id=D2_LABEL_FAMILY_ESTIMATOR_ID, details=details)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _label_set_hash(labels: Sequence[str]) -> str:
    normalized = "\n".join(sorted(label.strip().casefold() for label in labels))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _unsupported_signal_result(
    support: MNPS31WindowSupport,
    reason: str,
    *,
    details: dict[str, object] | None = None,
) -> EEGCandidateMeasurement:
    return _unsupported(
        candidate_id="D2",
        estimator_id=D2_LABEL_FAMILY_ESTIMATOR_ID,
        support=support,
        reason=reason,
        details={
            "pair_definition_id": D2_LABEL_FAMILY_PAIR_DEFINITION_ID,
            "coupling_estimator_id": D2_SIGNED_ZERO_LAG_PEARSON_ESTIMATOR_ID,
            "position_metadata_used": False,
            "coordinate_units": None,
            "coordinate_frame": None,
            "space_claim": "sensor_label_family_proxy_only",
            **(details or {}),
        },
    )


def _signed_zero_lag_pearson_matrix(sensor_signals: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """Return pairwise Pearson r, rejecting nonfinite or constant channels."""
    try:
        values = np.asarray(sensor_signals, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_numeric_input"
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 3:
        return None, "invalid_signal_matrix_shape_or_sample_count"
    if not np.isfinite(values).all():
        return None, "non_finite_input"

    # Per-channel scaling before centering avoids overflow for large finite
    # amplitudes without changing Pearson correlation.
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(scale == 0.0):
        return None, "constant_sensor_channel"
    centered = values / scale
    centered = centered - np.mean(centered, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        return None, "non_finite_pearson_support"
    if np.any(norms == 0.0):
        return None, "constant_sensor_channel"
    standardized = centered / norms
    matrix = standardized @ standardized.T
    if not np.isfinite(matrix).all():
        return None, "non_finite_pearson_estimate"
    matrix = np.clip((matrix + matrix.T) * 0.5, -1.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix, None


def compute_d2_sensor_label_family_from_signals(
    sensor_signals: np.ndarray,
    channel_names: Sequence[str],
    support: MNPS31WindowSupport,
    *,
    surface_id: str,
    source_id: str,
    source_content_sha256: str,
    preprocessing_id: str,
    signal_units: str,
    reference_mode: Literal["recorded", "average_over_exact_surface"],
    reference_id: str,
) -> EEGCandidateMeasurement:
    """Estimate Pearson coupling and the fixed label-family D2 contrast.

    Input is an already selected ``[sensor, sample]`` surface matrix. The
    coupling estimator is signed zero-lag Pearson r. The reference choice is
    mandatory: ``recorded`` leaves the supplied data untouched;
    ``average_over_exact_surface`` subtracts the instantaneous mean of exactly
    the supplied rows, with no channel dropping or substitution.

    Source, preprocessing, surface, units, and reference identities are
    required and returned with the value. This function reads no source files,
    position sidecars, or EEG data itself.
    """
    provenance = {
        "surface_id": surface_id,
        "source_id": source_id,
        "source_content_sha256": source_content_sha256,
        "preprocessing_id": preprocessing_id,
        "signal_units": signal_units,
        "reference_id": reference_id,
        "reference_mode": reference_mode,
    }
    support_error = _support_reason(support)
    if support_error:
        return _unsupported_signal_result(
            support, support_error, details=provenance
        )

    if any(not isinstance(value, str) or not value.strip() for value in
           (surface_id, source_id, preprocessing_id, signal_units, reference_id)):
        return _unsupported_signal_result(support, "measurement_provenance_missing")
    if not isinstance(source_content_sha256, str) or not _is_sha256(source_content_sha256):
        return _unsupported_signal_result(support, "source_content_hash_missing_or_invalid")
    if reference_mode not in {"recorded", "average_over_exact_surface"}:
        return _unsupported_signal_result(support, "unknown_reference_mode")

    pair_masks = build_d2_sensor_label_family_pair_masks(channel_names)
    if not pair_masks.supported:
        return _unsupported_signal_result(
            support,
            pair_masks.reason or "pair_class_support_missing",
            details={
                **provenance,
                "channel_names": list(pair_masks.channel_names),
                "label_families": (list(pair_masks.label_families)
                                   if pair_masks.label_families is not None else None),
                "local_pair_count": pair_masks.local_pair_count,
                "distributed_pair_count": pair_masks.distributed_pair_count,
            },
        )

    try:
        signals = np.asarray(sensor_signals, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return _unsupported_signal_result(support, "invalid_numeric_input", details=provenance)
    if signals.ndim != 2 or signals.shape[0] != len(pair_masks.channel_names):
        return _unsupported_signal_result(
            support,
            "channel_count_mismatch",
            details={**provenance, "channel_count": int(signals.shape[0]) if signals.ndim == 2 else None,
                     "label_count": len(pair_masks.channel_names)},
        )
    if signals.shape[1] != support.sample_count:
        return _unsupported_signal_result(
            support,
            "sample_count_mismatch",
            details={**provenance, "sample_count": int(signals.shape[1]),
                     "support_sample_count": int(support.sample_count)},
        )
    if support.sample_count < 3 or not np.isfinite(support.sample_rate_hz) or support.sample_rate_hz <= 0:
        return _unsupported_signal_result(support, "invalid_sample_grid", details=provenance)
    duration = float(support.window_end_sec - support.window_start_sec)
    expected_duration = float(support.sample_count / support.sample_rate_hz)
    if not np.isfinite(duration) or duration <= 0 or not np.isclose(
        duration, expected_duration, rtol=0.0, atol=max(1e-9, 0.5 / support.sample_rate_hz)
    ):
        return _unsupported_signal_result(support, "sample_grid_duration_mismatch", details=provenance)
    if not np.isfinite(signals).all():
        return _unsupported_signal_result(support, "non_finite_input", details=provenance)

    selected = signals.copy()
    reference_set_sha256 = None
    reference_channel_names: list[str] | None = None
    if reference_mode == "average_over_exact_surface":
        # The selected rows are the full declared reference set.
        selected -= np.mean(selected, axis=0, keepdims=True)
        reference_channel_names = list(pair_masks.channel_names)
        reference_set_sha256 = _ordered_label_hash(pair_masks.channel_names)
    if not np.isfinite(selected).all():
        return _unsupported_signal_result(support, "non_finite_referenced_input", details=provenance)

    coupling, coupling_error = _signed_zero_lag_pearson_matrix(selected)
    if coupling_error:
        return _unsupported_signal_result(
            support,
            coupling_error,
            details={**provenance,
                     "channel_names": list(pair_masks.channel_names),
                     "channel_count": len(pair_masks.channel_names),
                     "sample_count": int(selected.shape[1]),
                     "reference_channel_names": reference_channel_names,
                     "reference_set_sha256": reference_set_sha256},
        )
    assert coupling is not None
    assert pair_masks.local_mask is not None
    assert pair_masks.distributed_mask is not None
    base = compute_d2_sensor_pairclass_difference(
        coupling,
        pair_masks.local_mask,
        pair_masks.distributed_mask,
        support,
        variant="within_between",
        pair_definition_id=D2_LABEL_FAMILY_PAIR_DEFINITION_ID,
        coupling_estimator_id=D2_SIGNED_ZERO_LAG_PEARSON_ESTIMATOR_ID,
        node_order_sha256=_ordered_label_hash(pair_masks.channel_names),
    )
    local_mean = float(np.mean(coupling[pair_masks.local_mask]))
    distributed_mean = float(np.mean(coupling[pair_masks.distributed_mask]))
    details = {
        **base.details,
        **provenance,
        "estimator_definition": D2_LABEL_PREFIX_RULE,
        "coupling_formula": "Pearson r across samples at zero lag for each unordered sensor pair",
        "coupling_estimator_id": D2_SIGNED_ZERO_LAG_PEARSON_ESTIMATOR_ID,
        "pair_rule": D2_LABEL_PREFIX_RULE,
        "pair_definition_id": D2_LABEL_FAMILY_PAIR_DEFINITION_ID,
        "channel_names": list(pair_masks.channel_names),
        "label_families": list(pair_masks.label_families or ()),
        "channel_count": len(pair_masks.channel_names),
        "channel_order_sha256": _ordered_label_hash(pair_masks.channel_names),
        "channel_set_sha256": _label_set_hash(pair_masks.channel_names),
        "sample_count": int(selected.shape[1]),
        "sample_rate_hz": float(support.sample_rate_hz),
        "window_start_sec": float(support.window_start_sec),
        "window_end_sec": float(support.window_end_sec),
        "local_pair_count": pair_masks.local_pair_count,
        "distributed_pair_count": pair_masks.distributed_pair_count,
        "local_mean_coupling": local_mean,
        "distributed_mean_coupling": distributed_mean,
        "reference_channel_names": reference_channel_names,
        "reference_set_sha256": reference_set_sha256,
        "position_metadata_used": False,
        "coordinate_units": None,
        "coordinate_frame": None,
        "space_claim": "sensor_label_family_proxy_only",
        "contrast_order": "local_minus_distributed",
        "qc": {
            "input_finite": True,
            "reference_applied_to_exact_surface_only": (
                reference_mode == "average_over_exact_surface"
            ),
            "constant_channels": 0,
            "pair_classes_supported": True,
            "position_metadata_required": False,
        },
        "coupling_matrix": coupling.tolist(),
    }
    return replace(base, estimator_id=D2_LABEL_FAMILY_ESTIMATOR_ID, details=details)


__all__ = [
    "D2_LABEL_FAMILY_ESTIMATOR_ID",
    "D2_LABEL_FAMILY_PAIR_DEFINITION_ID",
    "D2_SIGNED_ZERO_LAG_PEARSON_ESTIMATOR_ID",
    "D2_LABEL_PREFIX_RULE",
    "D2LabelFamilyPairMasks",
    "build_d2_sensor_label_family_pair_masks",
    "compute_d2_sensor_label_family_local_distributed",
    "compute_d2_sensor_label_family_from_signals",
]
