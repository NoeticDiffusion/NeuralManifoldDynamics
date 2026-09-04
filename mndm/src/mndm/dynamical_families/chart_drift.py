"""Closed vocabulary for independent chart-space drift (Type C / DF-DRIFT-C1).

C1 uses a supplied chart-space velocity **only** for alignment scalars.
It does not residualize increments and does not identify an Itô SDE drift.
MNPS ``x_dot``, Jacobian intercepts, and same-sample increment means are
not admissible ``b`` sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

SOURCE_NOT_SUPPLIED = "not_supplied"
SOURCE_TRUTH_KNOWN = "truth_known_chart_b"
SOURCE_CROSSFIT = "crossfit_local_chart_b"
SOURCE_PROTOCOL = "protocol_imposed_chart_b"

ALLOWED_WRITE_SOURCES = frozenset(
    {
        SOURCE_NOT_SUPPLIED,
        SOURCE_TRUTH_KNOWN,
        SOURCE_CROSSFIT,
        SOURCE_PROTOCOL,
    }
)

FORBIDDEN_SOURCE_REASONS: dict[str, str] = {
    "mnps_xdot": "mnps_xdot_as_sde_drift",
    "jacobian_intercept": "jacobian_intercept_as_sde_drift",
    "jacobian_residual": "jacobian_derivative_residual_as_diffusion",
    "local_increment_mean": "local_increment_mean_as_sde_drift",
    "committor_potential_gradient": "committor_potential_gradient_as_drift",
}

REASON_NOT_SUPPLIED = "independent_drift_not_supplied"
REASON_C2_CLOSED = "c2_residualize_increments_not_authorized"
REASON_CROSSFIT_BEFORE_M3 = "crossfit_not_authorized_before_m3"
REASON_PROTOCOL_CLOSED = "protocol_imposed_chart_b_not_authorized"
REASON_RESIDUALIZE_REQUIRES_DRIFT = "residualize_increments_requires_drift"

MODE_ALIGNMENT_ONLY = "alignment_only"
MODE_RESIDUALIZE = "residualize_increments"
MODE_NOT_SUPPLIED = "not_supplied"

A_SEMANTICS_RAW = "raw_increment_covariance"
A_SEMANTICS_RESIDUALIZED = "residualized_increment_covariance"
RATIO_SEMANTICS_C1 = "chart_velocity_to_increment_spread"
RATIO_SEMANTICS_C2 = "chart_velocity_to_residualized_spread"
RATIO_SEMANTICS_NA = "not_applicable"
RESIDUALIZATION_NONE = "none"
RESIDUALIZATION_SUPPLIED = "supplied_drift_times_dt"


@dataclass(frozen=True)
class ChartDriftResolution:
    """Fail-closed resolution of a requested chart-space drift field."""

    field: np.ndarray | None
    source: str
    mode: str
    residualize_increments: bool
    failure_reason: str | None
    a_semantics: str
    ratio_semantics: str
    residualization_token: str


def _closed(reason: str, source: str) -> ChartDriftResolution:
    return ChartDriftResolution(
        field=None,
        source=source,
        mode=MODE_NOT_SUPPLIED,
        residualize_increments=False,
        failure_reason=reason,
        a_semantics=A_SEMANTICS_RAW,
        ratio_semantics=RATIO_SEMANTICS_NA,
        residualization_token=RESIDUALIZATION_NONE,
    )


def _normalize_token(value: str | None, default: str) -> str:
    token = str(value or default).strip()
    return token if token else default


def resolve_chart_drift(
    *,
    source: str | None = None,
    field: np.ndarray | None = None,
    mode: str | None = None,
    enabled: bool = False,
) -> ChartDriftResolution:
    """Resolve a requested drift source without inventing an alignment field.

    Ingest must call this with ``field=None``. A non-None field is only for
    library / synthetic qualification (truth-known chart ``b``).
    """
    token = _normalize_token(source, SOURCE_NOT_SUPPLIED)
    mode_token = _normalize_token(mode, MODE_ALIGNMENT_ONLY)

    if token in FORBIDDEN_SOURCE_REASONS:
        return _closed(FORBIDDEN_SOURCE_REASONS[token], token)

    if mode_token == MODE_RESIDUALIZE:
        return _closed(REASON_C2_CLOSED, token)

    if not enabled or token == SOURCE_NOT_SUPPLIED:
        return _closed(REASON_NOT_SUPPLIED, SOURCE_NOT_SUPPLIED)

    if token == SOURCE_CROSSFIT:
        return _closed(REASON_CROSSFIT_BEFORE_M3, token)

    if token == SOURCE_PROTOCOL:
        return _closed(REASON_PROTOCOL_CLOSED, token)

    if token not in ALLOWED_WRITE_SOURCES:
        return _closed(REASON_NOT_SUPPLIED, token)

    if token == SOURCE_TRUTH_KNOWN:
        if field is None:
            return _closed(REASON_NOT_SUPPLIED, token)
        return ChartDriftResolution(
            field=np.asarray(field, dtype=float),
            source=token,
            mode=MODE_ALIGNMENT_ONLY,
            residualize_increments=False,
            failure_reason=None,
            a_semantics=A_SEMANTICS_RAW,
            ratio_semantics=RATIO_SEMANTICS_C1,
            residualization_token=RESIDUALIZATION_NONE,
        )

    return _closed(REASON_NOT_SUPPLIED, token)


def resolve_ingest_chart_drift(
    diffusion_cfg: Mapping[str, Any] | None,
) -> ChartDriftResolution:
    """Ingest-only resolver: never accepts a vector, including ``x_dot``."""
    cfg = diffusion_cfg if isinstance(diffusion_cfg, Mapping) else {}
    drift_cfg = cfg.get("drift")
    if not isinstance(drift_cfg, Mapping):
        return resolve_chart_drift(source=SOURCE_NOT_SUPPLIED, enabled=False)
    return resolve_chart_drift(
        source=drift_cfg.get("source"),
        field=None,
        mode=drift_cfg.get("mode"),
        enabled=bool(drift_cfg.get("enabled", False)),
    )
