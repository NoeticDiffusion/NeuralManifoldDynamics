"""Support / capability signature from existing projection metadata.

Round 4 records which features contributed to each coordinate and what this
modality/contract allows. It does not invent coverage fractions or rewrite
``geometry_contract``.

```text
coords_9d present  ≠  same feature support  ≠  same modality capability
```

Readers never infer ``chart_3d=yes`` from ``/mnps_3d`` or ``spread=gated``
from a missing reachability group.
"""

from __future__ import annotations

from typing import Any, Mapping

from .measurement_certificate import NOT_RECORDED, _group_scalar
from .schema import MNPS_AXIS_NAMES, mnps_9d_CANONICAL_ORDER

SUPPORT_SIGNATURE_SCHEMA_VERSION = "mndm.support_signature.v1"

SOURCE_DIRECT = "direct"
SOURCE_FALLBACK = "fallback"
SOURCE_MIXED = "mixed"
SOURCE_NOT_APPLICABLE = "not_applicable"
FALLBACK_NONE = "none"

VALID_SOURCE_WRITE = frozenset(
    {SOURCE_DIRECT, SOURCE_FALLBACK, SOURCE_MIXED, SOURCE_NOT_APPLICABLE}
)
VALID_AXIS_SOURCE_WRITE = frozenset(
    {SOURCE_DIRECT, SOURCE_MIXED, SOURCE_NOT_APPLICABLE}
)
VALID_SEMANTIC_EQUIVALENCE_WRITE = frozenset({"true", "false", "not_applicable"})
VALID_MODALITY_WRITE = frozenset({"eeg", "ieeg", "fmri", "meg", "unknown"})
VALID_CAPABILITY_WRITE = frozenset(
    {
        "yes",
        "limited",
        "conditional",
        "gated",
        "overlay_only",
        "no_generic_ingest",
        "perturbational_only",
        "opt_in",
        "not_assessed",
    }
)

CAPABILITY_KEYS = (
    "chart_3d",
    "chart_9d",
    "mnj_3d",
    "mnj_9d",
    "diffusion",
    "spread",
    "destination",
    "resilience",
)

COORDINATE_FIELDS = ("source", "fallback_feature", "semantic_equivalence")
AXIS_CHILDREN = {
    "m": ("m_a", "m_e", "m_o"),
    "d": ("d_n", "d_l", "d_s"),
    "e": ("e_e", "e_s", "e_m"),
}

FAMILY_CAPABILITY = {
    "diffusion": "overlay_only",
    "spread": "gated",
    "destination": "no_generic_ingest",
    "resilience": "perturbational_only",
}

_MODALITY_CHART_MNJ = {
    "eeg": {"chart_3d": "yes", "chart_9d": "yes", "mnj_3d": "yes", "mnj_9d": "conditional"},
    "ieeg": {"chart_3d": "yes", "chart_9d": "yes", "mnj_3d": "yes", "mnj_9d": "yes"},
    "fmri": {"chart_3d": "yes", "chart_9d": "yes", "mnj_3d": "limited", "mnj_9d": "limited"},
    "meg": {"chart_3d": "yes", "chart_9d": "yes", "mnj_3d": "yes", "mnj_9d": "conditional"},
    "unknown": {
        "chart_3d": "not_assessed",
        "chart_9d": "not_assessed",
        "mnj_3d": "not_assessed",
        "mnj_9d": "not_assessed",
    },
}

_MODALITY_ALIASES = {
    "eeg": "eeg",
    "nwb": "eeg",
    "nwb_mouse_eeg": "eeg",
    "ieeg": "ieeg",
    "ephys": "ieeg",
    "fmri": "fmri",
    "meg": "meg",
}

SIGNATURE_ORIGIN_CANONICAL = "canonical"
SIGNATURE_ORIGIN_ABSENT = "legacy_absent"


def normalize_modality(raw: Any) -> str:
    """Map a config modality string onto the write vocabulary."""
    token = str(raw or "").strip().lower()
    if token in VALID_MODALITY_WRITE:
        return token
    return _MODALITY_ALIASES.get(token, "unknown")


def validate_write_coordinate(
    *,
    source: str,
    fallback_feature: str,
    semantic_equivalence: str,
) -> None:
    """Reject unknown coordinate tokens on write."""
    if source not in VALID_SOURCE_WRITE:
        raise ValueError(f"Unsupported coordinate source for write: {source}")
    if not str(fallback_feature or "").strip():
        raise ValueError("fallback_feature must be a non-empty string")
    if semantic_equivalence not in VALID_SEMANTIC_EQUIVALENCE_WRITE:
        raise ValueError(
            f"Unsupported semantic_equivalence for write: {semantic_equivalence}"
        )


def validate_write_capability(capability: Mapping[str, str]) -> None:
    """Reject unknown capability keys or tokens on write."""
    if not isinstance(capability, Mapping):
        raise TypeError("capability must be a mapping")
    unknown = [str(key) for key in capability if str(key) not in CAPABILITY_KEYS]
    if unknown:
        raise ValueError(f"Unknown capability keys: {unknown}")
    for key in CAPABILITY_KEYS:
        if key not in capability:
            raise ValueError(f"Missing capability key: {key}")
        value = str(capability[key])
        if value not in VALID_CAPABILITY_WRITE:
            raise ValueError(f"Unsupported capability token for {key}: {value}")


def validate_write_support_signature(mapping: Mapping[str, Any]) -> None:
    """Reject an incomplete or unknown-token write mapping."""
    if not isinstance(mapping, Mapping):
        raise TypeError("support_signature must be a mapping")
    modality = str(mapping.get("modality") or "").strip()
    if modality not in VALID_MODALITY_WRITE:
        raise ValueError(f"Unsupported modality for write: {modality}")
    coordinates = mapping.get("coordinates")
    if not isinstance(coordinates, Mapping):
        raise ValueError("support_signature.coordinates must be a mapping")
    for name in mnps_9d_CANONICAL_ORDER:
        record = coordinates.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Missing coordinate record: {name}")
        validate_write_coordinate(
            source=str(record.get("source") or ""),
            fallback_feature=str(record.get("fallback_feature") or ""),
            semantic_equivalence=str(record.get("semantic_equivalence") or ""),
        )
    axes = mapping.get("axes_3d")
    if not isinstance(axes, Mapping):
        raise ValueError("support_signature.axes_3d must be a mapping")
    for axis in MNPS_AXIS_NAMES:
        record = axes.get(axis)
        if not isinstance(record, Mapping):
            raise ValueError(f"Missing axes_3d record: {axis}")
        source = str(record.get("source") or "")
        if source not in VALID_AXIS_SOURCE_WRITE:
            raise ValueError(f"Unsupported axes_3d source for {axis}: {source}")
    validate_write_capability(mapping.get("capability") or {})


def capability_for_modality(modality: str) -> dict[str, str]:
    """Return the contract-class capability row for one modality."""
    normalized = normalize_modality(modality)
    chart_mnj = dict(_MODALITY_CHART_MNJ[normalized])
    return {**chart_mnj, **FAMILY_CAPABILITY}


def _direct_coordinate() -> dict[str, str]:
    return {
        "source": SOURCE_DIRECT,
        "fallback_feature": FALLBACK_NONE,
        "semantic_equivalence": "not_applicable",
    }


def _fallback_coordinate(feature: str) -> dict[str, str]:
    name = str(feature or "").strip() or "unspecified_fallback"
    validate_write_coordinate(
        source=SOURCE_FALLBACK,
        fallback_feature=name,
        semantic_equivalence="false",
    )
    return {
        "source": SOURCE_FALLBACK,
        "fallback_feature": name,
        "semantic_equivalence": "false",
    }


def _coordinate_from_policy(
    name: str,
    metric_policies: Mapping[str, Any],
    actual_metrics: Mapping[str, Any],
) -> dict[str, str]:
    record = _direct_coordinate()
    policy = metric_policies.get(name, {}) if isinstance(metric_policies, Mapping) else {}
    if not isinstance(policy, Mapping) or not policy:
        return record
    preferred = str(policy.get("preferred") or "").strip()
    fallbacks = [
        str(item).strip()
        for item in (policy.get("fallbacks") or [])
        if str(item).strip()
    ]
    actual = str(actual_metrics.get(name) or "").strip() if isinstance(actual_metrics, Mapping) else ""
    if not actual:
        return record
    if preferred and actual == preferred:
        return record
    if actual in fallbacks or (preferred and actual != preferred):
        return _fallback_coordinate(actual)
    return record


def _axes_from_coordinates(coordinates: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    axes: dict[str, dict[str, str]] = {}
    for axis in MNPS_AXIS_NAMES:
        children = AXIS_CHILDREN[axis]
        sources = [
            str((coordinates.get(child) or {}).get("source") or SOURCE_NOT_APPLICABLE)
            for child in children
        ]
        if all(source == SOURCE_DIRECT for source in sources):
            axes[axis] = {"source": SOURCE_DIRECT}
        elif all(source == SOURCE_NOT_APPLICABLE for source in sources):
            axes[axis] = {"source": SOURCE_NOT_APPLICABLE}
        else:
            axes[axis] = {"source": SOURCE_MIXED}
    return axes


def build_support_signature(
    *,
    modality: Any = None,
    metric_policies: Mapping[str, Any] | None = None,
    actual_metrics: Mapping[str, Any] | None = None,
    entropy_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a write-side support signature from existing metadata only."""
    normalized = normalize_modality(modality)
    policies = metric_policies if isinstance(metric_policies, Mapping) else {}
    actual = dict(actual_metrics) if isinstance(actual_metrics, Mapping) else {}
    if isinstance(entropy_meta, Mapping):
        metric = str(entropy_meta.get("metric") or "").strip()
        if metric:
            actual.setdefault("e_e", metric)

    coordinates = {
        name: _coordinate_from_policy(name, policies, actual)
        for name in mnps_9d_CANONICAL_ORDER
    }
    for name, record in coordinates.items():
        validate_write_coordinate(**record)
    capability = capability_for_modality(normalized)
    validate_write_capability(capability)
    return {
        "schema_version": SUPPORT_SIGNATURE_SCHEMA_VERSION,
        "modality": normalized,
        "coordinates": coordinates,
        "axes_3d": _axes_from_coordinates(coordinates),
        "capability": capability,
    }


def _group_child(group: Any, key: str) -> Any | None:
    if group is None:
        return None
    try:
        present = key in group
    except Exception:
        return None
    if not present:
        return None
    return group[key]


def read_support_signature(handle: Any) -> dict[str, Any]:
    """Read a support signature without inferring capability from HDF5 presence.

    Missing groups or fields become ``not_recorded``. ``/mnps_3d`` and missing
    reachability are never consulted.
    """
    root = handle
    v1 = None
    if handle is not None:
        try:
            if "support_signature" in handle:
                root = handle["support_signature"]
        except Exception:
            root = handle
        v1 = _group_child(root, "v1")
        if v1 is None and root is not None:
            try:
                if "modality" in root or "capability" in root or "coordinates" in root:
                    v1 = root
            except Exception:
                v1 = None

    origin = SIGNATURE_ORIGIN_CANONICAL if v1 is not None else SIGNATURE_ORIGIN_ABSENT
    modality = _group_scalar(v1, "modality") if v1 is not None else None
    if modality is None:
        modality = NOT_RECORDED

    coordinates: dict[str, dict[str, str]] = {}
    coords_group = _group_child(v1, "coordinates") if v1 is not None else None
    for name in mnps_9d_CANONICAL_ORDER:
        child = _group_child(coords_group, name)
        record: dict[str, str] = {}
        for field in COORDINATE_FIELDS:
            value = _group_scalar(child, field)
            record[field] = value if value is not None else NOT_RECORDED
        coordinates[name] = record

    axes: dict[str, dict[str, str]] = {}
    axes_group = _group_child(v1, "axes_3d") if v1 is not None else None
    for axis in MNPS_AXIS_NAMES:
        child = _group_child(axes_group, axis)
        value = _group_scalar(child, "source")
        axes[axis] = {"source": value if value is not None else NOT_RECORDED}

    capability: dict[str, str] = {}
    cap_group = _group_child(v1, "capability") if v1 is not None else None
    for key in CAPABILITY_KEYS:
        value = _group_scalar(cap_group, key)
        capability[key] = value if value is not None else NOT_RECORDED

    return {
        "schema_version": _group_scalar(v1, "schema_version") or NOT_RECORDED,
        "modality": modality,
        "coordinates": coordinates,
        "axes_3d": axes,
        "capability": capability,
        "signature_origin": origin,
    }
