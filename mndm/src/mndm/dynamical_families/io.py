"""HDF5 namespace helpers for dynamical families.

v3 writes only ``/dynamical_families/<family_id>/v1``. Legacy
``/orthogonal_dynamics/<old_name>/v1`` groups may be read, never written.
"""

from __future__ import annotations

from typing import Any

from .registry import (
    CANONICAL_HDF5_ROOT,
    FAMILIES,
    LEGACY_FAMILY_HDF5_NAMES,
    LEGACY_HDF5_ROOT,
    get_family,
)
from ..measurement_certificate import read_measurement_certificate
from ..inferential_grain import read_inferential_grain


def canonical_group_path(family_id: str) -> str | None:
    """Return the v3 write path without a leading slash, if the family is written."""
    namespace = get_family(family_id).get("namespace")
    if not namespace:
        return None
    return str(namespace).lstrip("/")


def legacy_group_path(family_id: str) -> str | None:
    """Return the pre-v3 path without a leading slash, if one existed."""
    legacy_name = LEGACY_FAMILY_HDF5_NAMES.get(family_id)
    if legacy_name is None:
        return None
    return f"{LEGACY_HDF5_ROOT}/{legacy_name}/v1"


def resolve_family_group(handle: Any, family_id: str) -> dict[str, Any]:
    """Open a family v1 group, preferring the canonical namespace.

    ``handle`` is an h5py File or Group that supports ``__contains__`` and
    item access. Missing families return ``group=None``.
    """
    canonical = canonical_group_path(family_id)
    if canonical is not None and canonical in handle:
        return {
            "group": handle[canonical],
            "path": f"/{canonical}",
            "namespace_origin": "canonical",
            "family_id": family_id,
        }
    legacy = legacy_group_path(family_id)
    if legacy is not None and legacy in handle:
        return {
            "group": handle[legacy],
            "path": f"/{legacy}",
            "namespace_origin": "legacy_orthogonal_dynamics",
            "family_id": family_id,
        }
    return {
        "group": None,
        "path": None,
        "namespace_origin": None,
        "family_id": family_id,
    }


def family_tree_present(handle: Any) -> bool:
    """Return True if a canonical or legacy family container exists."""
    return CANONICAL_HDF5_ROOT in handle or LEGACY_HDF5_ROOT in handle


def registered_family_ids() -> tuple[str, ...]:
    """Return canonical family ids in registry order."""
    return tuple(FAMILIES)


__all__ = [
    "canonical_group_path",
    "family_tree_present",
    "legacy_group_path",
    "read_inferential_grain",
    "read_measurement_certificate",
    "registered_family_ids",
    "resolve_family_group",
]
