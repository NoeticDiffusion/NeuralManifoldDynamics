"""Tests for MNPS payload schema normalization."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def _base_payload():
    from mndm.schema import MNPSPayload

    return MNPSPayload(
        time=np.arange(4, dtype=np.float64),
        x=np.zeros((4, 3), dtype=np.float32),
        x_dot=np.zeros((4, 3), dtype=np.float32),
    )


def test_normalize_payload_requires_jacobian_dot_shape_match():
    from mndm.schema import normalize_payload

    payload = _base_payload()
    payload.jacobian = np.zeros((3, 3, 3), dtype=np.float32)
    payload.jacobian_dot = np.zeros((2, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        normalize_payload(payload)


def test_normalize_payload_rejects_non_numeric_events():
    from mndm.schema import normalize_payload

    payload = _base_payload()
    payload.events = {"stim": np.array(["on", "off"], dtype=object)}
    with pytest.raises(TypeError):
        normalize_payload(payload)


def test_normalize_coords_9d_returns_contiguous_and_schema_version():
    from mndm.schema import mnps_9d_CANONICAL_ORDER, normalize_payload

    payload = _base_payload()
    names = list(mnps_9d_CANONICAL_ORDER)
    values = np.arange(4 * len(names), dtype=np.float32).reshape(4, len(names), order="F")
    payload.coords_9d = values[:, ::-1]  # shuffled order + non-contiguous view
    payload.coords_9d_names = list(reversed(names))
    out = normalize_payload(payload)
    assert out.coords_9d is not None
    assert out.coords_9d.flags["C_CONTIGUOUS"]
    assert out.attrs.get("schema_version") == "mnps_tensor_spec_v2"


def test_normalize_coords_9d_can_allow_all_non_finite_columns():
    from mndm.schema import mnps_9d_CANONICAL_ORDER, _normalize_coords_9d

    names = list(mnps_9d_CANONICAL_ORDER)
    values = np.arange(6 * len(names), dtype=np.float32).reshape(6, len(names))
    e_e_idx = names.index("e_e")
    values[:, e_e_idx] = np.nan

    out_values, out_names, diag = _normalize_coords_9d(
        values,
        names,
        allow_all_non_finite_columns=True,
        return_diagnostics=True,
    )
    assert out_names == names
    assert out_values.shape == values.shape
    assert diag["all_non_finite_names"] == ["e_e"]
    assert diag["all_non_finite_count"] == 1

