"""Tests for MNPS payload schema normalization."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def _base_payload():
    """Internal helper: base payload."""
    from mndm.schema import MNPSPayload

    return MNPSPayload(
        time=np.arange(4, dtype=np.float64),
        x=np.zeros((4, 3), dtype=np.float32),
        x_dot=np.zeros((4, 3), dtype=np.float32),
    )


def test_normalize_payload_requires_jacobian_dot_shape_match():
    """Test normalize payload requires jacobian dot shape match."""
    from mndm.schema import normalize_payload

    payload = _base_payload()
    payload.jacobian = np.zeros((3, 3, 3), dtype=np.float32)
    payload.jacobian_dot = np.zeros((2, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        normalize_payload(payload)


def test_normalize_payload_rejects_non_numeric_events():
    """Test normalize payload rejects non numeric events."""
    from mndm.schema import normalize_payload

    payload = _base_payload()
    payload.events = {"stim": np.array(["on", "off"], dtype=object)}
    with pytest.raises(TypeError):
        normalize_payload(payload)


def test_normalize_coords_9d_returns_contiguous_and_schema_version():
    """Test normalize coords 9d returns contiguous and schema version."""
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
    """Test normalize coords 9d can allow all non finite columns."""
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


def test_normalize_payload_validates_feature_surfaces_and_metadata():
    """Test normalize payload validates feature surfaces and metadata."""
    from mndm.schema import normalize_payload

    payload = _base_payload()
    payload.features_raw_values = np.arange(8, dtype=np.float32).reshape(4, 2)
    payload.features_raw_names = ["eeg_alpha", "eeg_alpha__g_frontal"]
    payload.features_robust_z_values = np.zeros((4, 2), dtype=np.float32)
    payload.features_robust_z_names = ["eeg_alpha", "eeg_alpha__g_frontal"]
    payload.feature_metadata = {
        "feature_name": np.array(["eeg_alpha", "eeg_alpha"], dtype=object),
        "group_label": np.array(["", "frontal"], dtype=object),
        "used_by_mnps_3d": np.array([1, 0], dtype=np.int8),
    }

    out = normalize_payload(payload)
    assert out.features_raw_values is not None
    assert out.features_raw_values.shape == (4, 2)
    assert out.features_robust_z_values is not None
    assert out.features_robust_z_values.shape == (4, 2)
    assert out.features_raw_names == ["eeg_alpha", "eeg_alpha__g_frontal"]
    assert set(out.feature_metadata.keys()) == {"feature_name", "group_label", "used_by_mnps_3d"}

