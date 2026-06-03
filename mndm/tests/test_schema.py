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


def test_normalize_coords_9d_can_allow_duplicate_constant_columns():
    """Duplicate constant subcoordinates can proceed in degraded mode."""
    from mndm.schema import mnps_9d_CANONICAL_ORDER, _normalize_coords_9d

    names = list(mnps_9d_CANONICAL_ORDER)
    values = np.arange(6 * len(names), dtype=np.float32).reshape(6, len(names))
    for name in ("m_a", "m_e", "d_n"):
        values[:, names.index(name)] = 0.0

    out_values, out_names, diag = _normalize_coords_9d(
        values,
        names,
        allow_duplicate_constant_columns=True,
        return_diagnostics=True,
    )

    assert out_names == names
    assert out_values.shape == values.shape
    assert diag["duplicate_constant_pairs"] == {"m_e": "m_a", "d_n": "m_a"}
    assert diag["duplicate_constant_count"] == 2


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


def test_normalize_payload_validates_coordinate_layers_and_sets_2_1_schema():
    """MNDM 2.1 coordinate layers are time-aligned and versioned."""
    from mndm.schema import normalize_payload

    payload = _base_payload()
    payload.coordinate_layers = {
        "coords_3d_cohort_anchored": {
            "values": np.ones((4, 3), dtype=np.float32),
            "names": ["m", "d", "e"],
            "attrs": {"coordinate_contract": "cohort_anchored"},
        }
    }
    payload.feature_anchors = {"spec": {"anchor_id": "unit-test"}}

    out = normalize_payload(payload)

    assert out.attrs["schema_version"] == "mnps_tensor_spec_v2_1"
    assert out.attrs["mndm_version"] == "2.1"
    assert "coords_3d_cohort_anchored" in out.coordinate_layers
    assert out.coordinate_layers["coords_3d_cohort_anchored"]["names"] == ["m", "d", "e"]


def test_normalize_payload_supports_event_windows_codebooks_and_qc():
    """New additive H5 contract payload blocks should normalize cleanly."""
    from mndm.schema import normalize_payload

    payload = _base_payload()
    payload.event_windows = {
        "event_id": np.array([0, 0, 1]),
        "window_id": np.array([0, 1, 2]),
        "bin_label": np.array(["event", "post_near", "event"], dtype=object),
    }
    payload.event_windows_attrs = {"reference": "onset"}
    payload.codebooks = {
        "stage": {
            "codes": np.array([10, 11]),
            "labels": ["Eyes Closed: Every 1000 ms", "Eyes Open: Every 1000 ms"],
            "label_keys": ["eyes_closed", "eyes_open"],
            "attrs": {"source": "consensus"},
        }
    }
    payload.qc_windows = {
        "coverage_ok": np.array([1, 1, 0, 1]),
        "stage_transition_flag": np.array([0, 1, 0, 0]),
        "artifact_score": np.array([0.0, 0.1, 0.2, 0.0]),
    }
    payload.provenance = {"contract": {"export_contract_version": "mndm.eeg_h5_contract.v1"}}
    payload.coverage = {"axis_fraction": np.ones((4, 3), dtype=np.float32)}
    payload.participant_clinical_meta = {"age": 72, "sex": "M"}

    out = normalize_payload(payload)

    assert set(out.event_windows.keys()) == {"event_id", "window_id", "bin_label"}
    assert out.event_windows["event_id"].dtype == np.int32
    assert out.event_windows["bin_label"].dtype.kind in {"U", "O"}
    assert out.qc_windows["coverage_ok"].dtype == np.int8
    assert out.qc_windows["artifact_score"].dtype == np.float32
    assert list(out.codebooks["stage"]["label_keys"]) == ["eyes_closed", "eyes_open"]
    assert out.participant_clinical_meta["age"] == 72

