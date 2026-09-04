"""Round-4 support/capability signature write/read contracts."""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from core.io.h5_writer import write_h5
from mndm.pipeline.robustness_helpers import compute_standard_geometry_contract
from mndm.schema import MNPSPayload, mnps_9d_CANONICAL_ORDER
from mndm.support_signature import (
    CAPABILITY_KEYS,
    SUPPORT_SIGNATURE_SCHEMA_VERSION,
    build_support_signature,
    read_support_signature,
    validate_write_capability,
    validate_write_support_signature,
)

EEG_POLICIES = {
    "e_e": {"preferred": "permutation_entropy", "fallbacks": ["spectral_entropy"]},
    "e_m": {"preferred": "ecg_rmssd", "fallbacks": ["eog_blink_rate"]},
}

GEOMETRY_CONTRACT_TOP_KEYS = {
    "policy_version",
    "status",
    "primary_requires_coords_9d",
    "shared_time_grid",
    "mnps_3d",
    "coords_9d",
    "_row_keep_mask",
}


def _payload(*, support_signature=None, x=None, jacobian=None) -> MNPSPayload:
    state = np.zeros((4, 3), dtype=np.float32) if x is None else np.asarray(x)
    j_hat = (
        np.eye(3, dtype=np.float32)[None, :, :]
        if jacobian is None
        else np.asarray(jacobian)
    )
    return MNPSPayload(
        time=np.arange(state.shape[0], dtype=np.float64),
        x=state,
        x_dot=np.zeros_like(state),
        jacobian=j_hat,
        jacobian_centers=np.asarray([0], dtype=np.int32),
        support_signature=support_signature or {},
    )


def test_eeg_preferred_entropy_is_direct() -> None:
    signature = build_support_signature(
        modality="eeg",
        metric_policies=EEG_POLICIES,
        entropy_meta={
            "metric": "permutation_entropy",
            "backend": "numpy",
            "degraded_mode": False,
        },
    )
    e_e = signature["coordinates"]["e_e"]
    assert e_e["source"] == "direct"
    assert e_e["fallback_feature"] == "none"
    assert e_e["semantic_equivalence"] == "not_applicable"
    assert signature["axes_3d"]["e"]["source"] == "direct"


def test_entropy_fallback_is_not_semantically_equivalent() -> None:
    signature = build_support_signature(
        modality="eeg",
        metric_policies=EEG_POLICIES,
        entropy_meta={
            "metric": "spectral_entropy",
            "backend": "numpy",
            "degraded_mode": True,
            "reason": "permutation_entropy_unavailable",
        },
    )
    e_e = signature["coordinates"]["e_e"]
    assert e_e["source"] == "fallback"
    assert e_e["fallback_feature"] == "spectral_entropy"
    assert e_e["semantic_equivalence"] == "false"
    assert signature["axes_3d"]["e"]["source"] == "mixed"


def test_degraded_permutation_entropy_does_not_invent_spectral() -> None:
    signature = build_support_signature(
        modality="eeg",
        metric_policies=EEG_POLICIES,
        entropy_meta={
            "metric": "permutation_entropy",
            "backend": "numpy",
            "degraded_mode": True,
            "reason": "empty_or_non_finite_signal",
        },
    )
    e_e = signature["coordinates"]["e_e"]
    assert e_e["source"] == "direct"
    assert e_e["fallback_feature"] == "none"
    assert e_e["semantic_equivalence"] == "not_applicable"


def test_e_m_fallback_is_not_semantically_equivalent() -> None:
    signature = build_support_signature(
        modality="eeg",
        metric_policies=EEG_POLICIES,
        actual_metrics={"e_m": "eog_blink_rate"},
        entropy_meta={"metric": "permutation_entropy", "degraded_mode": False},
    )
    e_m = signature["coordinates"]["e_m"]
    assert e_m["source"] == "fallback"
    assert e_m["fallback_feature"] == "eog_blink_rate"
    assert e_m["semantic_equivalence"] == "false"
    assert signature["axes_3d"]["e"]["source"] == "mixed"


def test_eeg_capability_matrix_row() -> None:
    signature = build_support_signature(modality="eeg", metric_policies=EEG_POLICIES)
    cap = signature["capability"]
    assert cap["chart_3d"] == "yes"
    assert cap["chart_9d"] == "yes"
    assert cap["mnj_3d"] == "yes"
    assert cap["mnj_9d"] == "conditional"
    assert cap["diffusion"] == "overlay_only"
    assert cap["spread"] == "gated"
    assert cap["destination"] == "no_generic_ingest"
    assert cap["resilience"] == "perturbational_only"
    validate_write_support_signature(signature)


def test_fmri_mnj_is_limited() -> None:
    signature = build_support_signature(modality="fmri")
    assert signature["capability"]["mnj_3d"] == "limited"
    assert signature["capability"]["mnj_9d"] == "limited"
    assert signature["capability"]["chart_3d"] == "yes"
    assert signature["capability"]["spread"] == "gated"


def test_ieeg_and_meg_capability_rows() -> None:
    ieeg = build_support_signature(modality="ieeg")
    assert ieeg["capability"]["mnj_9d"] == "yes"
    assert ieeg["capability"]["chart_3d"] == "yes"
    meg = build_support_signature(modality="meg")
    assert meg["capability"]["mnj_9d"] == "conditional"
    assert meg["capability"]["mnj_3d"] == "yes"
    ephys = build_support_signature(modality="ephys")
    assert ephys["modality"] == "ieeg"
    assert ephys["capability"]["mnj_9d"] == "yes"
    nwb = build_support_signature(modality="nwb")
    assert nwb["modality"] == "eeg"
    assert nwb["capability"]["chart_3d"] == "yes"


def test_unknown_modality_does_not_assess_charts() -> None:
    signature = build_support_signature(modality="nwb_rodent")
    assert signature["modality"] == "unknown"
    assert signature["capability"]["chart_3d"] == "not_assessed"
    assert signature["capability"]["mnj_9d"] == "not_assessed"
    assert signature["capability"]["spread"] == "gated"
    assert signature["capability"]["resilience"] == "perturbational_only"
    assert signature["capability"]["destination"] == "no_generic_ingest"
    assert signature["capability"]["diffusion"] == "overlay_only"


def test_legacy_reader_never_infers_chart_or_spread_from_presence(tmp_path: Path) -> None:
    path = tmp_path / "legacy_no_signature.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("mnps_3d", data=np.zeros((3, 3), dtype=np.float32))
        handle.create_dataset("mnps_3d_dot", data=np.zeros((3, 3), dtype=np.float32))
    with h5py.File(path, "r") as handle:
        read = read_support_signature(handle)
        assert "mnps_3d" in handle
        assert "support_signature" not in handle
    assert read["signature_origin"] == "legacy_absent"
    assert read["capability"]["chart_3d"] == "not_recorded"
    assert read["capability"]["chart_3d"] != "yes"
    assert read["capability"]["spread"] == "not_recorded"
    assert read["capability"]["spread"] != "gated"
    assert read["capability"]["resilience"] == "not_recorded"
    for name in mnps_9d_CANONICAL_ORDER:
        assert read["coordinates"][name]["source"] == "not_recorded"


def test_canonical_roundtrip_does_not_consult_geometry_contract(tmp_path: Path) -> None:
    signature = build_support_signature(
        modality="eeg",
        metric_policies=EEG_POLICIES,
        actual_metrics={"e_m": "eeg_highfreq_power_30_45"},
        entropy_meta={"metric": "permutation_entropy", "degraded_mode": False},
    )
    out = write_h5(tmp_path / "with_signature.h5", "sig", _payload(support_signature=signature))
    with h5py.File(out, "r") as handle:
        assert handle["support_signature/v1"].attrs["_schema_version"] == SUPPORT_SIGNATURE_SCHEMA_VERSION
        read = read_support_signature(handle)
        assert "_schema_version" not in handle["support_signature"].attrs
        assert "geometry_contract" not in handle["support_signature"]
        assert "geometry_contract" not in handle["support_signature/v1"]
    assert read["signature_origin"] == "canonical"
    assert read["capability"]["chart_3d"] == "yes"
    assert read["coordinates"]["e_m"]["source"] == "fallback"
    assert read["coordinates"]["e_m"]["fallback_feature"] == "eeg_highfreq_power_30_45"
    assert read["coordinates"]["e_m"]["semantic_equivalence"] == "false"


def test_closed_vocab_rejects_unknown_capability_token() -> None:
    with pytest.raises(ValueError, match="Unsupported capability token"):
        validate_write_capability(
            {
                **{key: "yes" for key in CAPABILITY_KEYS},
                "spread": "open",
            }
        )
    with pytest.raises(ValueError, match="Unsupported capability token"):
        validate_write_support_signature(
            {
                **build_support_signature(modality="eeg"),
                "capability": {
                    **build_support_signature(modality="eeg")["capability"],
                    "chart_3d": "licensed",
                },
            }
        )


def test_empty_signature_omits_hdf5_group(tmp_path: Path) -> None:
    out = write_h5(tmp_path / "no_signature.h5", "plain", _payload())
    with h5py.File(out, "r") as handle:
        assert "support_signature" not in handle
        assert "mnps_3d" in handle


def test_support_signature_does_not_change_mnps_or_jacobian(tmp_path: Path) -> None:
    x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]], dtype=np.float32)
    j_hat = np.array([[[1.0, 0.1, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]]], dtype=np.float32)
    with_sig = write_h5(
        tmp_path / "with.h5",
        "with",
        _payload(
            support_signature=build_support_signature(modality="eeg", metric_policies=EEG_POLICIES),
            x=x,
            jacobian=j_hat,
        ),
    )
    without_sig = write_h5(
        tmp_path / "without.h5",
        "without",
        _payload(x=x, jacobian=j_hat),
    )
    with h5py.File(with_sig, "r") as with_handle, h5py.File(without_sig, "r") as without_handle:
        for path in ("mnps_3d", "mnps_3d_dot", "jacobian/J_hat"):
            assert np.array_equal(with_handle[path][:], without_handle[path][:])
        assert "support_signature" in with_handle
        assert "support_signature" not in without_handle


def test_geometry_contract_keys_unchanged() -> None:
    names = list(mnps_9d_CANONICAL_ORDER)
    result = compute_standard_geometry_contract(
        x=np.zeros((5, 3), dtype=float),
        coords_9d=np.zeros((5, 9), dtype=float),
        coords_9d_names=names,
    )
    assert set(result) == GEOMETRY_CONTRACT_TOP_KEYS
    assert result["policy_version"] == "standard_invalidity_v1"
    import mndm.support_signature as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "compute_standard_geometry_contract" not in source
    assert "direct_support" not in source


def test_actual_metrics_from_existing_frame_columns() -> None:
    import pandas as pd

    from mndm.pipeline.summary import _actual_metrics_from_existing_metadata

    constructs = {"e_e": {"actual_metric_used": "permutation_entropy"}}
    meg = pd.DataFrame({"meg_entropy_metric": ["spectral_entropy", "spectral_entropy"]})
    actual = _actual_metrics_from_existing_metadata(meg, constructs)
    assert actual["e_e"] == "spectral_entropy"
    eeg = pd.DataFrame(
        {
            "eeg_entropy_metric": ["permutation_entropy"],
            "embodied_arousal_proxy_source": ["eog_blink_rate"],
        }
    )
    actual = _actual_metrics_from_existing_metadata(eeg, constructs)
    assert actual["e_e"] == "permutation_entropy"
    assert actual["e_m"] == "eog_blink_rate"
    empty = pd.DataFrame({"other": [1]})
    actual = _actual_metrics_from_existing_metadata(empty, constructs)
    assert actual["e_e"] == "permutation_entropy"
    assert "e_m" not in actual
