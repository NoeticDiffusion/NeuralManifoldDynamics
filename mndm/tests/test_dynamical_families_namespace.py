"""Round-1 dynamical-family namespace, registry, and reader tests."""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from core.io.h5_writer import write_h5
from mndm.dynamical_families import FAMILIES, family_forbids, get_family
from mndm.dynamical_families.io import canonical_group_path, read_measurement_certificate, resolve_family_group
from mndm.pipeline.dynamical_families_export import (
    build_dynamical_families_export,
    reject_legacy_family_config_key,
)
from mndm.schema import MNPSPayload


def test_registry_diffusion_forbids_jacobian_residual_as_diffusion() -> None:
    assert family_forbids("diffusion", "jacobian_derivative_residual_as_diffusion")
    assert family_forbids("diffusion", "mnps_xdot_as_sde_drift")
    assert family_forbids("diffusion", "jacobian_intercept_as_sde_drift")
    assert family_forbids("diffusion", "local_increment_mean_as_sde_drift")
    assert get_family("diffusion")["schema"] == "mndm.diffusion_geometry.v1"
    assert get_family("diffusion")["namespace"] == "/dynamical_families/diffusion/v1"
    assert (
        get_family("diffusion")["legacy_namespace"]
        == "/orthogonal_dynamics/diffusion_geometry/v1"
    )


def test_registry_spread_remains_gate_closed() -> None:
    spread = get_family("spread")
    assert spread["implementation"] == "gate_closed"
    assert spread["gate"] == "F"
    assert spread["default_enabled"] is False
    assert spread["namespace"] is None
    assert spread["out_of_family_namespace"] == "/stochastic_reachability/v1"
    assert "spread" in FAMILIES
    assert canonical_group_path("spread") is None
    missing = resolve_family_group({}, "spread")
    assert missing["group"] is None
    assert missing["path"] is None


def test_write_h5_refuses_non_writable_family_keys(require_real_h5py, tmp_path: Path) -> None:
    from mndm.schema import normalize_payload

    with pytest.raises(ValueError, match="gate_closed"):
        normalize_payload(
            MNPSPayload(
                time=np.array([0.0, 1.0]),
                x=np.zeros((2, 3), dtype=np.float32),
                x_dot=np.zeros((2, 3), dtype=np.float32),
                dynamical_families={"spread": {"schema_version": "mndm.stochastic_reachability.v1"}},
            )
        )
    with pytest.raises(ValueError, match="renamed to 'diffusion'"):
        normalize_payload(
            MNPSPayload(
                time=np.array([0.0, 1.0]),
                x=np.zeros((2, 3), dtype=np.float32),
                x_dot=np.zeros((2, 3), dtype=np.float32),
                dynamical_families={
                    "diffusion_geometry": {"schema_version": "mndm.diffusion_geometry.v1"}
                },
            )
        )


def test_nested_legacy_yaml_family_keys_are_refused() -> None:
    with pytest.raises(ValueError, match="dynamical_families.committor"):
        reject_legacy_family_config_key(
            {"dynamical_families": {"enabled": True, "committor": {"enabled": True}}}
        )
    with pytest.raises(ValueError, match="gate_closed"):
        reject_legacy_family_config_key({"dynamical_families": {"spread": {"enabled": True}}})
    assert family_forbids("spread", "gate_e_proxy_as_process_noise")
    assert family_forbids("diffusion", "jacobian_derivative_residual_as_diffusion")


def test_write_h5_emits_canonical_family_paths_only(require_real_h5py, tmp_path: Path) -> None:
    payload = MNPSPayload(
        time=np.array([0.0, 1.0]),
        x=np.zeros((2, 3), dtype=np.float32),
        x_dot=np.zeros((2, 3), dtype=np.float32),
        dynamical_families={
            "diffusion": {
                "schema_version": "mndm.diffusion_geometry.v1",
                "computation_status": "not_testable",
            },
            "destination": {
                "schema_version": "mndm.committor.v1",
                "computation_status": "not_testable",
            },
            "resilience": {
                "schema_version": "mndm.finite_amplitude_resilience.v1",
                "computation_status": "not_testable",
            },
        },
    )
    output = write_h5(tmp_path / "families.h5", "test", payload)
    with h5py.File(output, "r") as handle:
        assert "/dynamical_families/diffusion/v1" in handle
        assert "/dynamical_families/destination/v1" in handle
        assert "/dynamical_families/resilience/v1" in handle
        assert "orthogonal_dynamics" not in handle
        assert "/orthogonal_dynamics/diffusion_geometry/v1" not in handle


def test_resolve_family_group_reads_legacy_namespace(require_real_h5py, tmp_path: Path) -> None:
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as handle:
        group = handle.require_group("orthogonal_dynamics/diffusion_geometry/v1")
        group.create_dataset(
            "computation_status",
            data=np.bytes_("not_testable"),
        )
    with h5py.File(path, "r") as handle:
        resolved = resolve_family_group(handle, "diffusion")
        assert resolved["namespace_origin"] == "legacy_orthogonal_dynamics"
        assert resolved["path"] == "/orthogonal_dynamics/diffusion_geometry/v1"
        assert resolved["group"] is not None
        status = resolved["group"]["computation_status"][()]
        assert status.decode() == "not_testable"
        certificate = read_measurement_certificate(resolved["group"])
        assert certificate["computation_status"] == "not_testable"
        assert certificate["measurement_validity"] == "not_recorded"
        assert certificate["claim_status"] == "not_recorded"
        assert certificate["certificate_origin"] == "legacy_absent"


def test_legacy_yaml_key_is_refused() -> None:
    with pytest.raises(ValueError, match="renamed to 'dynamical_families'"):
        reject_legacy_family_config_key({"orthogonal_dynamics": {"enabled": True}})
    with pytest.raises(ValueError, match="renamed to 'dynamical_families'"):
        build_dynamical_families_export(
            config={"orthogonal_dynamics": {"enabled": True}},
            state=np.zeros((8, 3), dtype=np.float32),
            time=np.arange(8, dtype=np.float64),
            stage=None,
            segment_id=None,
            coordinate_layer="coords_3d_subject_anchored",
            coordinate_names=["m", "d", "e"],
        )
