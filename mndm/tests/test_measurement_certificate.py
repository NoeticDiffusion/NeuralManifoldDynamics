"""Round-2 three-way measurement certificate write/read contracts."""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from core.io.h5_writer import write_h5
from mndm.dynamics.finite_time_response import compute_finite_time_response
from mndm.dynamics.jacobian_metrics import compute_jacobian_metrics, flatten_metric_result
from mndm.measurement_certificate import (
    attach_certificate,
    read_measurement_certificate,
    validate_write_certificate,
)
from mndm.schema import MNPSPayload


def test_attach_certificate_rejects_unknown_write_tokens() -> None:
    with pytest.raises(ValueError, match="measurement_validity"):
        validate_write_certificate(
            measurement_validity="not_recorded",
            claim_status="no_biological_claim",
        )
    with pytest.raises(ValueError, match="claim_status"):
        validate_write_certificate(
            measurement_validity="not_assessed",
            claim_status="ndt_licensed",
        )
    with pytest.raises(ValueError, match="claim_status"):
        validate_write_certificate(
            measurement_validity="not_assessed",
            claim_status="not_recorded",
        )


def test_attach_certificate_does_not_remap_extra_computation_status() -> None:
    out = attach_certificate({"computation_status": "no_crossfit_valid_transitions"})
    assert out["computation_status"] == "no_crossfit_valid_transitions"
    assert out["measurement_validity"] == "not_applicable"
    assert out["claim_status"] == "no_biological_claim"


def test_attach_certificate_never_writes_ndt_licensed() -> None:
    out = attach_certificate({"computation_status": "computed"})
    assert out["measurement_validity"] == "not_assessed"
    assert out["claim_status"] == "no_biological_claim"
    assert "ndt_licensed" not in out.values()


def test_translation_qualified_requires_computed() -> None:
    with pytest.raises(ValueError, match="translation_qualified"):
        attach_certificate(
            {"computation_status": "not_testable"},
            translation_qualified=True,
        )


def test_legacy_reader_never_infers_translation_qualified_or_claim(tmp_path: Path) -> None:
    path = tmp_path / "legacy_absent.h5"
    with h5py.File(path, "w") as handle:
        group = handle.require_group("legacy")
        series = group.require_group("series")
        series.create_dataset("spectral_abscissa", data=np.array([1.0], dtype=np.float32))
        provenance = group.require_group("provenance")
        provenance.create_dataset("qualification_id", data=np.bytes_("OD-TQ1-test-certificate"))
        provenance.create_dataset("validation_level", data=np.bytes_("model_derived"))
    with h5py.File(path, "r") as handle:
        certificate = read_measurement_certificate(handle["legacy"])
    assert certificate["computation_status"] == "not_recorded"
    assert certificate["measurement_validity"] == "not_recorded"
    assert certificate["claim_status"] == "not_recorded"
    assert certificate["certificate_origin"] == "legacy_absent"
    assert certificate["measurement_validity"] != "translation_qualified"
    assert certificate["claim_status"] != "no_biological_claim"


def test_legacy_reader_promotes_provenance_claim_only(tmp_path: Path) -> None:
    path = tmp_path / "legacy_claim.h5"
    with h5py.File(path, "w") as handle:
        group = handle.require_group("family")
        group.create_dataset("computation_status", data=np.bytes_("computed"))
        provenance = group.require_group("provenance")
        provenance.create_dataset("claim_status", data=np.bytes_("no_biological_claim"))
        provenance.create_dataset("qualification_id", data=np.bytes_("OD-TQ1-test-certificate"))
        provenance.create_dataset("validation_level", data=np.bytes_("mndm_translation_validated"))
    with h5py.File(path, "r") as handle:
        certificate = read_measurement_certificate(handle["family"])
    assert certificate["computation_status"] == "computed"
    assert certificate["measurement_validity"] == "not_recorded"
    assert certificate["claim_status"] == "no_biological_claim"
    assert certificate["certificate_origin"] == "legacy_promoted_provenance"
    assert certificate["measurement_validity"] != "translation_qualified"


def test_canonical_reader_copies_top_level_fields(tmp_path: Path) -> None:
    path = tmp_path / "canonical.h5"
    with h5py.File(path, "w") as handle:
        group = handle.require_group("family")
        group.create_dataset("computation_status", data=np.bytes_("computed"))
        group.create_dataset("measurement_validity", data=np.bytes_("translation_qualified"))
        group.create_dataset("claim_status", data=np.bytes_("no_biological_claim"))
    with h5py.File(path, "r") as handle:
        certificate = read_measurement_certificate(handle["family"])
    assert certificate == {
        "computation_status": "computed",
        "measurement_validity": "translation_qualified",
        "claim_status": "no_biological_claim",
        "certificate_origin": "canonical",
    }


def test_jacobian_metrics_certificate_is_not_assessed_when_finite() -> None:
    result = compute_jacobian_metrics(np.repeat(np.eye(3)[None], 2, axis=0))
    assert result["computation_status"] == "computed"
    assert result["measurement_validity"] == "not_assessed"
    assert result["claim_status"] == "no_biological_claim"
    flattened = flatten_metric_result(result)
    assert flattened["computation_status"] == "computed"
    assert flattened["measurement_validity"] == "not_assessed"
    assert flattened["claim_status"] == "no_biological_claim"


def test_jacobian_metrics_insufficient_support_is_not_applicable() -> None:
    result = compute_jacobian_metrics(None)
    assert result["computation_status"] == "insufficient_support"
    assert result["measurement_validity"] == "not_applicable"
    assert result["claim_status"] == "no_biological_claim"


def test_writer_exports_jacobian_metrics_certificate(tmp_path: Path) -> None:
    j_hat = np.repeat(np.eye(3)[None], 1, axis=0)
    payload = MNPSPayload(
        time=np.array([0.0, 1.0]),
        x=np.zeros((2, 3)),
        x_dot=np.zeros((2, 3)),
        jacobian=j_hat,
        jacobian_derived_metrics=compute_jacobian_metrics(j_hat),
    )
    out = write_h5(tmp_path / "metrics-certificate.h5", "test", payload)
    with h5py.File(out, "r") as handle:
        group = handle["/jacobian/derived_metrics/v1"]
        assert group["computation_status"][()].decode() == "computed"
        assert group["measurement_validity"][()].decode() == "not_assessed"
        assert group["claim_status"][()].decode() == "no_biological_claim"
        assert np.array_equal(handle["jacobian/J_hat"][:], j_hat)


def test_writer_exports_finite_time_response_certificate(tmp_path: Path) -> None:
    result = compute_finite_time_response(
        np.repeat(np.array([[[-1.0]]]), 3, axis=0),
        horizon_steps=[1],
        time=np.array([0.0, 1.0, 2.0]),
        validation_level="model_derived",
    )
    payload = MNPSPayload(
        time=np.array([0.0, 1.0, 2.0]),
        x=np.zeros((3, 3)),
        x_dot=np.zeros((3, 3)),
        jacobian=np.repeat(np.eye(3)[None], 3, axis=0),
        finite_time_response=result,
    )
    out = write_h5(tmp_path / "ftr-certificate.h5", "test", payload)
    with h5py.File(out, "r") as handle:
        group = handle["/finite_time_response/v1/primary"]
        assert group["computation_status"][()].decode() == "computed"
        assert group["validation_level"][()].decode() == "model_derived"
        assert group["measurement_validity"][()].decode() == "not_assessed"
        assert group["claim_status"][()].decode() == "no_biological_claim"


def test_finite_time_response_keeps_model_derived_and_not_assessed() -> None:
    result = compute_finite_time_response(
        np.repeat(np.array([[[-1.0]]]), 3, axis=0),
        horizon_steps=[1],
        time=np.array([0.0, 1.0, 2.0]),
        validation_level="model_derived",
    )
    assert result["computation_status"] == "computed"
    assert result["validation_level"] == "model_derived"
    assert result["measurement_validity"] == "not_assessed"
    assert result["claim_status"] == "no_biological_claim"
    assert result["measurement_validity"] != "translation_qualified"
