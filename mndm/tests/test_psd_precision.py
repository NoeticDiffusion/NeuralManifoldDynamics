"""Independent precision and serialization checks for covariance projection."""

from io import BytesIO

import numpy as np
import pytest

from mndm.dynamics.validity import project_to_psd


def test_float64_preserves_huge_finite_covariance_scale() -> None:
    covariance = np.diag(np.array([1.0e173, 2.0e173, 4.0e173], dtype=np.float64))

    result, qc = project_to_psd(covariance, output_dtype=np.float64)

    assert result.dtype == np.dtype(np.float64)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, covariance, rtol=1e-12, atol=0.0)
    assert qc["q_output_dtype"] == "float64"
    assert qc["q_psd_post_dtype"] is True
    assert qc["q_min_eigenvalue"] > 0.0
    assert np.isfinite(qc["q_psd_tolerance"])


def test_float64_near_rank_deficient_nine_dimensional_matrix_is_auditable() -> None:
    rng = np.random.default_rng(302)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(9, 9)))
    eigenvalues = 1.0e20 * np.geomspace(1.0, 1.0e-18, 9)
    covariance = orthogonal @ np.diag(eigenvalues) @ orthogonal.T

    result, qc = project_to_psd(covariance, min_eigenvalue=1.0e-8, output_dtype=np.float64)
    post_eigenvalues = np.linalg.eigvalsh(0.5 * (result + result.T))

    assert result.shape == (9, 9)
    assert np.all(np.isfinite(result))
    assert float(post_eigenvalues[0]) >= -qc["q_psd_tolerance"]
    assert qc["q_psd_post_dtype"] is True
    assert qc["q_floor_met_post_dtype"] is True


def test_indefinite_input_gets_absolute_eigenvalue_floor() -> None:
    covariance = np.array([[2.0, -3.0], [-3.0, 2.0]], dtype=np.float64)

    result, qc = project_to_psd(covariance, min_eigenvalue=0.5, output_dtype=np.float64)
    post_eigenvalues = np.linalg.eigvalsh(0.5 * (result + result.T))

    assert qc["q_psd_correction"] is True
    assert float(post_eigenvalues[0]) >= 0.5 - qc["q_psd_tolerance"]
    assert qc["q_requested_min_eigenvalue"] == 0.5
    assert qc["q_floor_met_post_dtype"] is True


def test_default_float32_remains_equivalent_to_explicit_legacy_dtype() -> None:
    covariance = np.array([[2.0, 0.25], [0.25, 1.0]], dtype=np.float64)

    default, default_qc = project_to_psd(covariance)
    explicit, explicit_qc = project_to_psd(covariance, output_dtype=np.float32)

    assert default.dtype == np.dtype(np.float32)
    assert explicit.dtype == np.dtype(np.float32)
    assert np.array_equal(default, explicit)
    assert default_qc["q_output_dtype"] == explicit_qc["q_output_dtype"] == "float32"


def test_precision_keyword_is_compatible_with_float64_opt_in() -> None:
    covariance = np.diag(np.array([1.0e173, 3.0e173], dtype=np.float64))

    result, qc = project_to_psd(covariance, precision="float64")

    assert result.dtype == np.dtype(np.float64)
    assert qc["q_output_dtype"] == "float64"


def test_float64_roundtrip_through_nested_h5_writer_preserves_dtype() -> None:
    h5py = pytest.importorskip("h5py")
    from core.io.h5_writer import _write_nested_mapping_group

    covariance = np.array([[3.0e12, 2.0e5], [2.0e5, 4.0e12]], dtype=np.float64)
    result, qc = project_to_psd(covariance, output_dtype=np.float64)

    with h5py.File(BytesIO(), "w") as h5:
        _write_nested_mapping_group(h5, "w_q", {"covariance": result})
        stored = np.asarray(h5["w_q"]["covariance"])
        assert stored.dtype == np.dtype(np.float64)
        assert np.allclose(stored, result, rtol=0.0, atol=0.0)
        stored_min = float(np.linalg.eigvalsh(0.5 * (stored + stored.T))[0])

    assert abs(stored_min - qc["q_min_eigenvalue"]) <= qc["q_psd_tolerance"]
