"""Tests for regional MNPS/MNJ computation."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from mndm.pipeline.regional_mnps import (
    compute_block_jacobian_rows,
    RegionalMNPSResult,
    RegionalMNPSSummary,
    compute_all_regional_mnps,
    compute_jacobian_metrics,
    compute_mnps_metrics,
    compute_regional_mnps_for_network,
    summary_to_dataframe_rows,
)


def _fake_mnps(n: int = 120) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(n, 3)).astype(np.float32)


class TestComputeJacobianMetrics:
    def test_metrics_computed(self):
        j_hat = np.random.randn(10, 3, 3).astype(np.float32)
        metrics = compute_jacobian_metrics(j_hat)
        assert "trace_mean" in metrics
        assert "frobenius_mean" in metrics
        assert "rotation_norm_mean" in metrics
        assert "anisotropy_mean" in metrics

    def test_zero_jacobian_has_finite_anisotropy(self):
        metrics = compute_jacobian_metrics(np.zeros((5, 3, 3), dtype=np.float32))
        assert metrics["anisotropy_mean"] == pytest.approx(1.0)


class TestComputeMNPSMetrics:
    def test_metrics_computed(self):
        metrics = compute_mnps_metrics(np.random.rand(20, 3).astype(np.float32))
        assert "m_mean" in metrics
        assert "d_mean" in metrics
        assert "e_mean" in metrics


class TestComputeRegionalMNPSForNetwork:
    def test_valid_network_produces_result(self):
        config = {
            "mnps": {"time_step_sec": 2.0, "derivative": {"method": "central"}},
            "jacobian": {"enabled": True, "super_window": 3},
        }
        result = compute_regional_mnps_for_network(
            network_label="DMN",
            mnps_trajectory=_fake_mnps(120),
            config=config,
            min_length=20,
        )
        assert result.valid
        assert result.mnps.shape[1] == 3
        assert result.jacobian is not None
        assert "trace_mean" in result.metrics

    def test_invalid_when_not_precomputed_trajectory(self):
        result = compute_regional_mnps_for_network(
            network_label="DMN",
            mnps_trajectory=np.random.randn(120),  # 1D legacy input must fail
            config={"mnps": {}, "jacobian": {"enabled": False}},
            min_length=20,
        )
        assert not result.valid
        assert "precomputed MNPS trajectory" in (result.drop_reason or "")


class TestComputeAllRegionalMNPS:
    def test_requires_precomputed_network_mnps(self):
        summary = compute_all_regional_mnps(
            group_ts={"DMN": np.random.randn(200)},
            sfreq=0.5,
            config={},
            subject="sub-001",
        )
        assert summary.n_networks == 0

    def test_multiple_networks(self):
        config = {
            "min_segment_length_tr": 20,
            "networks": [],
            "mnps": {"time_step_sec": 2.0, "derivative": {"method": "central"}},
            "jacobian": {"enabled": True},
        }
        network_mnps = {"DMN": _fake_mnps(120), "FPN": _fake_mnps(120), "SAL": _fake_mnps(120)}
        summary = compute_all_regional_mnps(
            group_ts=None,
            sfreq=None,
            config=config,
            subject="sub-001",
            condition="awake",
            task="rest",
            network_mnps=network_mnps,
        )
        assert summary.n_networks == 3
        assert set(summary.results) == {"DMN", "FPN", "SAL"}


class TestSummaryToDataframeRows:
    def test_rows_generated(self):
        summary = RegionalMNPSSummary(
            subject="sub-001",
            session=None,
            condition="awake",
            task="rest",
            results={
                "DMN": RegionalMNPSResult(
                    network="DMN",
                    mnps=np.zeros((10, 3)),
                    mnps_dot=np.zeros((10, 3)),
                    n_timepoints=10,
                    metrics={"m_mean": 0.5, "trace_mean": 1.0},
                )
            },
            n_networks=1,
        )
        rows = summary_to_dataframe_rows(summary)
        assert len(rows) == 1
        assert rows[0]["network_label"] == "DMN"


class TestBlockJacobiansRemoved:
    def test_block_rows_return_empty(self):
        summary = RegionalMNPSSummary(
            subject="sub-001",
            session=None,
            condition="awake",
            task="rest",
            results={},
        )
        rows = compute_block_jacobian_rows(summary, config={}, include_self=False)
        assert rows == []

