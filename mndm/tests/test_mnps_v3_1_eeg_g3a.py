"""Synthetic checks for the frozen G3a channel-aggregation protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "project"
    / "new_mnps_v.3.1"
    / "run_anphy_epctl01_e1_g3a.py"
)
spec = importlib.util.spec_from_file_location("mnps_v31_e1_g3a_runner", RUNNER_PATH)
assert spec is not None and spec.loader is not None
g3a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g3a)


CANONICAL = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
)
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def test_g3a_subset_uses_frozen_anphy_aliases_and_canonical_order() -> None:
    labels = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
        "C4", "T4", "T5", "P3", "Pz", "P4", "P8", "O1", "O2",
    ] + [f"EXTRA_{index:02d}" for index in range(64)]

    indices = g3a._common_channel_indices(
        labels,
        canonical_labels=CANONICAL,
        aliases=ALIASES,
    )

    assert indices == tuple(range(19))


def test_g3a_subset_fails_closed_for_missing_or_ambiguous_alias() -> None:
    labels = list(CANONICAL) + [f"EXTRA_{index:02d}" for index in range(64)]
    labels.remove("O2")
    with pytest.raises(ValueError, match="labels_missing"):
        g3a._common_channel_indices(
            labels,
            canonical_labels=CANONICAL,
            aliases={},
        )

    duplicate = labels + ["T3"]
    with pytest.raises(ValueError, match="duplicate_or_empty"):
        g3a._common_channel_indices(
            duplicate,
            canonical_labels=CANONICAL,
            aliases=ALIASES,
        )


def test_g3a_paired_statistics_orient_difference_common_minus_e1_83() -> None:
    full = np.asarray([0.2, 0.4, 0.6, 0.8])
    common = np.asarray([0.3, 0.5, 0.7, 0.9])

    result = g3a._paired_statistics(full, common)

    assert result["pair_orientation"] == "common_19_minus_frozen_83"
    assert result["spearman_rank_correlation"] == pytest.approx(1.0)
    assert result["median_absolute_difference"] == pytest.approx(0.1)
    assert result["bland_altman_mean_bias"] == pytest.approx(0.1)
    assert result["p_value"] is None
    assert result["confidence_interval"] is None


def test_g3a_ar1_control_is_deterministic_and_finite() -> None:
    first = g3a._synthetic_channel_matrix(
        "heterogeneous_ar1_stationary_unit_variance",
        rng=np.random.default_rng(310903),
        channel_count=83,
        sample_count=100,
    )
    second = g3a._synthetic_channel_matrix(
        "heterogeneous_ar1_stationary_unit_variance",
        rng=np.random.default_rng(310903),
        channel_count=83,
        sample_count=100,
    )

    assert first.shape == (83, 100)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)


def test_g3a_refuses_existing_output_root(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        g3a._new_output_root(tmp_path)


def test_g3a_reads_frozen_g2_hashes_from_grid_provenance() -> None:
    summary = {
        "grid_provenance": {
            "overlay_sha256": "overlay",
            "source_sha256": "source",
            "positions_sha256": "positions",
        }
    }

    provenance = g3a._g2_grid_provenance(summary)

    assert provenance["overlay_sha256"] == "overlay"
    assert provenance["source_sha256"] == "source"
