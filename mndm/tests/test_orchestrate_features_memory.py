"""Tests for memory-aware worker capping in orchestrate features."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm import orchestrate


def test_cap_workers_by_memory_reduces_parallelism_for_large_edf():
    """Test cap workers by memory reduces parallelism for large edf."""
    tasks = [
        (Path("sub-01_task-rest_eeg.edf"), "sub-01/eeg/sub-01_task-rest_eeg.edf", 150 * 1024 * 1024),
        (Path("sub-02_task-rest_eeg.edf"), "sub-02/eeg/sub-02_task-rest_eeg.edf", 140 * 1024 * 1024),
    ]
    workers, est_gb = orchestrate._cap_workers_by_memory(  # type: ignore[attr-defined]
        requested_workers=6,
        file_tasks=tasks,
        mem_budget_gb=2.0,
    )
    assert 0.4 <= est_gb <= 0.7
    assert workers == 3


def test_cap_workers_by_memory_keeps_requested_when_budget_allows():
    """Test cap workers by memory keeps requested when budget allows."""
    tasks = [
        (Path("sub-01_task-rest_eeg.edf"), "sub-01/eeg/sub-01_task-rest_eeg.edf", 150 * 1024 * 1024),
    ]
    workers, est_gb = orchestrate._cap_workers_by_memory(  # type: ignore[attr-defined]
        requested_workers=6,
        file_tasks=tasks,
        mem_budget_gb=4.0,
    )
    assert 0.4 <= est_gb <= 0.7
    assert workers == 6


def test_estimate_peak_ram_per_file_returns_model_label():
    """Test estimate peak ram per file returns model label."""
    est_gb, model = orchestrate._estimate_peak_ram_per_file(  # type: ignore[attr-defined]
        Path("sub-01_task-rest_eeg.edf"),
        150 * 1024 * 1024,
    )
    assert 0.4 <= est_gb <= 0.7
    assert model == "size_fallback_edf_v2"

