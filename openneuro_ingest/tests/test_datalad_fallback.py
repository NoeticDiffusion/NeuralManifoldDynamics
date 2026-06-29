"""Tests for the DataLad OpenNeuro fallback downloader helpers."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


class _DummyDataset:
    def __init__(self, pathobj: Path):
        self.pathobj = pathobj


def test_normalize_subject_ids_adds_sub_prefix_and_deduplicates():
    from datalad_fallback import _normalize_subject_ids

    assert _normalize_subject_ids(["002", "sub-003", "002", ""]) == ["sub-002", "sub-003"]


def test_build_get_plan_subject_subset_includes_core_metadata(tmp_path: Path):
    from datalad_fallback import _build_get_plan

    dataset = _DummyDataset(tmp_path / "ds003645")
    plan = _build_get_plan(
        dataset,
        paths=None,
        subjects=["sub-002", "003"],
        include_derivatives=False,
    )

    assert plan.derivatives_skipped is True
    assert plan.targets == [
        "README",
        "CHANGES",
        "dataset_description.json",
        "participants.tsv",
        "participants.json",
        "sub-002",
        "sub-003",
    ]
