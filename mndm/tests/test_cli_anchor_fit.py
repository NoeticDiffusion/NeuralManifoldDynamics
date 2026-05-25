"""Tests for one-shot anchor CLI flags."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm import cli


def test_build_parser_accepts_fit_anchor_for_summarize():
    """Summarize parser should accept one-shot anchor flags."""
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "summarize",
            "--dataset",
            "ds001",
            "--fit-anchor",
            "--anchor-id",
            "unit-anchor",
            "--anchor-scale-method",
            "mad",
            "--anchor-min-subjects",
            "5",
        ]
    )

    assert args.fit_anchor is True
    assert args.anchor_id == "unit-anchor"
    assert args.anchor_scale_method == "mad"
    assert args.anchor_min_subjects == 5


def test_build_parser_accepts_fit_anchor_for_all():
    """All parser should expose the same one-shot anchor flags."""
    parser = cli.build_parser()
    args = parser.parse_args(["all", "--dataset", "ds001", "--fit-anchor"])

    assert args.command == "all"
    assert args.fit_anchor is True
