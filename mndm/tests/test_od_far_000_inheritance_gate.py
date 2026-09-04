"""Tests for the FAR-000 inheritance and provenance gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from od_far_000_inheritance_gate import (  # noqa: E402
    O3_AMENDED_DEV_SEEDS,
    O3_HELD_OUT_SEEDS,
    _audit_o3_inheritance,
    _default_config_semantics,
    _fixture_semantics,
    run_gate,
)


def _write_o3_fixture(root: Path, *, consistent: bool) -> None:
    effect_path = root / "results/orthogonal_dynamics_stage_o3_gate"
    prereg_path = root / "project/orthagonal_dynamics"
    dev_path = root / "results/orthogonal_dynamics_stage_o3_dev_calibration"
    effect_path.mkdir(parents=True)
    prereg_path.mkdir(parents=True)
    dev_path.mkdir(parents=True)
    (effect_path / "effect_summary.json").write_text(
        json.dumps(
            {
                "preregistration": (
                    "project/orthagonal_dynamics/"
                    "017_stage_o3_preregistration.md"
                ),
                "held_out_seeds": O3_HELD_OUT_SEEDS,
                "o3_a": {"status": "pass"},
                "o3_b": {"status": "pass"},
            }
        ),
        encoding="utf-8",
    )
    (prereg_path / "017_stage_o3_preregistration.md").write_text(
        "DEV 7300..7319; held-out seeds frozen.\n",
        encoding="utf-8",
    )
    (prereg_path / "015_o1_o3_program_archive.md").write_text(
        (
            "## O3 status\n**PASS, frozen.** O3 status is pass.\n"
            if consistent
            else "## O3 status\n**Not started; blocked.** O3 status is blocked.\n"
        ),
        encoding="utf-8",
    )
    (dev_path / "dev.json").write_text(
        json.dumps(
            {
                "dev_seeds": (
                    O3_AMENDED_DEV_SEEDS
                    if consistent
                    else list(range(7000, 7020))
                ),
                "held_out_reserved": O3_HELD_OUT_SEEDS,
            }
        ),
        encoding="utf-8",
    )


def test_fixture_semantics_freezes_explicit_amplitudes_and_direction() -> None:
    semantics = _fixture_semantics(
        Path(__file__).resolve().parents[2],
    )

    assert semantics["status"] == "PASS"
    assert semantics["amplitudes"] == [0.0, 1.0, 2.0]
    assert semantics["return_fractions"] == pytest.approx([1.0, 0.6, 0.3])
    assert semantics["r50_discrete"] == 2.0
    assert semantics["direction_semantics"] == "positive_x"
    assert semantics["non_return_is_escape"] is True


def test_default_config_remains_fail_closed() -> None:
    result = _default_config_semantics(Path(__file__).resolve().parents[2])

    assert result["status"] == "PASS"
    assert all(result["checks"].values())


def test_o3_archive_and_seed_contradictions_block_inheritance(tmp_path: Path) -> None:
    _write_o3_fixture(tmp_path, consistent=False)

    result = _audit_o3_inheritance(tmp_path)

    assert result["status"] == "BLOCKED"
    assert "o3_archive_contradiction" in result["reasons"]
    assert "o3_dev_seed_provenance_mismatch" in result["reasons"]


def test_consistent_o3_provenance_can_pass_in_isolated_fixture(
    tmp_path: Path,
) -> None:
    _write_o3_fixture(tmp_path, consistent=True)

    result = _audit_o3_inheritance(tmp_path)

    assert result["status"] == "PASS"
    assert result["reasons"] == []


def test_run_gate_blocks_far_001_without_empirical_reads(tmp_path: Path) -> None:
    _write_o3_fixture(tmp_path / "simulator", consistent=False)
    output = tmp_path / "result.json"

    result = run_gate(
        output,
        simulator_root=tmp_path / "simulator",
        test_runner=lambda _repo_root: {
            "status": "PASS",
            "returncode": 0,
            "command": ["synthetic-test-runner"],
        },
    )

    assert result["status"] == "BLOCKED"
    assert result["decision"]["far_001_authorized"] is False
    assert result["fail_closed_assertions"]["empirical_dataset_read"] is False
    assert result["fail_closed_assertions"]["far_001_source_scout_run"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "BLOCKED"


def test_run_gate_authorizes_far_001_only_for_consistent_inheritance(
    tmp_path: Path,
) -> None:
    _write_o3_fixture(tmp_path / "simulator", consistent=True)

    result = run_gate(
        tmp_path / "result.json",
        simulator_root=tmp_path / "simulator",
        test_runner=lambda _repo_root: {"status": "PASS"},
    )

    assert result["status"] == "PASS"
    assert result["decision"]["far_001_authorized"] is True


def test_translation_failure_is_not_misclassified_as_provenance_block(
    tmp_path: Path,
) -> None:
    _write_o3_fixture(tmp_path / "simulator", consistent=True)

    result = run_gate(
        tmp_path / "result.json",
        simulator_root=tmp_path / "simulator",
        test_runner=lambda _repo_root: {"status": "FAIL"},
    )

    assert result["status"] == "FAIL"
    assert result["decision"]["far_001_authorized"] is False
    assert result["decision"]["reasons"] == [
        "od_tq3_translation_replay_not_pass"
    ]


def test_run_gate_refuses_overwrite(tmp_path: Path) -> None:
    _write_o3_fixture(tmp_path / "simulator", consistent=True)
    output = tmp_path / "result.json"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing_to_overwrite"):
        run_gate(
            output,
            simulator_root=tmp_path / "simulator",
            test_runner=lambda _repo_root: {"status": "PASS"},
        )
