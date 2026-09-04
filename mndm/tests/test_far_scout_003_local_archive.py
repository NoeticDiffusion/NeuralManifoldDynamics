"""Synthetic tests for FAR-SCOUT-003."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.far_scout_003_local_archive import (  # noqa: E402
    PERTURBATION_VALID_NMD_TIMEBASE_INCOMPATIBLE,
    RHO_CANDIDATE,
    SOURCE_UNCERTAIN,
    _audit_spec,
    _passes_hard_block,
    run_inventory,
)
from od_far_scout_003_local_archive_gate import run_gate  # noqa: E402


def _write_protocol(tmp_path: Path) -> Path:
    protocol = (
        tmp_path
        / "project/orthagonal_axis/orthagonal_dynamics/"
        "finite-amplitude_resilience/007_far_scout_003_local_archive_prereg.md"
    )
    protocol.parent.mkdir(parents=True)
    protocol.write_text("FAR-SCOUT-003 fixture\n", encoding="utf-8")
    return protocol


def _write_entry_certificates(tmp_path: Path) -> tuple[Path, Path]:
    scout002 = tmp_path / "scout002.json"
    scout002.write_text(
        json.dumps({"protocol_id": "FAR-SCOUT-002", "gate_status": "NOT_TESTABLE"}),
        encoding="utf-8",
    )
    far003a = tmp_path / "far003a.json"
    far003a.write_text(
        json.dumps(
            {
                "protocol_id": "FAR-003A",
                "global_status": "METHOD_LIMITED",
                "far_003b_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    return scout002, far003a


def _fixture_source_roots(tmp_path: Path) -> dict[str, Path]:
    return {
        dataset_id: tmp_path / dataset_id
        for dataset_id in (
            "dandi_000458",
            "dandi_000690",
            "ibl_bwm_000409",
            "dryad_vpm_s1_dbrv15f3n",
            "dryad_cecchetto_dbrv15f23",
            "dandi_000574",
        )
    }


def test_dryad_power_candidate_is_blocked_without_continuous_signal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dryad"
    root.mkdir()
    (root / "README.md").write_text(
        "LED power 0, 2.5, and 5.0 mW/mm²; repeated trial events; "
        "ITI 300 ms; peri-event arrays only.\n",
        encoding="utf-8",
    )
    result = _audit_spec(
        {
            "dataset_id": "dryad_vpm_s1_dbrv15f3",
            "source_family": "dryad",
            "roots": [root],
            "profile": "dryad_power_no_continuous_signal",
            "nmd_window_sec": 4.0,
        }
    )
    assert result["pass1"]["classification"] == RHO_CANDIDATE
    assert result["classification"] == SOURCE_UNCERTAIN
    assert result["scores_P_A_T_R_M"]["P"] == 1
    assert result["scores_P_A_T_R_M"]["A"] == 0
    assert result["scores_P_A_T_R_M"]["M"] == 0
    assert result["audit_scope"]["signal_payloads_opened"] is False


def test_contrast_tokens_without_physical_scale_remain_uncertain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dandi690"
    root.mkdir()
    (root / "review.md").write_text(
        "Repeated visual stimulus tables include Cntst0 and Cntst1. "
        "No physical contrast scale or unit is frozen here.\n",
        encoding="utf-8",
    )
    result = _audit_spec(
        {
            "dataset_id": "dandi_000690",
            "source_family": "dandi_allen",
            "roots": [root],
            "profile": "dandi_000690_contrast_uncertain",
            "nmd_window_sec": 4.0,
            "evidence_relative_paths": ["review.md"],
        }
    )
    assert result["pass1"]["classification"] == "RHO_CANDIDATE"
    assert result["classification"] == SOURCE_UNCERTAIN
    assert result["pass1"]["units"] is None


def test_carried_forward_incompatible_source_cannot_open_gate(
    tmp_path: Path,
) -> None:
    protocol = _write_protocol(tmp_path)
    local_root = tmp_path / "local-458"
    local_root.mkdir()
    result = run_inventory(
        repo_root=tmp_path,
        protocol_path=protocol,
        source_roots={
            **_fixture_source_roots(tmp_path),
            "dandi_000458": local_root,
        },
    )
    dandi = next(
        candidate
        for candidate in result["datasets"]
        if candidate["dataset_id"] == "dandi_000458"
    )
    assert (
        dandi["classification"]
        == PERTURBATION_VALID_NMD_TIMEBASE_INCOMPATIBLE
    )
    assert result["gate_status"] == "NOT_TESTABLE"


def test_evidence_scope_excludes_payloads_and_outcome_tables(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ibl"
    root.mkdir()
    (root / "README.md").write_text(
        "Graded visual contrast and stimulus times are documented.\n",
        encoding="utf-8",
    )
    (root / "shared_state_truth_known.csv").write_text(
        "mechanism,false_positive_rate\ncommon_state,0.063\n",
        encoding="utf-8",
    )
    for suffix in (".nwb", ".h5", ".mat", ".npy", ".npz", ".dat"):
        (root / f"payload{suffix}").write_bytes(b"not opened")
    result = _audit_spec(
        {
            "dataset_id": "ibl_bwm_000409",
            "source_family": "ibl",
            "roots": [root],
            "profile": "ibl_contrast_timing_uncertain",
            "nmd_window_sec": 4.0,
            "evidence_patterns": ["README.md"],
        }
    )
    records = result["metadata_file_records"]
    assert [record["relative_path"] for record in records] == ["README.md"]
    assert result["metadata_inventory"]["metadata_file_count"] == 1
    assert result["audit_scope"]["outcome_tables_opened"] is False


def test_unavailable_root_is_explicitly_uncertain(tmp_path: Path) -> None:
    result = _audit_spec(
        {
            "dataset_id": "dandi_000690",
            "source_family": "dandi",
            "roots": [tmp_path / "missing"],
            "profile": "dandi_000690_contrast_uncertain",
            "nmd_window_sec": 4.0,
        }
    )
    assert result["source_status"] == "LOCAL_UNAVAILABLE"
    assert result["classification"] == SOURCE_UNCERTAIN
    assert result["hard_blocked"] is True


def test_dandi_000574_is_direction_only_and_does_not_enter_pass_two(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dandi574"
    root.mkdir()
    (root / "README.md").write_text(
        "Sternberg working-memory task with several set sizes.\n",
        encoding="utf-8",
    )
    result = _audit_spec(
        {
            "dataset_id": "dandi_000574",
            "source_family": "dandi",
            "roots": [root],
            "profile": "dandi_000574_no_replacement_qualification",
            "nmd_window_sec": 4.0,
            "evidence_relative_paths": ["README.md"],
        }
    )
    assert result["pass1"]["classification"] == "DIRECTION_ONLY"
    assert result["classification"] == "DIRECTION_ONLY"
    assert result["stage2_status"] == "NOT_ENTERED"


def test_missing_allowlisted_evidence_does_not_fall_back_to_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scope"
    root.mkdir()
    (root / "unscoped.md").write_text(
        "contrast 100% and stimulus onset are mentioned here.\n",
        encoding="utf-8",
    )
    result = _audit_spec(
        {
            "dataset_id": "ibl_bwm_000409",
            "source_family": "ibl",
            "roots": [root],
            "profile": "ibl_contrast_timing_uncertain",
            "nmd_window_sec": 4.0,
            "evidence_relative_paths": ["missing.md"],
        }
    )
    assert result["source_status"] == "LOCAL_METADATA_SCOPE_UNRESOLVED"
    assert result["metadata_inventory"]["metadata_file_count"] == 0
    assert result["classification"] == SOURCE_UNCERTAIN


def test_root_bound_evidence_paths_match_their_declared_root(
    tmp_path: Path,
) -> None:
    cctg = tmp_path / "cctg"
    sdsr = tmp_path / "sdsr"
    cctg.mkdir()
    sdsr.mkdir()
    (cctg / "README.md").write_text("CCTG metadata\n", encoding="utf-8")
    (cctg / "results").mkdir()
    (cctg / "results" / "coverage_summary.json").write_text(
        '{"n_sessions": 149}\n',
        encoding="utf-8",
    )
    (sdsr / "IBL.md").write_text(
        "Signed contrast 6.25% and stimulus times are documented.\n",
        encoding="utf-8",
    )
    (sdsr / "README.md").write_text(
        "This unallowlisted file must not be read.\n",
        encoding="utf-8",
    )
    result = _audit_spec(
        {
            "dataset_id": "ibl_bwm_000409",
            "source_family": "ibl",
            "roots": [cctg, sdsr],
            "profile": "ibl_contrast_timing_uncertain",
            "nmd_window_sec": 4.0,
            "evidence_relative_paths_by_root": [
                ["README.md", "results/coverage_summary.json"],
                ["IBL.md"],
            ],
        }
    )
    assert result["metadata_inventory"]["metadata_file_count"] == 3
    assert {
        record["relative_path"]
        for record in result["metadata_file_records"]
    } == {"README.md", "results/coverage_summary.json", "IBL.md"}
    missing_root = _audit_spec(
        {
            "dataset_id": "ibl_bwm_000409",
            "source_family": "ibl",
            "roots": [cctg],
            "profile": "ibl_contrast_timing_uncertain",
            "nmd_window_sec": 4.0,
            "evidence_relative_paths_by_root": [
                ["README.md", "results/coverage_summary.json"],
                ["IBL.md"],
            ],
        }
    )
    assert missing_root["source_status"] == "LOCAL_METADATA_SCOPE_UNRESOLVED"


def test_hard_block_prevents_zero_t_promotion() -> None:
    assert not _passes_hard_block(
        "PROMOTION_PASS",
        {"P": 4, "A": 4, "T": 0},
    )
    assert _passes_hard_block(
        "PROMOTION_PASS",
        {"P": 4, "A": 4, "T": 1},
    )
    assert not _passes_hard_block(
        "PROMOTION_PASS",
        {"P": 4, "A": 4, "T": 1},
        "TIMEBASE_UNRESOLVED",
    )
    assert not _passes_hard_block(
        "PROMOTION_PASS",
        {"P": 4, "A": 4, "T": 1},
        "TIMEBASE_INCOMPATIBLE",
    )


def test_gate_requires_entries_and_refuses_overwrite(tmp_path: Path) -> None:
    protocol = _write_protocol(tmp_path)
    scout002, far003a = _write_entry_certificates(tmp_path)
    output_json = tmp_path / "result.json"
    output_report = tmp_path / "result.md"

    result = run_gate(
        repo_root=tmp_path,
        output_json=output_json,
        output_report=output_report,
        scout002_certificate=scout002,
        far003a_certificate=far003a,
        source_roots=_fixture_source_roots(tmp_path),
        subsystems_root=tmp_path / "subsystems",
    )

    assert result["gate_status"] == "NOT_TESTABLE"
    assert output_json.is_file()
    assert output_report.is_file()
    with pytest.raises(FileExistsError, match="refusing_to_overwrite"):
        run_gate(
            repo_root=tmp_path,
            output_json=output_json,
            output_report=tmp_path / "new.md",
            scout002_certificate=scout002,
            far003a_certificate=far003a,
            source_roots=_fixture_source_roots(tmp_path),
            subsystems_root=tmp_path / "subsystems",
        )
