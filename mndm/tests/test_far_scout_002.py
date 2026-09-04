"""Synthetic tests for FAR-SCOUT-002."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.far_scout_002 import (  # noqa: E402
    TIME_COMPATIBLE_CURVE,
    TIME_INCOMPATIBLE,
    _audit_dandi_000458_time_compatibility,
    _nmd_window_sec,
    _select_root,
    run_inventory,
)
from od_far_scout_002_gate import run_gate  # noqa: E402


h5py = pytest.importorskip("h5py")


def _write_nwb_fixture(
    path: Path,
    *,
    separation_sec: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        trials = handle.create_group("intervals").create_group("trials")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        trials.create_dataset(
            "behavioral_epoch",
            data=["awake", "awake", "awake", "awake"],
            dtype=string_dtype,
        )
        current = trials.create_dataset(
            "estim_current",
            data=[10.0, 20.0, 30.0, 10.0],
        )
        current.attrs["description"] = "electrical current in μA"
        trials.create_dataset(
            "is_valid",
            data=[True, True, True, True],
        )
        trials.create_dataset(
            "start_time",
            data=[
                1.0,
                1.0 + separation_sec,
                1.0 + 2 * separation_sec,
                1.0 + 3 * separation_sec,
            ],
        )
        trials.create_dataset(
            "stop_time",
            data=[
                1.001,
                1.0 + separation_sec + 0.001,
                1.0 + 2 * separation_sec + 0.001,
                1.0 + 3 * separation_sec + 0.001,
            ],
        )
        trials.create_dataset(
            "stimulus_type",
            data=["electrical", "electrical", "electrical", "electrical"],
            dtype=string_dtype,
        )


def _write_repo_scaffold(tmp_path: Path) -> tuple[Path, Path]:
    config_path = (
        tmp_path
        / "mndm/config/sources/dandi/config_ingest_dandi_000458.yaml"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "version: 2.0\nmnps:\n  window_sec: 4.0\n",
        encoding="utf-8",
    )
    protocol = (
        tmp_path
        / "project/orthagonal_axis/orthagonal_dynamics/"
        "finite-amplitude_resilience/006_far_scout_002_time_compatible_prereg.md"
    )
    protocol.parent.mkdir(parents=True)
    protocol.write_text("FAR-SCOUT-002 fixture\n", encoding="utf-8")
    return config_path, protocol


def _write_entry_certificates(tmp_path: Path) -> tuple[Path, Path]:
    far002 = tmp_path / "far002.json"
    far002.write_text(
        json.dumps({"protocol_id": "FAR-002", "global_status": "PASS"}),
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
    return far002, far003a


def test_matching_config_resolves_rodent_window(tmp_path: Path) -> None:
    config_path, _protocol = _write_repo_scaffold(tmp_path)
    assert _nmd_window_sec(config_path) == 4.0


def test_window_resolution_uses_mnps_block_not_feature_window(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "features:\n"
        "  window_sec: 12.0\n"
        "mnps:\n"
        "  window_sec: 4.0\n",
        encoding="utf-8",
    )
    assert _nmd_window_sec(config_path) == 4.0


def test_empty_source_stub_is_not_selected(tmp_path: Path) -> None:
    empty_stub = tmp_path / "empty-stub"
    empty_stub.mkdir()
    live_root = tmp_path / "live"
    _write_nwb_fixture(
        live_root / "sub-001" / "sub-001_ses-1.nwb",
        separation_sec=9.0,
    )
    assert _select_root(
        dataset_id="dandi_000458",
        roots=[empty_stub, live_root],
        override=None,
    ) == live_root


def test_000458_like_source_is_time_incompatible(tmp_path: Path) -> None:
    root = tmp_path / "000458"
    for subject in ("sub-001", "sub-002"):
        _write_nwb_fixture(
            root / subject / f"{subject}_ses-1.nwb",
            separation_sec=3.0,
        )
    config_path, _protocol = _write_repo_scaffold(tmp_path)

    result = _audit_dandi_000458_time_compatibility(root, config_path)

    assert result["classification"] == TIME_INCOMPATIBLE
    assert result["isolated_post_horizon_sec"]["max"] < 4.0
    assert result["nmd_contract"]["nmd_window_sec"] == 4.0
    assert result["audit_scope"]["forbidden_paths_not_opened"]


def test_long_iti_source_is_time_compatible_curve(tmp_path: Path) -> None:
    root = tmp_path / "000458"
    for subject in ("sub-001", "sub-002"):
        _write_nwb_fixture(
            root / subject / f"{subject}_ses-1.nwb",
            separation_sec=9.0,
        )
    config_path, protocol = _write_repo_scaffold(tmp_path)

    result = run_inventory(
        repo_root=tmp_path,
        source_roots={"dandi_000458": root},
        protocol_path=protocol,
    )

    dandi = next(
        candidate
        for candidate in result["datasets"]
        if candidate["dataset_id"] == "dandi_000458"
    )
    assert dandi["classification"] == TIME_COMPATIBLE_CURVE
    assert result["gate_status"] == "PASS"
    assert dandi["supported_rho_levels_at_hard_minimum"] == [10.0, 20.0, 30.0]


def test_frozen_far003a_horizon_blocks_current_nmd_candidate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "000458"
    for subject in ("sub-001", "sub-002"):
        _write_nwb_fixture(
            root / subject / f"{subject}_ses-1.nwb",
            separation_sec=9.0,
        )
    config_path, _protocol = _write_repo_scaffold(tmp_path)

    result = _audit_dandi_000458_time_compatibility(
        root,
        config_path,
        frozen_far003a_horizon_sec=2.0,
    )

    assert result["classification"] == TIME_INCOMPATIBLE
    assert "FAR-003A frozen native support horizon" in result["classification_reason"]


def test_gate_requires_entries_and_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "000458"
    _write_nwb_fixture(
        root / "sub-001" / "sub-001_ses-1.nwb",
        separation_sec=9.0,
    )
    _config_path, protocol = _write_repo_scaffold(tmp_path)
    far002, far003a = _write_entry_certificates(tmp_path)
    output_json = tmp_path / "result.json"
    output_report = tmp_path / "result.md"

    result = run_gate(
        repo_root=tmp_path,
        output_json=output_json,
        output_report=output_report,
        source_roots={"dandi_000458": root},
        far002_certificate=far002,
        far003a_certificate=far003a,
    )

    assert result["gate_status"] == "NOT_TESTABLE"
    with pytest.raises(FileExistsError, match="refusing_to_overwrite"):
        run_gate(
            repo_root=tmp_path,
            output_json=output_json,
            output_report=tmp_path / "new.md",
            source_roots={"dandi_000458": root},
            far002_certificate=far002,
            far003a_certificate=far003a,
        )
