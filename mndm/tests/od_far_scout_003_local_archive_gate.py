"""Run the FAR-SCOUT-003 local archive perturbation sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from mndm.dynamical_families.far_scout_003_local_archive import (  # noqa: E402
    PROTOCOL_ID,
    run_inventory,
    sha256_file,
)


PROTOCOL_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/007_far_scout_003_local_archive_prereg.md"
)
DEFAULT_JSON_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_scout_003/"
    "far_scout_003.json"
)
DEFAULT_REPORT_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_scout_003/"
    "far_scout_003.md"
)
DEFAULT_SCOUT002_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_scout_002/"
    "far_scout_002_reconciled_20260821.json"
)
DEFAULT_FAR003A_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_003a/"
    "far_003a.json"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _certificate(path: Path, *, protocol_id: str, status: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing_entry_certificate:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != protocol_id:
        raise ValueError(f"entry_protocol_mismatch:{path}")
    observed_status = payload.get(
        "gate_status",
        payload.get("global_status", payload.get("status")),
    )
    if observed_status != status:
        raise ValueError(f"entry_status_mismatch:{path}:{observed_status}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "protocol_id": protocol_id,
        "status": observed_status,
    }


def _require_far003a(path: Path) -> dict[str, Any]:
    entry = _certificate(path, protocol_id="FAR-003A", status="METHOD_LIMITED")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("far_003b_authorized") is not False:
        raise ValueError("far_003b_must_remain_closed")
    return entry


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# FAR-SCOUT-003 local archive perturbation sweep",
        "",
        "## Gate status",
        "",
        "```text",
        f"status: {payload.get('gate_status')}",
        f"reason: {payload.get('gate_reason')}",
        f"scope complete: {payload.get('scope_complete')}",
        "metadata only: true",
        "signal payloads opened: false",
        "NMD outputs opened: false",
        "downloads performed: false",
        "```",
        "",
        "## Candidate decisions",
        "",
    ]
    for candidate in payload.get("datasets", []):
        scores = candidate.get("scores_P_A_T_R_M", {})
        lines.extend(
            [
                f"### {candidate.get('dataset_id')}",
                "",
                f"- Classification: `{candidate.get('classification')}`",
                f"- Pass 1: `{candidate.get('pass1', {}).get('classification')}`",
                f"- Stage 2: `{candidate.get('stage2_status')}`",
                f"- Reason: {candidate.get('classification_reason', candidate.get('reason'))}",
                f"- Source status: `{candidate.get('source_status')}`",
                f"- Score P/A/T/R/M: `{scores}`",
                f"- Time-base rank: `{candidate.get('timebase_rank')}`",
                f"- Time-base blocked: `{candidate.get('timebase_blocked')}`",
                f"- Hard-blocked: `{candidate.get('hard_blocked')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Claim boundary",
            "",
            payload.get("claim_boundary", ""),
            "",
        ]
    )
    return "\n".join(lines)


def run_gate(
    *,
    repo_root: Path,
    output_json: Path,
    output_report: Path,
    scout002_certificate: Path | None = None,
    far003a_certificate: Path | None = None,
    subsystems_root: Path | None = None,
    source_roots: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    if output_json.exists() or output_report.exists():
        raise FileExistsError("refusing_to_overwrite_far_scout_003_archive")
    protocol_path = repo_root / PROTOCOL_RELATIVE
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing_protocol:{protocol_path}")
    scout002_certificate = scout002_certificate or (
        repo_root / DEFAULT_SCOUT002_RELATIVE
    )
    far003a_certificate = far003a_certificate or (
        repo_root / DEFAULT_FAR003A_RELATIVE
    )
    entry = {
        "far_scout_002": _certificate(
            scout002_certificate,
            protocol_id="FAR-SCOUT-002",
            status="NOT_TESTABLE",
        ),
        "far_003a": _require_far003a(far003a_certificate),
    }
    payload = run_inventory(
        repo_root=repo_root,
        protocol_path=protocol_path,
        source_roots=source_roots,
        subsystems_root=subsystems_root,
    )
    payload["entry_criteria"] = entry
    _write_json(payload, output_json)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_render_report(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=_repo_root() / DEFAULT_JSON_RELATIVE,
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=_repo_root() / DEFAULT_REPORT_RELATIVE,
    )
    parser.add_argument(
        "--scout002-certificate",
        type=Path,
        default=_repo_root() / DEFAULT_SCOUT002_RELATIVE,
    )
    parser.add_argument(
        "--far003a-certificate",
        type=Path,
        default=_repo_root() / DEFAULT_FAR003A_RELATIVE,
    )
    parser.add_argument(
        "--subsystems-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)
    payload = run_gate(
        repo_root=_repo_root(),
        output_json=args.output_json,
        output_report=args.output_report,
        scout002_certificate=args.scout002_certificate,
        far003a_certificate=args.far003a_certificate,
        subsystems_root=args.subsystems_root,
    )
    print(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "gate_status": payload["gate_status"],
                "gate_reason": payload["gate_reason"],
                "classification_counts": payload["classification_counts"],
                "output_json": str(args.output_json),
                "output_report": str(args.output_report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
