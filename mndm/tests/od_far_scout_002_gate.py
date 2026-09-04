"""Run the FAR-SCOUT-002 time-compatible amplitude-source scout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from mndm.dynamical_families.far_scout_002 import (  # noqa: E402
    PROTOCOL_ID,
    run_inventory,
    sha256_file,
)


PROTOCOL_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/006_far_scout_002_time_compatible_prereg.md"
)
DEFAULT_JSON_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_scout_002/"
    "far_scout_002.json"
)
DEFAULT_REPORT_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_scout_002/"
    "far_scout_002.md"
)
DEFAULT_FAR002_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_002/"
    "far_002_reconciled_20260820.json"
)
DEFAULT_FAR003A_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_003a/far_003a.json"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _certificate(path: Path, *, protocol_id: str, status: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing_entry_certificate:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != protocol_id:
        raise ValueError(f"entry_protocol_mismatch:{path}")
    observed_status = payload.get("global_status", payload.get("status"))
    if observed_status != status:
        raise ValueError(f"entry_status_mismatch:{path}:{observed_status}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "protocol_id": protocol_id,
        "status": observed_status,
    }


def _require_far003a(path: Path) -> dict[str, Any]:
    entry = _certificate(
        path,
        protocol_id="FAR-003A",
        status="METHOD_LIMITED",
    )
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
        "# FAR-SCOUT-002 time-compatible amplitude perturbation scout",
        "",
        "## Gate status",
        "",
        "```text",
        f"status: {payload.get('gate_status')}",
        f"reason: {payload.get('gate_reason')}",
        "signal payloads opened: false",
        "NMD outputs opened: false",
        "downloads performed: false",
        "```",
        "",
        "This is a metadata-only source scout. It does not estimate a "
        "resilience outcome.",
        "",
        "## Candidate decisions",
        "",
    ]
    for candidate in payload.get("datasets", []):
        timing = candidate.get("isolated_post_horizon_sec") or {}
        contract = candidate.get("nmd_contract") or {}
        lines.extend(
            [
                f"### {candidate.get('dataset_id')}",
                "",
                f"- Classification: `{candidate.get('classification')}`",
                f"- Reason: {candidate.get('classification_reason')}",
                f"- Source status: `{candidate.get('source_status')}`",
                f"- NMD window: `{contract.get('nmd_window_sec')}` s",
                f"- Maximum isolated post period: `{timing.get('max')}` s",
                f"- Events ≥4 s: `{timing.get('n_events_ge_4s')}`",
                f"- Events ≥8 s: `{timing.get('n_events_ge_8s')}`",
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
    source_roots: Mapping[str, Path] | None = None,
    far002_certificate: Path | None = None,
    far003a_certificate: Path | None = None,
) -> dict[str, Any]:
    if output_json.exists() or output_report.exists():
        raise FileExistsError("refusing_to_overwrite_far_scout_002_archive")
    protocol_path = repo_root / PROTOCOL_RELATIVE
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing_protocol:{protocol_path}")
    far002_certificate = far002_certificate or (
        repo_root / DEFAULT_FAR002_RELATIVE
    )
    far003a_certificate = far003a_certificate or (
        repo_root / DEFAULT_FAR003A_RELATIVE
    )
    entry = {
        "far_002": _certificate(
            far002_certificate,
            protocol_id="FAR-002",
            status="PASS",
        ),
        "far_003a": _require_far003a(far003a_certificate),
    }
    far003a_payload = json.loads(
        far003a_certificate.read_text(encoding="utf-8")
    )
    payload = run_inventory(
        repo_root=repo_root,
        source_roots=source_roots,
        protocol_path=protocol_path,
        far003a_payload=far003a_payload,
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
        "--far002-certificate",
        type=Path,
        default=_repo_root() / DEFAULT_FAR002_RELATIVE,
    )
    parser.add_argument(
        "--far003a-certificate",
        type=Path,
        default=_repo_root() / DEFAULT_FAR003A_RELATIVE,
    )
    parser.add_argument(
        "--dandi-000458-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)
    source_roots = (
        {"dandi_000458": args.dandi_000458_root}
        if args.dandi_000458_root is not None
        else None
    )
    payload = run_gate(
        repo_root=_repo_root(),
        output_json=args.output_json,
        output_report=args.output_report,
        source_roots=source_roots,
        far002_certificate=args.far002_certificate,
        far003a_certificate=args.far003a_certificate,
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
