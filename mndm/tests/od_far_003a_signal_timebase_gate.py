"""Run the FAR-003A native EEG and NMD time-base eligibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from mndm.dynamical_families.far_signal_timebase_eligibility import (  # noqa: E402
    PROTOCOL_ID,
    audit_far_003a,
    sha256_file,
)


PROTOCOL_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/005_far_003a_signal_timebase_prereg.md"
)
DEFAULT_JSON_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_003a/far_003a.json"
)
DEFAULT_REPORT_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_003a/far_003a.md"
)
DEFAULT_FAR002_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_002/"
    "far_002_reconciled_20260820.json"
)
DEFAULT_SOURCE_ROOT = Path(
    r"K:\ExternalReceivedDatasets\DANDI\dandi_000458\raw\000458"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_far002(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing_far_002_certificate:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "FAR-002":
        raise ValueError("far_002_protocol_id_mismatch")
    if payload.get("global_status") != "PASS":
        raise ValueError("far_002_entry_status_not_pass")
    curve_rows = [
        row
        for row in payload.get("perturbation_family_ledger", [])
        if row.get("curve_status") == "CURVE_SEMANTICS_PASS"
    ]
    if len(curve_rows) != 10:
        raise ValueError(f"far_002_curve_row_count_mismatch:{len(curve_rows)}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": payload.get("global_status"),
        "curve_row_count": len(curve_rows),
    }


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# FAR-003A native signal and NMD time-base eligibility",
        "",
        "## Gate status",
        "",
        "```text",
        f"status: {payload.get('global_status')}",
        f"reason: {payload.get('global_reason')}",
        f"FAR-003B authorized: {payload.get('far_003b_authorized')}",
        "response statistics: false",
        "home/away: false",
        "```",
        "",
        "The gate is outcome-blind and uses the primary EEG surface only.",
        "",
        "## Family decisions",
        "",
    ]
    for row in payload.get("family_ledger", []):
        lines.extend(
            [
                f"### {row.get('family_id')} / {row.get('w')}",
                "",
                f"- rho levels: `{row.get('n_rho_levels')}`",
                f"- raw sync/continuity: `{row.get('raw_signal_sync')}` / "
                f"`{row.get('raw_signal_continuity')}`",
                f"- artifact: `{row.get('artifact_status')}`",
                f"- candidate horizon: `{row.get('candidate_horizon')}`",
                f"- eligible events: `{row.get('n_eligible_events')}`",
                f"- status: `{row.get('status')}`",
                f"- reason: `{row.get('reason')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## NMD time base",
            "",
            f"- Status: `{payload.get('nmd_timebase', {}).get('status')}`",
            f"- Reason: `{payload.get('nmd_timebase', {}).get('reason')}`",
            "",
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
    far002_certificate: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    if output_json.exists() or output_report.exists():
        raise FileExistsError("refusing_to_overwrite_far_003a_archive")
    protocol_path = repo_root / PROTOCOL_RELATIVE
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing_far_003a_protocol:{protocol_path}")
    far002_certificate = far002_certificate or (
        repo_root / DEFAULT_FAR002_RELATIVE
    )
    entry_far002 = _require_far002(far002_certificate)
    source_root = source_root or DEFAULT_SOURCE_ROOT
    payload = audit_far_003a(
        source_root=source_root,
        far002_path=far002_certificate,
        protocol_path=protocol_path,
        repo_root=repo_root,
    )
    payload["entry_criteria"] = {"far_002": entry_far002}
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
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )
    args = parser.parse_args(argv)
    payload = run_gate(
        repo_root=_repo_root(),
        output_json=args.output_json,
        output_report=args.output_report,
        far002_certificate=args.far002_certificate,
        source_root=args.source_root,
    )
    print(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "global_status": payload["global_status"],
                "global_reason": payload["global_reason"],
                "family_count": len(payload["family_ledger"]),
                "far_003b_authorized": payload["far_003b_authorized"],
                "output_json": str(args.output_json),
                "output_report": str(args.output_report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
