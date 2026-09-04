"""Run the FAR-001B native source-specific metadata audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from mndm.dynamical_families.far_source_metadata_audit import (  # noqa: E402
    PROTOCOL_ID,
    run_inventory,
    sha256_file,
)


PROTOCOL_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/003_far_001b_source_specific_prereg.md"
)
DEFAULT_JSON_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_001b/far_001b.json"
)
DEFAULT_REPORT_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_001b/far_001b.md"
)
DEFAULT_FAR000_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_000/"
    "far_000_reconciled_20260819.json"
)
DEFAULT_FAR001_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_001/"
    "far_001_reconciled_20260819.json"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_certificate(path: Path, *, expected: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing_{expected}_certificate:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if expected == "far_000":
        valid = (
            payload.get("status") == "PASS"
            and (payload.get("decision") or {}).get("far_001_authorized") is True
        )
    else:
        valid = (
            payload.get("protocol_id") == "FAR-001"
            and payload.get("gate_status") == "NOT_TESTABLE"
        )
    if not valid:
        raise ValueError(f"{expected}_entry_criterion_not_satisfied:{path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": payload.get("status") or payload.get("gate_status"),
    }


def _write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# FAR-001B native source-specific metadata audit",
        "",
        "## Gate status",
        "",
        "```text",
        f"protocol: {payload.get('protocol_id')}",
        f"status: {payload.get('gate_status')}",
        f"reason: {payload.get('gate_reason')}",
        f"FAR-002 authorized: {payload.get('far_002_authorized')}",
        "signal payloads opened: false",
        "NMD outputs read: false",
        "dose reconstructed: false",
        "```",
        "",
        "This audit reads native metadata only. It does not measure resilience.",
        "",
        "## Source decisions",
        "",
    ]
    for dataset in payload.get("datasets", []):
        lines.extend(
            [
                f"### {dataset.get('dataset_id')}",
                "",
                f"- Source status: `{dataset.get('source_status')}`",
                f"- Root: `{dataset.get('source_root')}`",
                f"- Dataset classification: `{dataset.get('classification')}`",
                f"- Reason: {dataset.get('classification_reason', dataset.get('failure_reason', ''))}",
            ]
        )
        identities = dataset.get("identities") or []
        for identity in identities:
            lines.append(
                f"- Identity `{identity.get('identity')}`: "
                f"`{identity.get('classification')}`, "
                f"rho field `{identity.get('rho_field')}`, "
                f"unit `{identity.get('rho_unit')}`"
            )
            if identity.get("candidate_scope"):
                lines.append(f"- Candidate scope: {identity.get('candidate_scope')}")
        if dataset.get("rho_field"):
            lines.append(
                f"- rho field: `{dataset.get('rho_field')}`, "
                f"unit: `{dataset.get('rho_unit')}`, "
                f"levels: `{dataset.get('rho_levels')}`"
            )
        if dataset.get("candidate_rho_fields"):
            lines.append(
                f"- Candidate fields without admissible units/timing: "
                f"`{dataset.get('candidate_rho_fields')}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Eligible candidates",
            "",
            "```text",
            *[
                f"{candidate.get('dataset_id')} / {candidate.get('identity')}: "
                f"{candidate.get('classification')}"
                for candidate in payload.get("eligible_candidates", [])
            ],
            "```",
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
    far000_certificate: Path | None = None,
    far001_certificate: Path | None = None,
    source_roots: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    if output_json.exists() or output_report.exists():
        raise FileExistsError("refusing_to_overwrite_far_001b_archive")
    far000_certificate = far000_certificate or (repo_root / DEFAULT_FAR000_RELATIVE)
    far001_certificate = far001_certificate or (repo_root / DEFAULT_FAR001_RELATIVE)
    entry_far000 = _require_certificate(far000_certificate, expected="far_000")
    entry_far001 = _require_certificate(far001_certificate, expected="far_001")
    protocol_path = repo_root / PROTOCOL_RELATIVE
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing_far_001b_protocol:{protocol_path}")
    payload = run_inventory(
        source_roots=source_roots,
        protocol_path=protocol_path,
    )
    payload["entry_criteria"] = {
        "far_000": entry_far000,
        "far_001": entry_far001,
    }
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
        "--far000-certificate",
        type=Path,
        default=_repo_root() / DEFAULT_FAR000_RELATIVE,
    )
    parser.add_argument(
        "--far001-certificate",
        type=Path,
        default=_repo_root() / DEFAULT_FAR001_RELATIVE,
    )
    parser.add_argument("--dandi-000458-root", type=Path, default=None)
    parser.add_argument("--dandi-000009-root", type=Path, default=None)
    parser.add_argument("--ds006623-root", type=Path, default=None)
    parser.add_argument("--ds005917-root", type=Path, default=None)
    args = parser.parse_args(argv)
    overrides = {
        dataset_id: root
        for dataset_id, root in (
            ("dandi_000458", args.dandi_000458_root),
            ("dandi_000009", args.dandi_000009_root),
            ("ds006623", args.ds006623_root),
            ("ds005917", args.ds005917_root),
        )
        if root is not None
    }
    payload = run_gate(
        repo_root=_repo_root(),
        output_json=args.output_json,
        output_report=args.output_report,
        far000_certificate=args.far000_certificate,
        far001_certificate=args.far001_certificate,
        source_roots=overrides,
    )
    print(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "gate_status": payload["gate_status"],
                "gate_reason": payload["gate_reason"],
                "eligible_candidates": payload["eligible_candidates"],
                "far_002_authorized": payload["far_002_authorized"],
                "output_json": str(args.output_json),
                "output_report": str(args.output_report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
