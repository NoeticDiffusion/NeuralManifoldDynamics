"""Run the FAR-002 DANDI 000458 perturbation-family semantics gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from mndm.dynamical_families.far_perturbation_family_semantics import (  # noqa: E402
    PROTOCOL_ID,
    audit_dandi_000458,
    sha256_file,
)


PROTOCOL_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/004_far_002_perturbation_family_semantics_prereg.md"
)
DEFAULT_JSON_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_002/far_002.json"
)
DEFAULT_REPORT_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_002/far_002.md"
)
DEFAULT_FAR001B_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_001b/"
    "far_001b_reconciled_20260820.json"
)
DEFAULT_SOURCE_ROOT = Path(
    r"K:\ExternalReceivedDatasets\DANDI\dandi_000458\raw\000458"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_far001b(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing_far_001b_certificate:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    eligible = payload.get("eligible_candidates", [])
    selected = {
        "dataset_id": "dandi_000458",
        "identity": "electrical_stimulation_current",
        "classification": "CURVE_CANDIDATE",
    }
    if payload.get("protocol_id") != "FAR-001B":
        raise ValueError("far_001b_protocol_id_mismatch")
    if payload.get("gate_status") != "PASS":
        raise ValueError("far_001b_entry_criterion_not_pass")
    if selected not in eligible:
        raise ValueError("far_001b_selected_source_missing")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": payload.get("gate_status"),
        "selected_source": selected,
    }


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# FAR-002 DANDI 000458 perturbation-family semantics",
        "",
        "## Gate status",
        "",
        "```text",
        f"status: {payload.get('global_status')}",
        f"reason: {payload.get('global_reason')}",
        f"FAR-003A authorized: {payload.get('far_003a_authorized')}",
        "signal payloads opened: false",
        "NMD outcomes constructed: false",
        "home/away constructed: false",
        "```",
        "",
        "The ledger is a source-semantics audit, not a resilience measurement.",
        "",
        "## Perturbation-family ledger",
        "",
    ]
    for row in payload.get("perturbation_family_ledger", []):
        lines.extend(
            [
                f"### {row.get('family_id')} / {row.get('operating_state_w')}",
                "",
                f"- v: `{row.get('v_key')}`",
                f"- rho: `{row.get('rho_field')}` in `{row.get('rho_units')}`",
                f"- levels: `{row.get('rho_levels')}`",
                f"- per-level support: `{row.get('rho_level_ledger')}`",
                f"- subjects/sessions: `{row.get('n_subjects')}` / `{row.get('n_sessions')}`",
                f"- valid trials: `{row.get('n_trials_valid')}` "
                f"(invalid `{row.get('n_trials_invalid')}`)",
                f"- observed duration sec: `{row.get('stim_duration_min_sec')}`–"
                f"`{row.get('stim_duration_max_sec')}`",
                f"- zero control: `{row.get('zero_control_type')}`",
                f"- timing: `{row.get('stim_onset_semantics')}` / "
                f"`{row.get('stim_offset_semantics')}`",
                f"- status: `{row.get('curve_status')}`",
                f"- exclusion: `{row.get('exclusion_reason')}`",
                f"- unresolved v fields: `{row.get('unresolved_v_fields')}`",
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
    far001b_certificate: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    if output_json.exists() or output_report.exists():
        raise FileExistsError("refusing_to_overwrite_far_002_archive")
    protocol_path = repo_root / PROTOCOL_RELATIVE
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing_far_002_protocol:{protocol_path}")
    far001b_certificate = far001b_certificate or (
        repo_root / DEFAULT_FAR001B_RELATIVE
    )
    entry_far001b = _require_far001b(far001b_certificate)
    source_root = source_root or DEFAULT_SOURCE_ROOT
    payload = audit_dandi_000458(
        source_root=source_root,
        protocol_path=protocol_path,
    )
    payload["entry_criteria"] = {"far_001b": entry_far001b}
    payload["far_003a_authorized"] = payload["global_status"] in {
        "PASS",
        "LIMITED_PASS",
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
        "--far001b-certificate",
        type=Path,
        default=_repo_root() / DEFAULT_FAR001B_RELATIVE,
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
        far001b_certificate=args.far001b_certificate,
        source_root=args.source_root,
    )
    print(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "global_status": payload["global_status"],
                "global_reason": payload["global_reason"],
                "family_count": len(payload["perturbation_family_ledger"]),
                "far_003a_authorized": payload["far_003a_authorized"],
                "output_json": str(args.output_json),
                "output_report": str(args.output_report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
