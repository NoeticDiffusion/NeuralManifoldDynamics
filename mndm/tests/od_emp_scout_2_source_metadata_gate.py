"""Run the bounded OD-EMP-SCOUT-2 source-metadata inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from mndm.dynamical_families.emp_scout_2 import (  # noqa: E402
    PROTOCOL_ID,
    run_inventory,
)

PROTOCOL_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/commitor_dataset/"
    "OD-EMP-SCOUT-2.md"
)
DEFAULT_JSON_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/commitor_dataset/"
    "od_emp_scout_2_inventory.json"
)
DEFAULT_REPORT_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/commitor_dataset/"
    "od_emp_scout_2_inventory.md"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _candidate_lines(dataset: Mapping[str, Any]) -> list[str]:
    adjudication = dataset.get("adjudication") or {}
    if adjudication.get("candidate_present") is None:
        return [
            "- Candidate adjudication unavailable; this is not a "
            "`NO_EXTERNAL_RC` finding."
        ]
    matches = adjudication.get("matched_candidates") or []
    if not matches:
        return ["- No non-physiological external candidate matched."]
    lines = ["- Matched source candidates:"]
    for match in matches:
        lines.append(
            f"  - `{match.get('name')}` "
            f"({match.get('classification')}) "
            f"from `{match.get('relative_path') or match.get('source_file')}`"
        )
    return lines


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# OD-EMP-SCOUT-2 source metadata inventory",
        "",
        "## Gate status",
        "",
        "```text",
        f"gate: {PROTOCOL_ID}",
        f"status: {payload.get('gate_status')}",
        "downloaded: false",
        "NMD outputs read: false",
        "models fit: false",
        "committor estimator called: false",
        "```",
        "",
        "This is a source-metadata inventory, not a committor eligibility or",
        "empirical qualification result.",
        "",
        "## Dataset decisions",
        "",
    ]
    for dataset in payload.get("datasets", []):
        adjudication = dataset.get("adjudication") or {}
        lines.extend(
            [
                f"### {dataset.get('dataset_id')}",
                "",
                f"- Source status: `{dataset.get('source_status')}`",
                f"- Source root: `{dataset.get('source_root')}`",
                f"- Source root kind: `{dataset.get('source_root_kind')}`",
                "- Transition-proximity status: "
                f"`{adjudication.get('transition_proximity_status')}`",
                f"- Decision: `{adjudication.get('decision')}`",
                f"- Rationale: {adjudication.get('rationale')}",
                *_candidate_lines(dataset),
                "- File classes observed: "
                f"`{dataset.get('file_class_counts')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Re-ranking",
            "",
            "The prospective priority order is:",
            "",
            "```text",
            *[
                f"{index}. {dataset_id}"
                for index, dataset_id in enumerate(
                    payload.get("reranked_dataset_ids", []),
                    start=1,
                )
            ],
            "```",
            "",
            "The ranking identifies candidates for a future protocol; it does",
            "not establish transition-proximal state sufficiency.",
            "",
            "## RichSleep fallback",
            "",
            f"- Status: `{payload['richsleep_fallback']['status']}`",
            f"- Risk flag: `{payload['richsleep_fallback']['risk_flag']}`",
            f"- {payload['richsleep_fallback']['reason']}",
            "",
            "## Safe claim",
            "",
            "The inspected source metadata identifies candidate task variables",
            "for bounded follow-up, but no candidate is empirically qualified",
            "or authorized for a committor run by this audit.",
            "",
        ]
    )
    return "\n".join(lines)


def run_gate(
    *,
    repo_root: Path,
    output_json: Path,
    output_report: Path,
) -> dict[str, Any]:
    if output_json.exists() or output_report.exists():
        raise FileExistsError("refusing_to_overwrite_scout_2_archive")
    protocol_path = repo_root / PROTOCOL_RELATIVE
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing_protocol:{protocol_path}")
    payload = run_inventory(
        repo_root=repo_root,
        protocol_path=protocol_path,
    )
    _write_json(payload, output_json)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        _render_report(payload),
        encoding="utf-8",
    )
    return payload


def main() -> int:
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
    args = parser.parse_args()
    payload = run_gate(
        repo_root=_repo_root(),
        output_json=args.output_json,
        output_report=args.output_report,
    )
    print(
        json.dumps(
            {
                "protocol_id": payload["protocol_id"],
                "gate_status": payload["gate_status"],
                "datasets": len(payload["datasets"]),
                "output_json": str(args.output_json),
                "output_report": str(args.output_report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
