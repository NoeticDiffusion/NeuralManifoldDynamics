"""Run the read-only OD-EMP-DS004511-SEM-000 source audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from mndm.dynamical_families.ds004511_semantics import (  # noqa: E402
    DATASET_ID,
    PROTOCOL_ID,
    audit_source,
)

PROTOCOL_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "OD-EMP-DS004511-SEM-000.md"
)
DEFAULT_JSON_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "od_emp_ds004511_sem_000.json"
)
DEFAULT_REPORT_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "od_emp_ds004511_sem_000.md"
)
DEFAULT_SOURCE_ROOT = Path("M:/datasets/received/openneuro/ds004511")


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


def _render_report(payload: Mapping[str, Any]) -> str:
    variables = payload.get("variables", [])
    variable_lines = [
        f"- `{variable.get('name')}`: `{variable.get('classification')}` — "
        f"{variable.get('semantic_role', 'no semantic role recorded')}"
        for variable in variables
    ]
    pair_lines = [
        f"- `{pair.get('reaction_coordinate')}` → "
        f"`{pair.get('proposed_endpoints')}`: "
        f"`{', '.join(pair.get('classification', []))}` — "
        f"{pair.get('reason')}"
        for pair in payload.get("rc_pair_classifications", [])
    ]
    source_metadata = payload.get("source_metadata", {})
    source_files = source_metadata.get("source_files", {})
    events = source_files.get("events", {})
    behavior_summary = payload.get("behavior_summary", {})
    grain = behavior_summary.get("round_grain", {})
    reconstruction = payload.get("round_reconstruction", {})
    gate_status = payload.get("gate_status")
    negative_finding = gate_status == "NO_DYNAMIC_EXTERNAL_TASK_RC"
    if negative_finding:
        safe_claim = (
            "No genuine within-round external task RC was identified."
        )
        next_gate = "Do not start OD-EMP-000 for ds004511."
    else:
        safe_claim = (
            "No negative RC finding is permitted because the source audit "
            "did not complete."
        )
        next_gate = "Resolve the source-audit incompleteness before any next gate."
    return "\n".join(
        [
            "# OD-EMP-DS004511-SEM-000 source-semantics audit",
            "",
            "## Gate status",
            "",
            "```text",
            f"gate: {payload.get('protocol_id')}",
            f"dataset: {payload.get('dataset')}",
            f"status: {payload.get('gate_status')}",
            "read-only: true",
            "NMD outputs read: false",
            "models fit: false",
            "committor estimator called: false",
            "A/B selected: false",
            "```",
            "",
            "This is a source-semantics and RC-dynamicality audit. It is not",
            "a committor eligibility or empirical qualification result.",
            "",
            "## Source evidence",
            "",
            f"- Source root: `{payload.get('source_root')}`",
            f"- Source status: `{payload.get('source_status')}`",
            f"- Behavior files: "
            f"`{payload.get('behavior_summary', {}).get('n_behavior_files')}`",
            f"- Behavior rows: "
            f"`{payload.get('behavior_summary', {}).get('n_behavior_rows')}`",
            f"- Canonical EEG event rows: `{events.get('n_rows')}`",
            f"- EEG event values: `{events.get('trial_type_values')}`",
            f"- EEG events Sync-only: `{events.get('sync_only')}`",
            f"- Original source code present locally: "
            f"`{source_files.get('source_code_present')}`",
            f"- Available timestamp fields: "
            f"`{reconstruction.get('available_timestamp_fields', [])}`",
            "",
            "## Round reconstruction",
            "",
            "```text",
            "round onset → trial type → stake/probability → prediction",
            "→ die result → post-result reports → balance update → round end",
            "```",
            "",
            f"Behavior grain status: "
            f"`{grain.get('all_files_one_row_per_trial')}` "
            f"(row identifier: `{grain.get('row_identifier_field')}`, "
            f"Trial_Order complete: `{grain.get('trial_order_complete')}`).",
            f"EEG stage events present: "
            f"`{reconstruction.get('eeg_stage_events_present')}`; "
            f"event semantics: `{reconstruction.get('eeg_event_semantics')}`.",
            "",
            "## Variable classifications",
            "",
            *variable_lines,
            "",
            "## Candidate terminal events",
            "",
            *[
                f"- `{candidate.get('name')}`: fields "
                f"`{candidate.get('source_fields')}`, timestamps "
                f"`{candidate.get('timestamp_fields')}`; "
                f"`{candidate.get('classification')}`"
                + (
                    f"; semantic status "
                    f"`{candidate.get('semantic_status')}`"
                    if candidate.get("semantic_status")
                    else ""
                )
                for candidate in payload.get("terminal_candidates", [])
            ],
            "",
            "## RC/A-B classifications",
            "",
            *pair_lines,
            "",
            "## Unresolved semantics",
            "",
            *[
                f"- {reason}"
                for reason in payload.get("unresolved_semantics", [])
            ],
            "",
            "## Measurement-lead verdict",
            "",
            "```text",
            f"VERDICT: {payload.get('gate_status')}",
            "Contract impact: no ingest or committor contract changed.",
            "Validation: source hashes, behavior-table census, round-grain,",
            "and event/physiology metadata were audited read-only.",
            f"Safe claim: {safe_claim}",
            "Not established: any A/B pair, committor eligibility, or q estimate.",
            f"Next gate: {next_gate}",
            "```",
            "",
        ]
    )


def run_gate(
    *,
    repo_root: Path,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_json: Path,
    output_report: Path,
) -> dict[str, Any]:
    if output_json.exists() or output_report.exists():
        raise FileExistsError(
            "refusing_to_overwrite_od_emp_ds004511_sem_000_archive"
        )
    protocol_path = repo_root / PROTOCOL_RELATIVE
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing_protocol:{protocol_path}")
    payload = audit_source(
        source_root=source_root,
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
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )
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
        source_root=args.source_root,
        output_json=args.output_json,
        output_report=args.output_report,
    )
    print(
        json.dumps(
            {
                "protocol_id": payload["protocol_id"],
                "dataset": DATASET_ID,
                "gate_status": payload["gate_status"],
                "output_json": str(args.output_json),
                "output_report": str(args.output_report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
