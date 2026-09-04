"""Run the FAR-001 perturbation-source scout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from mndm.dynamical_families.far_source_scout import (  # noqa: E402
    PROTOCOL_ID,
    run_inventory,
    sha256_file,
)


PROTOCOL_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/002_far_source_scout_prereg.md"
)
DEFAULT_JSON_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_001/far_001.json"
)
DEFAULT_REPORT_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_001/far_001.md"
)
DEFAULT_FAR000_RELATIVE = Path(
    "project/orthagonal_axis/orthagonal_dynamics/"
    "finite-amplitude_resilience/results/far_000/"
    "far_000_reconciled_20260819.json"
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


def _require_far000_pass(certificate_path: Path) -> dict[str, Any]:
    if not certificate_path.is_file():
        raise FileNotFoundError(f"missing_far_000_certificate:{certificate_path}")
    payload = json.loads(certificate_path.read_text(encoding="utf-8"))
    decision = payload.get("decision") or {}
    if payload.get("status") != "PASS" or decision.get("far_001_authorized") is not True:
        raise ValueError(f"far_000_not_pass:{certificate_path}")
    return {
        "path": str(certificate_path),
        "sha256": sha256_file(certificate_path),
        "status": payload.get("status"),
        "far_001_authorized": decision.get("far_001_authorized"),
    }


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# FAR-001 source scout",
        "",
        "## Gate status",
        "",
        "```text",
        f"protocol: {payload.get('protocol_id')}",
        f"status: {payload.get('gate_status')}",
        f"reason: {payload.get('gate_reason')}",
        "empirical outputs read: false",
        "signal payloads opened: false",
        "downloads performed: false",
        "resilience curve estimated: false",
        "```",
        "",
        "This is a source-semantic inventory, not an empirical resilience "
        "measurement.",
        "",
        "## Dataset decisions",
        "",
    ]
    for dataset in payload.get("datasets", []):
        semantics = dataset.get("source_semantics") or {}
        lines.extend(
            [
                f"### {dataset.get('dataset_id')}",
                "",
                f"- Source status: `{dataset.get('source_status')}`",
                f"- Source kind: `{dataset.get('source_kind', 'photic_eeg')}`",
                f"- Source root: `{dataset.get('source_root')}`",
                f"- Classification: `{dataset.get('classification')}`",
                f"- Rationale: {dataset.get('classification_reason')}",
                f"- Explicit perturbation: `{semantics.get('perturbation_explicitly_imposed')}`",
                f"- rho known: `{semantics.get('rho_known')}`",
                f"- rho unit: `{semantics.get('rho_unit')}`",
                f"- Multiple rho levels: `{semantics.get('multiple_rho_levels')}`",
                f"- rho=0/sham: `{semantics.get('rho_zero_or_sham')}`",
                f"- Direction/type known: `{semantics.get('v_known')}`",
                f"- Onset known: `{semantics.get('onset_known')}`",
                f"- Offset known: `{semantics.get('offset_known')}`",
                f"- Repeated perturbations: `{semantics.get('repeated_perturbations')}`",
                f"- Continuous neural signal: `{semantics.get('continuous_neural_signal')}`",
                "",
            ]
        )
        if dataset.get("dataset_id") == "ds006036":
            lines.extend(
                [
                    f"- Participants: `{dataset.get('participant_count')}`",
                    f"- Event files: `{dataset.get('event_file_count')}`",
                    f"- Photostimulation events: `{dataset.get('photo_event_count')}`",
                    f"- Frequency levels (Hz): `{dataset.get('photo_frequency_levels_hz')}`",
                    f"- Event fields: `{dataset.get('event_fields')}`",
                    f"- Intensity/luminance fields: `{dataset.get('intensity_or_luminance_fields')}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## Guardrail",
            "",
            "For ds006036, stimulation frequency remains `v` (direction/type). "
            "It is not interpreted as `rho` because the audited source does "
            "not provide a physical intensity/amplitude ladder or sham.",
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
    source_roots: Mapping[str, Path] | None = None,
    far000_certificate: Path | None = None,
) -> dict[str, Any]:
    if output_json.exists() or output_report.exists():
        raise FileExistsError("refusing_to_overwrite_far_001_archive")
    protocol_path = repo_root / PROTOCOL_RELATIVE
    far000_certificate = far000_certificate or (
        repo_root / DEFAULT_FAR000_RELATIVE
    )
    entry_criterion = _require_far000_pass(far000_certificate)
    payload = run_inventory(
        repo_root=repo_root,
        source_roots=source_roots,
        protocol_path=protocol_path,
    )
    payload["entry_criterion"] = entry_criterion
    _write_json(payload, output_json)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        _render_report(payload),
        encoding="utf-8",
    )
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
    parser.add_argument("--ds006036-root", type=Path, default=None)
    parser.add_argument(
        "--far000-certificate",
        type=Path,
        default=_repo_root() / DEFAULT_FAR000_RELATIVE,
    )
    args = parser.parse_args(argv)
    source_roots = (
        {"ds006036": args.ds006036_root}
        if args.ds006036_root is not None
        else None
    )
    payload = run_gate(
        repo_root=_repo_root(),
        output_json=args.output_json,
        output_report=args.output_report,
        source_roots=source_roots,
        far000_certificate=args.far000_certificate,
    )
    print(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "gate_status": payload["gate_status"],
                "gate_reason": payload["gate_reason"],
                "classifications": payload["classification_counts"],
                "output_json": str(args.output_json),
                "output_report": str(args.output_report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
