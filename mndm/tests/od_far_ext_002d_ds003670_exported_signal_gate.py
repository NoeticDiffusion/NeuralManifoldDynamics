"""Gate runner for FAR-EXT-002D ds003670 exported-signal eligibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MNDM_SRC = _REPO_ROOT / "mndm" / "src"
if str(_MNDM_SRC) not in sys.path:
    sys.path.insert(0, str(_MNDM_SRC))

from mndm.dynamical_families.far_ext_002d_ds003670_exported_signal import (  # noqa: E402
    DATASET_ID,
    PROTOCOL_ID,
    audit_far_ext_002d,
    load_scope,
    sha256_file,
)


FAR_ROOT = (
    _REPO_ROOT
    / "project"
    / "orthagonal_axis"
    / "orthagonal_dynamics"
    / "finite-amplitude_resilience"
)
PREREG_PATH = (
    FAR_ROOT
    / "012_far_ext_002d_ds003670_exported_signal_eligibility_prereg.md"
)
SCOPE_PATH = FAR_ROOT / "FAR-EXT-002D-SCOPE-0.1.json"
DEFAULT_FAR002A = FAR_ROOT / "results" / "far_ext_002a" / "far_ext_002a.json"
DEFAULT_METADATA_ROOT = (
    FAR_ROOT / "data" / "far_ext_001" / "openneuro_metadata_only" / DATASET_ID
)
DEFAULT_EEG_ROOT = (
    Path("K:/NeuralManifoldDynamics")
    / "far_ext_002b"
    / "openneuro_eeg_allowlist"
    / DATASET_ID
)
DEFAULT_OUTPUT = FAR_ROOT / "results" / "far_ext_002d" / "far_ext_002d.json"
DEFAULT_REPORT = FAR_ROOT / "results" / "far_ext_002d" / "far_ext_002d.md"


def _prereg_scope_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"scope manifest SHA256:\s*\n\s*([0-9a-f]{64})",
        text,
    )
    if match is None:
        raise ValueError("prereg_scope_hash_missing")
    return match.group(1)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# FAR-EXT-002D — ds003670 exported-signal / NMD-timebase eligibility",
        "",
        "## Gate status",
        "",
        "```text",
        f"status: {payload.get('global_status')}",
        f"reason: {payload.get('global_reason')}",
        f"promoted events: {payload.get('promoted_event_count')}",
        f"primary family: {payload.get('primary_family_id')}",
        f"technical status: {payload.get('exported_signal_technical_status', {}).get('selected_population_status')}",
        f"clock status: {payload.get('clock_status', {}).get('status')}",
        f"NMD timebase: {payload.get('nmd_timebase_status')}",
        "post-stimulation biological interpretability: NOT_ESTABLISHED",
        "MNPS calculated: false",
        "FAR calculated: false",
        "```",
        "",
        "## Family ledger",
        "",
    ]
    for family in payload.get("family_ledger", []):
        lines.extend(
            [
                f"### {family.get('family_id')}",
                "",
                f"- subjects/rho: `{family.get('subjects_per_rho')}`",
                f"- events/rho: `{family.get('events_per_rho')}`",
                f"- selected horizon: `{family.get('selected_post_horizon')}` s",
                f"- exported signal: `{family.get('exported_signal_technical_status')}`",
                f"- clock: `{family.get('clock_status')}`",
                f"- NMD timebase: `{family.get('nmd_timebase_status')}`",
                f"- eligibility: `{family.get('eligibility_status')}`",
                f"- reason: `{family.get('reason')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Claim boundary",
            "",
            str(payload.get("claim_boundary", "")),
            "",
            "## Audit scope",
            "",
            f"- Signal payloads opened: `{payload.get('audit_scope', {}).get('signal_payloads_opened')}`",
            f"- Outcome tables opened: `{payload.get('audit_scope', {}).get('outcome_tables_opened')}`",
            f"- NMD outputs opened: `{payload.get('audit_scope', {}).get('nmd_outputs_opened')}`",
            f"- MNPS calculated: `{payload.get('audit_scope', {}).get('mnps_calculated')}`",
            f"- FAR calculated: `{payload.get('audit_scope', {}).get('far_calculated')}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_gate(
    *,
    repo_root: Path = _REPO_ROOT,
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    eeg_root: Path = DEFAULT_EEG_ROOT,
    far002a_path: Path = DEFAULT_FAR002A,
    scope_path: Path = SCOPE_PATH,
    prereg_path: Path = PREREG_PATH,
    output_json: Path = DEFAULT_OUTPUT,
    output_report: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    if output_json.exists() or output_report.exists():
        raise FileExistsError("refusing_to_overwrite_far_ext_002d_archive")
    if not prereg_path.is_file():
        raise FileNotFoundError(f"missing_far_ext_002d_prereg:{prereg_path}")
    if not scope_path.is_file():
        raise FileNotFoundError(f"missing_far_ext_002d_scope:{scope_path}")
    expected_scope_hash = _prereg_scope_hash(prereg_path)
    actual_scope_hash = sha256_file(scope_path)
    if actual_scope_hash != expected_scope_hash:
        raise ValueError("far_ext_002d_scope_hash_mismatch")
    scope, _ = load_scope(scope_path)
    payload = audit_far_ext_002d(
        metadata_root=Path(metadata_root),
        eeg_root=Path(eeg_root),
        far002_path=Path(far002a_path),
        scope_path=Path(scope_path),
        repo_root=Path(repo_root),
    )
    payload.update(
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_path": str(prereg_path),
            "protocol_sha256": sha256_file(prereg_path),
            "scope_path": str(scope_path),
            "scope_sha256": actual_scope_hash,
            "metadata_root": str(metadata_root),
            "eeg_root": str(eeg_root),
            "scope_execution_status": scope["execution_status"],
        }
    )
    _write_json(output_json, payload)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_render_report(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run FAR-EXT-002D ds003670 exported-signal eligibility gate"
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--far002a", type=Path, default=DEFAULT_FAR002A)
    parser.add_argument("--scope", type=Path, default=SCOPE_PATH)
    parser.add_argument("--prereg", type=Path, default=PREREG_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    payload = run_gate(
        repo_root=args.repo_root,
        metadata_root=args.metadata_root,
        eeg_root=args.eeg_root,
        far002a_path=args.far002a,
        scope_path=args.scope,
        prereg_path=args.prereg,
        output_json=args.output_json,
        output_report=args.output_report,
    )
    print(
        json.dumps(
            {
                "protocol_id": payload["protocol_id"],
                "global_status": payload["global_status"],
                "global_reason": payload["global_reason"],
                "promoted_event_count": payload.get("promoted_event_count"),
                "family_count": payload.get("family_count"),
                "primary_family_id": payload.get("primary_family_id"),
                "post_stimulation_biological_interpretability": payload.get(
                    "post_stimulation_biological_interpretability"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
