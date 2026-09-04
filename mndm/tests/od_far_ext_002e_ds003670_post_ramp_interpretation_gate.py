"""Gate runner for FAR-EXT-002E source-only interpretation bridge."""

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

from mndm.dynamical_families.far_ext_002e_ds003670_post_ramp_interpretation import (  # noqa: E402
    DATASET_ID,
    PROTOCOL_ID,
    audit_far_ext_002e,
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
    / "013_far_ext_002e_ds003670_post_ramp_interpretation_prereg.md"
)
SCOPE_PATH = FAR_ROOT / "FAR-EXT-002E-SCOPE-0.1.json"
DEFAULT_RESULT_002D = (
    FAR_ROOT / "results" / "far_ext_002d" / "far_ext_002d.json"
)
DEFAULT_OUTPUT = FAR_ROOT / "results" / "far_ext_002e" / "far_ext_002e.json"
DEFAULT_REPORT = FAR_ROOT / "results" / "far_ext_002e" / "far_ext_002e.md"


def _prereg_scope_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"scope manifest SHA256:\s*\n\s*([0-9a-f]{64})",
        text,
    )
    if match is None:
        raise ValueError("far_ext_002e_prereg_scope_hash_missing")
    return match.group(1)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _render_report(payload: Mapping[str, Any]) -> str:
    semantics = payload.get("source_semantics", {})
    interval = semantics.get("published_post_interval", {})
    amplitude = semantics.get("amplitude_transfer", {})
    lines = [
        "# FAR-EXT-002E — ds003670 F30 post-ramp interpretation bridge",
        "",
        "## Gate status",
        "",
        "```text",
        f"status: {payload.get('global_status')}",
        f"reason: {payload.get('global_reason')}",
        f"F30 family: {payload.get('f30_family_id')}",
        f"published post duration: {payload.get('f30_published_post_sec')} s",
        f"source-defined post delay: {interval.get('source_defined_offset_after_ramp_down_sec')} s",
        f"1.0 mA transfer: {amplitude.get('1.0_mA', {}).get('status')}",
        f"0.5 mA transfer: {amplitude.get('0.5_mA', {}).get('status')}",
        "artifact-free claim: false",
        "MNPS calculated: false",
        "FAR calculated: false",
        "FAR-EXT-003B authorized: false",
        "```",
        "",
        "## Binding",
        "",
        f"- 002D: `{payload.get('entry_far_ext_002d', {}).get('status')}`",
        f"- 002D result SHA256: `{payload.get('entry_far_ext_002d', {}).get('result', {}).get('sha256')}`",
        f"- F30 subset SHA256: `{payload.get('entry_far_ext_002d', {}).get('result', {}).get('f30_subset_sha256')}`",
        "",
        "## Source documents",
        "",
    ]
    for document in payload.get("source_documents", []):
        lines.append(
            f"- `{document.get('id')}`: `{document.get('status')}` — "
            f"{document.get('url')}"
        )
    lines.extend(
        [
            "",
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
    result_002d: Path = DEFAULT_RESULT_002D,
    scope_path: Path = SCOPE_PATH,
    prereg_path: Path = PREREG_PATH,
    output_json: Path = DEFAULT_OUTPUT,
    output_report: Path = DEFAULT_REPORT,
    fetch_documents: bool = True,
) -> dict[str, Any]:
    if output_json.exists() or output_report.exists():
        raise FileExistsError("refusing_to_overwrite_far_ext_002e_archive")
    if not prereg_path.is_file():
        raise FileNotFoundError(f"missing_far_ext_002e_prereg:{prereg_path}")
    if not scope_path.is_file():
        raise FileNotFoundError(f"missing_far_ext_002e_scope:{scope_path}")
    expected_scope_hash = _prereg_scope_hash(prereg_path)
    actual_scope_hash = sha256_file(scope_path)
    if actual_scope_hash != expected_scope_hash:
        raise ValueError("far_ext_002e_scope_hash_mismatch")
    scope, _ = load_scope(scope_path)
    payload = audit_far_ext_002e(
        scope_path=scope_path,
        repo_root=repo_root,
        result_path=result_002d,
        fetch_documents=fetch_documents,
    )
    payload.update(
        {
            "protocol_id": PROTOCOL_ID,
            "dataset_id": DATASET_ID,
            "protocol_path": str(prereg_path),
            "protocol_sha256": sha256_file(prereg_path),
            "scope_path": str(scope_path),
            "scope_sha256": actual_scope_hash,
            "result_002d_path": str(result_002d),
            "scope_execution_status": scope["execution_status"],
        }
    )
    _write_json(output_json, payload)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_render_report(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run FAR-EXT-002E source-only interpretation bridge"
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--result-002d", type=Path, default=DEFAULT_RESULT_002D)
    parser.add_argument("--scope", type=Path, default=SCOPE_PATH)
    parser.add_argument("--prereg", type=Path, default=PREREG_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--skip-documents",
        dest="fetch_documents",
        action="store_false",
        help="skip public source fetches (non-canonical dry run)",
    )
    parser.set_defaults(fetch_documents=True)
    args = parser.parse_args(argv)
    payload = run_gate(
        repo_root=args.repo_root,
        result_002d=args.result_002d,
        scope_path=args.scope,
        prereg_path=args.prereg,
        output_json=args.output_json,
        output_report=args.output_report,
        fetch_documents=args.fetch_documents,
    )
    print(
        json.dumps(
            {
                "protocol_id": payload["protocol_id"],
                "global_status": payload["global_status"],
                "global_reason": payload["global_reason"],
                "f30_family_id": payload.get("f30_family_id"),
                "f30_published_post_sec": payload.get(
                    "f30_published_post_sec"
                ),
                "allowed_later_scope": payload.get("allowed_later_scope"),
                "output_json": str(args.output_json),
                "output_report": str(args.output_report),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
