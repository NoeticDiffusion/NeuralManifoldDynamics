"""Gate runner for FAR-EXT-002C ds003670 native rail methods audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MNDM_SRC = _REPO_ROOT / "mndm" / "src"
if str(_MNDM_SRC) not in sys.path:
    sys.path.insert(0, str(_MNDM_SRC))

from mndm.dynamical_families.far_ext_002c_ds003670_native_rail_methods import (  # noqa: E402
    DATASET_ID,
    PROTOCOL_ID,
    audit_far_ext_002c,
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
    / "011_far_ext_002c_ds003670_native_rail_methods_prereg.md"
)
DEFAULT_FAR002B = FAR_ROOT / "results" / "far_ext_002b" / "far_ext_002b.json"
DEFAULT_METADATA_ROOT = (
    FAR_ROOT / "data" / "far_ext_001" / "openneuro_metadata_only" / DATASET_ID
)
DEFAULT_EEG_ROOT = (
    Path("K:/NeuralManifoldDynamics")
    / "far_ext_002b"
    / "openneuro_eeg_allowlist"
    / DATASET_ID
)
DEFAULT_OUTPUT = FAR_ROOT / "results" / "far_ext_002c" / "far_ext_002c.json"
DEFAULT_REPORT = FAR_ROOT / "results" / "far_ext_002c" / "far_ext_002c.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _render_report(payload: Mapping[str, Any]) -> str:
    criterion = payload.get("criterion_assessment", {})
    lines = [
        "# FAR-EXT-002C — ds003670 native rail/full-scale methods audit",
        "",
        "## Audit status",
        "",
        "```text",
        f"status: {payload.get('global_status')}",
        f"reason: {payload.get('global_reason')}",
        f"documentation: {payload.get('documentation_status')}",
        f"native float rail: {payload.get('native_float_rail_status')}",
        f"reconciliation authorized: {payload.get('reconciliation_authorized')}",
        "002B reconciliation performed: false",
        "MNPS calculated: false",
        "FAR calculated: false",
        "```",
        "",
        "## Required criterion fields",
        "",
    ]
    for field, assessment in criterion.get("field_ledger", {}).items():
        lines.append(
            f"- `{field}`: `{assessment.get('status')}` "
            f"(candidates: {assessment.get('candidates')})"
        )
    lines.extend(
        [
            "",
            "## Evidence sources",
            "",
        ]
    )
    for document in payload.get("source_documents", []):
        lines.append(
            f"- `{document.get('id')}`: `{document.get('status')}` — "
            f"{document.get('url')}"
        )
    lines.extend(
        [
            "",
            "## Local metadata audit",
            "",
            f"- EEG sidecars: `{payload.get('local_sidecar_audit', {}).get('sidecar_count')}`",
            f"- SET headers: `{payload.get('local_set_header_audit', {}).get('set_header_count')}`",
            f"- SET scale fields: `{payload.get('local_set_header_audit', {}).get('scale_fields_present')}`",
            f"- FDT payload opened: `{payload.get('local_set_header_audit', {}).get('fdt_payload_opened')}`",
            "",
            "## Remote CNT header audit",
            "",
            f"- status: `{payload.get('remote_cnt_header_audit', {}).get('status')}`",
            f"- reason: `{payload.get('remote_cnt_header_audit', {}).get('reason')}`",
            f"- signal samples inspected: `{payload.get('remote_cnt_header_audit', {}).get('signal_samples_inspected')}`",
            "",
            "## Claim boundary",
            "",
            str(payload.get("claim_boundary", "")),
            "",
        ]
    )
    return "\n".join(lines)


def run_gate(
    *,
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    eeg_root: Path = DEFAULT_EEG_ROOT,
    far002b_path: Path = DEFAULT_FAR002B,
    output_json: Path = DEFAULT_OUTPUT,
    output_report: Path = DEFAULT_REPORT,
    fetch_documents: bool = True,
    probe_remote_header: bool = True,
) -> dict[str, Any]:
    if output_json.exists() or output_report.exists():
        raise FileExistsError("refusing_to_overwrite_far_ext_002c_archive")
    if not PREREG_PATH.is_file():
        raise FileNotFoundError(f"missing_far_ext_002c_prereg:{PREREG_PATH}")
    payload = audit_far_ext_002c(
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        far002b_path=far002b_path,
        fetch_documents=fetch_documents,
        probe_remote_header=probe_remote_header,
    )
    payload.update(
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_path": str(PREREG_PATH),
            "protocol_sha256": sha256_file(PREREG_PATH),
            "metadata_root": str(metadata_root),
            "eeg_root": str(eeg_root),
        }
    )
    _write_json(output_json, payload)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_render_report(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run FAR-EXT-002C ds003670 native rail methods audit"
    )
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--far002b", type=Path, default=DEFAULT_FAR002B)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--skip-documents",
        dest="fetch_documents",
        action="store_false",
        help="skip the required public-document fetch (non-canonical dry run)",
    )
    parser.add_argument(
        "--skip-zenodo-header",
        dest="probe_remote_header",
        action="store_false",
        help="skip the remote CNT header probe (non-canonical dry run)",
    )
    parser.set_defaults(fetch_documents=True, probe_remote_header=True)
    args = parser.parse_args(argv)
    payload = run_gate(
        metadata_root=args.metadata_root,
        eeg_root=args.eeg_root,
        far002b_path=args.far002b,
        output_json=args.output_json,
        output_report=args.output_report,
        fetch_documents=args.fetch_documents,
        probe_remote_header=args.probe_remote_header,
    )
    print(
        json.dumps(
            {
                "protocol_id": payload["protocol_id"],
                "global_status": payload["global_status"],
                "global_reason": payload["global_reason"],
                "native_float_rail_status": payload.get(
                    "native_float_rail_status"
                ),
                "reconciliation_authorized": payload.get(
                    "reconciliation_authorized"
                ),
                "output_json": str(args.output_json),
                "output_report": str(args.output_report),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
