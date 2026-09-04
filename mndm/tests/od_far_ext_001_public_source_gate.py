"""Gate runner for FAR-EXT-001 public OpenNeuro source preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mndm.dynamical_families.far_ext_001_public_source_preflight import (
    PROTOCOL_ID,
    SOURCE_SPECS,
    download_public_metadata,
    metadata_download_config,
    run_inventory,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FAR_ROOT = (
    REPO_ROOT
    / "project"
    / "orthagonal_axis"
    / "orthagonal_dynamics"
    / "finite-amplitude_resilience"
)
PREREG_PATH = FAR_ROOT / "008_far_ext_001_public_source_preflight_prereg.md"
DEFAULT_OUTPUT = FAR_ROOT / "results" / "far_ext_001" / "far_ext_001.json"
DEFAULT_REPORT = FAR_ROOT / "results" / "far_ext_001" / "far_ext_001.md"
DEFAULT_METADATA_ROOT = (
    FAR_ROOT / "data" / "far_ext_001" / "openneuro_metadata_only"
)
DEFAULT_ENTRY_PATHS = {
    "far_scout_002": FAR_ROOT / "results" / "far_scout_002" / "far_scout_002_reconciled_20260821.json",
    "far_003a": FAR_ROOT / "results" / "far_003a" / "far_003a.json",
    "far_scout_003": FAR_ROOT / "results" / "far_scout_003" / "far_scout_003_reconciled_20260821_v3.json",
}
ENTRY_STATUSES = {
    "far_scout_002": {"NOT_TESTABLE"},
    "far_003a": {"METHOD_LIMITED"},
    "far_scout_003": {"NOT_TESTABLE"},
}
ENTRY_PROTOCOL_IDS = {
    "far_scout_002": "FAR-SCOUT-002",
    "far_003a": "FAR-003A",
    "far_scout_003": "FAR-SCOUT-003",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observed_status(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("gate_status")
    if value is None:
        value = payload.get("global_status", payload.get("status"))
    return str(value) if value is not None else None


def _require_entries(
    entry_paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for name, path in entry_paths.items():
        if name not in ENTRY_STATUSES:
            raise ValueError(f"unknown_entry:{name}")
        if not path.exists():
            raise FileNotFoundError(f"entry_missing:{name}:{path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = _observed_status(payload)
        if status not in ENTRY_STATUSES[name]:
            raise RuntimeError(f"entry_status_not_accepted:{name}:{status}")
        if payload.get("protocol_id") != ENTRY_PROTOCOL_IDS[name]:
            raise RuntimeError(f"entry_protocol_not_accepted:{name}")
        if payload.get("far_003b_authorized", False):
            raise RuntimeError(f"far_003b_must_remain_blocked:{name}")
        entries[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "protocol_id": payload.get("protocol_id"),
            "status": status,
        }
    if set(entries) != set(ENTRY_STATUSES):
        raise RuntimeError("entry_scope_incomplete")
    return entries


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# FAR-EXT-001 — Public-source semantics preflight",
        "",
        f"- Gate status: `{payload['gate_status']}`",
        f"- Gate reason: {payload['gate_reason']}",
        f"- NMD windows: `{payload['nmd_window_sec']}`",
        f"- Promotion candidates (audit classification only): "
        f"`{payload.get('promotion_candidates', [])}`",
        "",
        "## Source results",
        "",
    ]
    for dataset in payload["datasets"]:
        lines.extend(
            [
                f"### {dataset['dataset_id']}",
                "",
                f"- Classification: `{dataset['classification']}`",
                f"- Reason: {dataset['classification_reason']}",
                f"- Records audited: `{dataset.get('records_audited', 0)}`",
                f"- Event files: `{dataset.get('event_inventory', {}).get('event_file_count', 0)}`",
                f"- Signal payloads present: "
                f"`{dataset.get('audit_scope', {}).get('signal_payloads_present', False)}`",
            ]
        )
        for family in dataset.get("families", []):
            lines.extend(
                [
                    "",
                    f"- Family `{family['v_key']}`: "
                    f"`{family['classification']}`, "
                    f"rho=`{family['rho_levels']}`, "
                    f"T_isolated=`{family['clean_isolated_post_horizon_sec']}`, "
                    f"T_NMD=`{family['nmd_window_sec']}`",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Claim boundary",
            "",
            payload["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def run_gate(
    *,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    entry_paths: Mapping[str, str | Path] | None = None,
    download: bool = False,
    audit_existing_download: bool = False,
) -> dict[str, Any]:
    output = Path(output_path)
    report = Path(report_path)
    if output.exists() or report.exists():
        raise FileExistsError("refusing_to_overwrite_far_ext_001_output")

    normalized_entries = {
        name: Path(path)
        for name, path in (entry_paths or DEFAULT_ENTRY_PATHS).items()
    }
    entries = _require_entries(normalized_entries)
    root = Path(metadata_root)
    if download and audit_existing_download:
        raise ValueError("download_and_audit_existing_are_mutually_exclusive")
    if download:
        downloaded = download_public_metadata(root)
        source_roots = {dataset_id: str(path) for dataset_id, path in downloaded.items()}
    else:
        source_roots = {
            dataset_id: str(root / dataset_id) for dataset_id in SOURCE_SPECS
        }

    payload = run_inventory(source_roots)
    payload.update(
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_path": str(PREREG_PATH),
            "protocol_sha256": _sha256(PREREG_PATH),
            "entry_criteria": entries,
            "metadata_download": {
                "performed": download or audit_existing_download,
                "mode": (
                    "downloaded_during_gate"
                    if download
                    else "audited_previously_downloaded_root"
                    if audit_existing_download
                    else "not_performed"
                ),
                "root": str(root),
                "contract": metadata_download_config(),
            },
        }
    )
    _write_json(output, payload)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_render_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FAR-EXT-001 metadata-only gate")
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download only the explicit metadata/events subset first",
    )
    parser.add_argument(
        "--audit-existing-download",
        action="store_true",
        help="Audit a previously completed metadata-only download",
    )
    args = parser.parse_args()
    payload = run_gate(
        metadata_root=args.metadata_root,
        output_path=args.output,
        report_path=args.report,
        download=args.download,
        audit_existing_download=args.audit_existing_download,
    )
    print(json.dumps(
        {
            "protocol_id": payload["protocol_id"],
            "gate_status": payload["gate_status"],
            "gate_reason": payload["gate_reason"],
            "promotion_candidates": payload.get("promotion_candidates", []),
        },
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
