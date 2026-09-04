"""Gate runner for FAR-EXT-002A ds003670 semantic completion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mndm.dynamical_families.far_ext_002a_ds003670_semantic_join import (
    DATASET_ID,
    PROTOCOL_ID,
    SOURCE_DOCUMENTS,
    audit_ds003670,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FAR_ROOT = (
    REPO_ROOT
    / "project"
    / "orthagonal_axis"
    / "orthagonal_dynamics"
    / "finite-amplitude_resilience"
)
PREREG_PATH = FAR_ROOT / "009_far_ext_002a_ds003670_semantic_completion_prereg.md"
DEFAULT_METADATA_ROOT = (
    FAR_ROOT / "data" / "far_ext_001" / "openneuro_metadata_only" / DATASET_ID
)
DEFAULT_ENTRY_PATH = FAR_ROOT / "results" / "far_ext_001" / "far_ext_001.json"
DEFAULT_OUTPUT = FAR_ROOT / "results" / "far_ext_002a" / "far_ext_002a.json"
DEFAULT_REPORT = FAR_ROOT / "results" / "far_ext_002a" / "far_ext_002a.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observed_status(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("gate_status", payload.get("global_status", payload.get("status")))
    return str(value) if value is not None else None


def _require_far_ext_001(path: Path, metadata_root: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"entry_missing:far_ext_001:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "FAR-EXT-001":
        raise RuntimeError("entry_protocol_not_accepted:far_ext_001")
    if _observed_status(payload) != "CENSUS_COMPLETE":
        raise RuntimeError(f"entry_status_not_accepted:far_ext_001:{_observed_status(payload)}")
    dataset = next(
        (item for item in payload.get("datasets", []) if item.get("dataset_id") == DATASET_ID),
        None,
    )
    if dataset is None:
        raise RuntimeError("entry_dataset_missing:ds003670")
    if dataset.get("audit_scope", {}).get("signal_payloads_present", False):
        raise RuntimeError("entry_signal_payload_present:ds003670")
    expected_root = Path(dataset.get("source_root", ""))
    if expected_root.resolve() != metadata_root.resolve():
        raise RuntimeError("entry_root_mismatch:ds003670")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "protocol_id": payload["protocol_id"],
        "status": _observed_status(payload),
        "dataset_source_root": str(expected_root),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# FAR-EXT-002A — ds003670 semantic completion",
        "",
        f"- Gate status: `{payload['global_status']}`",
        f"- Entry: `{payload['entry_criteria']['status']}`",
        f"- Source binding: `{payload['source_binding']['status']}`",
        f"- Event files: `{payload['event_file_count']}`",
        f"- Events joined: `{len(payload['events'])}`",
        "",
        "## Fixed-v families",
        "",
    ]
    for family in payload["families"]:
        lines.extend(
            [
                f"- `{family['v']}`: `{family['semantic_status']}`; "
                f"rho=`{family['rho_levels_mA']}`; "
                f"subjects/rho=`{family['subjects_per_rho']}`; "
                f"min T_isolated=`{family['min_T_isolated_sec']}` s",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            payload["claim_boundary"],
            "",
            "## Forbidden-action audit",
            "",
            f"- Signal payloads opened: `{payload['audit_scope']['signal_payloads_opened']}`",
            f"- Outcome tables opened: `{payload['audit_scope']['outcome_tables_opened']}`",
            f"- NMD outputs opened: `{payload['audit_scope']['nmd_outputs_opened']}`",
            f"- FAR calculated: `{payload['audit_scope']['far_calculated']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_gate(
    *,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    entry_path: str | Path = DEFAULT_ENTRY_PATH,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    root = Path(metadata_root)
    entry = Path(entry_path)
    output = Path(output_path)
    report = Path(report_path)
    if output.exists() or report.exists():
        raise FileExistsError("refusing_to_overwrite_far_ext_002a_output")
    if not root.exists():
        raise FileNotFoundError(f"metadata_root_missing:{root}")
    entry_certificate = _require_far_ext_001(entry, root)
    far001_payload = json.loads(entry.read_text(encoding="utf-8"))
    payload = audit_ds003670(root, far001_payload)
    payload.update(
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_path": str(PREREG_PATH),
            "protocol_sha256": _sha256(PREREG_PATH),
            "entry_criteria": entry_certificate,
            "source_documents": SOURCE_DOCUMENTS,
        }
    )
    _write_json(output, payload)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_render_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FAR-EXT-002A semantic gate")
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--entry", type=Path, default=DEFAULT_ENTRY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run_gate(
        metadata_root=args.metadata_root,
        entry_path=args.entry,
        output_path=args.output,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "protocol_id": payload["protocol_id"],
                "global_status": payload["global_status"],
                "source_binding": payload["source_binding"]["status"],
                "family_count": len(payload["families"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
