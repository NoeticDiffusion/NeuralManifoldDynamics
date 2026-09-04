"""Gate runner for FAR-EXT-002B ds003670 signal/time-base eligibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MNDM_SRC = _REPO_ROOT / "mndm" / "src"
_OPENNEURO_SRC = _REPO_ROOT / "openneuro_ingest" / "src"
for _path in (_MNDM_SRC, _OPENNEURO_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mndm.dynamical_families.far_ext_002b_ds003670_signal_timebase import (  # noqa: E402
    DATASET_ID,
    PROTOCOL_ID,
    audit_far_ext_002b,
    eeg_payload_allowlist,
    promoted_event_ledger,
    referenced_fdt_path,
    sha256_file,
)


FAR_ROOT = (
    _REPO_ROOT
    / "project"
    / "orthagonal_axis"
    / "orthagonal_dynamics"
    / "finite-amplitude_resilience"
)
PREREG_PATH = FAR_ROOT / "010_far_ext_002b_ds003670_signal_timebase_prereg.md"
DEFAULT_FAR002 = FAR_ROOT / "results" / "far_ext_002a" / "far_ext_002a.json"
DEFAULT_METADATA_ROOT = (
    FAR_ROOT / "data" / "far_ext_001" / "openneuro_metadata_only" / DATASET_ID
)
DEFAULT_EEG_ROOT = (
    FAR_ROOT / "data" / "far_ext_002b" / "openneuro_eeg_allowlist" / DATASET_ID
)
DEFAULT_OUTPUT = FAR_ROOT / "results" / "far_ext_002b" / "far_ext_002b.json"
DEFAULT_REPORT = FAR_ROOT / "results" / "far_ext_002b" / "far_ext_002b.md"


def _load_entry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing_far_ext_002a_certificate:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("far_ext_002a_certificate_not_object")
    if payload.get("protocol_id") != "FAR-EXT-002A":
        raise ValueError("far_ext_002a_protocol_id_mismatch")
    if payload.get("dataset_id") != DATASET_ID:
        raise ValueError("far_ext_002a_dataset_id_mismatch")
    if payload.get("global_status") != "DS003670_LIMITED_CURVE_SEMANTICS_PASS":
        raise ValueError("far_ext_002a_global_status_not_pass")
    return payload


def _download_payload(
    *,
    far002_payload: Mapping[str, Any],
    eeg_root: Path,
) -> dict[str, Any]:
    """Download sets, inspect only their declared FDT names, then fetch FDTs."""
    from openneuro.download import download_explicit_allowlist

    ledger = promoted_event_ledger(far002_payload)
    set_paths = eeg_payload_allowlist(ledger)
    target_parent = eeg_root.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    download_explicit_allowlist(
        DATASET_ID,
        set_paths,
        target_parent,
    )
    fdt_paths: set[str] = set()
    for relative_set in set_paths:
        fdt_path = referenced_fdt_path(eeg_root / Path(relative_set))
        if fdt_path is None:
            continue
        fdt_paths.add(
            str(fdt_path.relative_to(eeg_root)).replace("\\", "/")
        )
    if fdt_paths:
        download_explicit_allowlist(
            DATASET_ID,
            sorted(fdt_paths),
            target_parent,
            existing_paths=set_paths,
        )
    return {
        "performed": True,
        "mode": "downloaded_during_gate",
        "dataset_id": DATASET_ID,
        "set_paths": set_paths,
        "fdt_paths": sorted(fdt_paths),
        "payload_root": str(eeg_root),
        "payload_hashes": {
            relative: sha256_file(eeg_root / Path(relative))
            for relative in sorted(set(set_paths) | fdt_paths)
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# FAR-EXT-002B — ds003670 signal/time-base eligibility",
        "",
        "## Gate status",
        "",
        "```text",
        f"status: {payload.get('global_status')}",
        f"reason: {payload.get('global_reason')}",
        f"promoted events: {payload.get('promoted_event_count')}",
        f"far_003b_authorized: {payload.get('far_003b_authorized')}",
        "mnps calculated: false",
        "far calculated: false",
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
                f"- v: `{family.get('v')}`",
                f"- rho levels: `{family.get('rho_levels')}`",
                f"- subjects/rho: `{family.get('subjects_per_rho')}`",
                f"- selected horizon: `{family.get('selected_post_horizon')}` s",
                f"- clock: `{family.get('clock_status')}`",
                f"- native signal: `{family.get('native_signal_status')}`",
                f"- NMD lattice: `{family.get('nmd_timebase_status')}`",
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
    far002_path: Path = DEFAULT_FAR002,
    output_json: Path = DEFAULT_OUTPUT,
    output_report: Path = DEFAULT_REPORT,
    download: bool = False,
    audit_existing_download: bool = False,
) -> dict[str, Any]:
    if output_json.exists() or output_report.exists():
        raise FileExistsError("refusing_to_overwrite_far_ext_002b_archive")
    if not PREREG_PATH.is_file():
        raise FileNotFoundError(f"missing_far_ext_002b_prereg:{PREREG_PATH}")
    far002_payload = _load_entry(far002_path)
    if download:
        download_info = _download_payload(
            far002_payload=far002_payload,
            eeg_root=eeg_root,
        )
    elif audit_existing_download:
        if not eeg_root.is_dir():
            raise FileNotFoundError(
                f"existing_eeg_payload_root_missing:{eeg_root}"
            )
        download_info = {
            "performed": True,
            "mode": "audited_previously_downloaded_root",
            "payload_root": str(eeg_root),
        }
    else:
        download_info = {
            "performed": False,
            "mode": "audited_existing_root",
            "payload_root": str(eeg_root),
        }
    payload = audit_far_ext_002b(
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        far002_payload=far002_payload,
        repo_root=repo_root,
        far002_path=far002_path,
    )
    payload.update(
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_path": str(PREREG_PATH),
            "protocol_sha256": sha256_file(PREREG_PATH),
            "metadata_root": str(metadata_root),
            "payload_download": download_info,
        }
    )
    _write_json(output_json, payload)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_render_report(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run FAR-EXT-002B ds003670 signal/time-base gate"
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--far002", type=Path, default=DEFAULT_FAR002)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download only the frozen Experiment 1 EEG allowlist",
    )
    parser.add_argument(
        "--audit-existing-download",
        action="store_true",
        help="audit a payload downloaded by a previous gate attempt",
    )
    args = parser.parse_args(argv)
    payload = run_gate(
        repo_root=args.repo_root,
        metadata_root=args.metadata_root,
        eeg_root=args.eeg_root,
        far002_path=args.far002,
        output_json=args.output_json,
        output_report=args.output_report,
        download=args.download,
        audit_existing_download=args.audit_existing_download,
    )
    print(
        json.dumps(
            {
                "protocol_id": payload["protocol_id"],
                "global_status": payload["global_status"],
                "global_reason": payload["global_reason"],
                "promoted_event_count": payload["promoted_event_count"],
                "family_count": payload.get("family_count"),
                "far_003b_authorized": payload["far_003b_authorized"],
                "output_json": str(args.output_json),
                "output_report": str(args.output_report),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
