"""Run-output structure checking for Noetic ingest.

This module validates that a summarization run has the expected
file layout and (optionally) that fMRI recordings embed `/regions/*` inside
their per-recording H5.

Designed to be invoked either:
- via `noetic_ingest.cli check-structure` (recommended), or
- directly: `python -m noetic_ingest.pipeline.check_structure ...`
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None  # type: ignore

from core import config_loader

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    # Keep consistent with other outputs (UTC + Z suffix).
    return datetime.utcnow().isoformat() + "Z"


def _as_dict(m: Any) -> dict:
    return dict(m) if isinstance(m, Mapping) else {}


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _looks_fmri_summary(summary: Mapping[str, Any]) -> bool:
    # Heuristic: summary keys that are emitted only by fMRI paths.
    for key in ("fmri_modularity_provisional_frac",):
        if key in summary:
            return True
    # Also accept explicit extension marker if present
    ext = summary.get("extensions", {})
    if isinstance(ext, Mapping):
        # some runs expose regional flags there
        if "regional_data_available" in ext:
            return True
    return False


def _find_recording_h5(recording_dir: Path, require_single: bool = True) -> Tuple[Optional[Path], List[str]]:
    issues: List[str] = []
    h5s = sorted(recording_dir.glob("*.h5"))
    if not h5s:
        issues.append("missing_h5")
        return None, issues
    if require_single and len(h5s) != 1:
        issues.append(f"expected_single_h5_found_{len(h5s)}")
    return h5s[0], issues


def _check_h5_paths(h5: h5py.File, required_paths: List[str]) -> List[str]:
    issues: List[str] = []
    for p in required_paths:
        key = str(p).lstrip("/")
        if not key:
            continue
        if key not in h5:
            issues.append(f"missing_h5_path:{p}")
    return issues


def _check_h5_group_attrs(h5: h5py.File, group_path: str, attrs: List[str]) -> List[str]:
    issues: List[str] = []
    key = str(group_path).lstrip("/")
    if key not in h5:
        issues.append(f"missing_h5_path:{group_path}")
        return issues
    g = h5[key]
    for a in attrs:
        if a not in g.attrs:
            issues.append(f"missing_h5_attr:{group_path}:{a}")
    return issues


@dataclass
class RecordingCheck:
    recording_dir: str
    ok: bool
    modality: str
    issues: List[str]


@dataclass
class RunReport:
    dataset: str
    run_dir: str
    created_at: str
    ok: bool
    n_recordings: int
    n_ok: int
    recordings: List[RecordingCheck]


def check_run_dir(
    dataset_id: str,
    run_dir: Path,
    spec: Mapping[str, Any],
) -> RunReport:
    if h5py is None:
        raise RuntimeError("h5py is required for structure checks but is not installed.")
    spec_d = _as_dict(spec)
    scan_common = _as_dict(spec_d.get("common", {}))
    required_files = list(scan_common.get("required_files", []) or [])
    require_single_h5 = bool(scan_common.get("require_single_h5", True))

    eeg_spec = _as_dict(spec_d.get("eeg", {}))
    fmri_spec = _as_dict(spec_d.get("fmri", {}))

    eeg_enabled = bool(eeg_spec.get("enabled", True))
    fmri_enabled = bool(fmri_spec.get("enabled", True))

    eeg_required_paths = list(eeg_spec.get("h5_required_paths", []) or [])
    fmri_required_paths = list(fmri_spec.get("h5_required_paths", []) or [])

    fmri_require_regions = bool(fmri_spec.get("require_regions", False))
    fmri_regions_spec = _as_dict(fmri_spec.get("regions", {}))
    fmri_regions_required_paths = list(fmri_regions_spec.get("required_paths", []) or [])
    fmri_regions_required_attrs = _as_dict(fmri_regions_spec.get("required_attrs", {}))

    recordings: List[RecordingCheck] = []

    # Typical structure: <run_dir>/sub-XXX_.../{summary.json,qc_*.json,*.h5}
    rec_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")]
    rec_dirs = sorted(rec_dirs, key=lambda p: p.name)

    for rd in rec_dirs:
        issues: List[str] = []

        # Required files
        for fname in required_files:
            if not (rd / fname).exists():
                issues.append(f"missing_file:{fname}")

        # Load summary.json to infer modality (best-effort)
        summary = _load_json(rd / "summary.json") if (rd / "summary.json").exists() else {}
        modality = "fmri" if _looks_fmri_summary(summary) else "eeg"

        # H5 checks
        h5_path, h5_issues = _find_recording_h5(rd, require_single=require_single_h5)
        issues.extend(h5_issues)

        if h5_path is not None and h5_path.exists():
            try:
                with h5py.File(h5_path, "r") as h5:
                    # If the file contains /regions, treat as fMRI even if summary heuristic missed it.
                    if "regions" in h5 and modality != "fmri":
                        modality = "fmri"

                    if modality == "eeg":
                        if not eeg_enabled:
                            issues.append("eeg_disabled_in_spec")
                        else:
                            issues.extend(_check_h5_paths(h5, eeg_required_paths))
                    else:
                        if not fmri_enabled:
                            issues.append("fmri_disabled_in_spec")
                        else:
                            issues.extend(_check_h5_paths(h5, fmri_required_paths))
                            if fmri_require_regions:
                                issues.extend(_check_h5_paths(h5, fmri_regions_required_paths))
                                for grp, attrs in fmri_regions_required_attrs.items():
                                    issues.extend(_check_h5_group_attrs(h5, str(grp), list(attrs or [])))

                                # Optional: cross-check against summary.json extension marker
                                ext = summary.get("extensions", {})
                                if isinstance(ext, Mapping) and "regional_data_available" in ext:
                                    if not bool(ext.get("regional_data_available")):
                                        issues.append("summary_reports_no_regional_data_available")
            except Exception as exc:
                issues.append(f"h5_read_error:{type(exc).__name__}")

        ok = len(issues) == 0
        recordings.append(
            RecordingCheck(
                recording_dir=str(rd),
                ok=ok,
                modality=modality,
                issues=sorted(set(issues)),
            )
        )

    n_ok = sum(1 for r in recordings if r.ok)
    ok = n_ok == len(recordings) and len(recordings) > 0
    return RunReport(
        dataset=dataset_id,
        run_dir=str(run_dir),
        created_at=_utc_now_iso(),
        ok=ok,
        n_recordings=len(recordings),
        n_ok=n_ok,
        recordings=recordings,
    )


def _find_latest_run_dir(ds_processed_dir: Path) -> Optional[Path]:
    runs = [
        p
        for pattern in ("neuralmanifolddynamics_*", "mnps_*")
        for p in ds_processed_dir.glob(pattern)
        if p.is_dir()
    ]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def _find_all_run_dirs(ds_processed_dir: Path) -> List[Path]:
    runs = [
        p
        for pattern in ("neuralmanifolddynamics_*", "mnps_*")
        for p in ds_processed_dir.glob(pattern)
        if p.is_dir()
    ]
    return sorted(runs, key=lambda p: p.stat().st_mtime)


def run_structure_check(
    ingest_config: Mapping[str, Any],
    check_spec: Mapping[str, Any],
    dataset_ids: List[str],
    processed_base: Path,
    run_selector: str = "latest",
) -> Dict[str, Any]:
    """Run checks for datasets and return a JSON-serializable report."""
    _ = ingest_config  # reserved for future (dataset modality detection via file_index.csv)
    spec = _as_dict(check_spec)
    io_cfg = _as_dict(spec.get("io", {}))
    report_name = str(io_cfg.get("report_name", "structure_check_report.json"))

    out: Dict[str, Any] = {
        "created_at": _utc_now_iso(),
        "datasets": [],
        "ok": True,
        "run_selector": run_selector,
        "report_name": report_name,
    }

    for ds in dataset_ids:
        ds_dir = processed_base / ds
        if not ds_dir.exists():
            out["datasets"].append({"dataset": ds, "ok": False, "error": "missing_processed_dir"})
            out["ok"] = False
            continue

        if run_selector == "all":
            run_dirs = _find_all_run_dirs(ds_dir)
        else:
            latest = _find_latest_run_dir(ds_dir)
            run_dirs = [latest] if latest is not None else []

        if not run_dirs:
            out["datasets"].append({"dataset": ds, "ok": False, "error": "no_mnps_runs_found"})
            out["ok"] = False
            continue

        ds_reports: List[dict] = []
        for rd in run_dirs:
            rep = check_run_dir(ds, rd, spec)
            rep_d = {
                "dataset": rep.dataset,
                "run_dir": rep.run_dir,
                "created_at": rep.created_at,
                "ok": rep.ok,
                "n_recordings": rep.n_recordings,
                "n_ok": rep.n_ok,
                "recordings": [
                    {
                        "recording_dir": r.recording_dir,
                        "ok": r.ok,
                        "modality": r.modality,
                        "issues": r.issues,
                    }
                    for r in rep.recordings
                ],
            }
            ds_reports.append(rep_d)

            # Write per-run JSON report next to the run_dir
            try:
                (Path(rd) / report_name).write_text(json.dumps(rep_d, indent=2), encoding="utf-8")
            except Exception:
                logger.exception("Failed to write report to %s", Path(rd) / report_name)

            if not rep.ok:
                out["ok"] = False

        out["datasets"].append({"dataset": ds, "ok": all(r.get("ok") for r in ds_reports), "runs": ds_reports})

    return out


def build_parser(argv: Optional[List[str]] = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check structure of summarized outputs")
    p.add_argument("--config", type=Path, required=True, help="Path to ingest config_ingest.yaml")
    p.add_argument("--check-config", type=Path, required=True, help="Path to check_structure.yaml")
    p.add_argument("--dataset", nargs="*", default=None, help="Dataset id(s) to check (default: all from ingest config)")
    p.add_argument("--processed-dir", type=Path, default=None, help="Override processed base dir (default: from ingest config paths.processed_dir)")
    p.add_argument("--run-selector", choices=["latest", "all"], default=None, help="Which summarized run(s) to check (default: from check-config)")
    p.add_argument("--out", type=Path, default=None, help="Write a combined JSON report to this path (optional)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser(argv).parse_args(argv)

    ingest_cfg = config_loader.load_config(args.config)
    check_cfg = config_loader.load_config(args.check_config)

    dataset_ids = args.dataset or list(ingest_cfg.get("datasets", []) or [])
    processed_base = args.processed_dir or Path(_as_dict(ingest_cfg.get("paths", {})).get("processed_dir", "."))

    # default run selector from check-config
    run_selector = args.run_selector
    if not run_selector:
        scan_cfg = _as_dict(check_cfg.get("scan", {}))
        run_selector = str(scan_cfg.get("run_selector", "latest"))
    if run_selector not in {"latest", "all"}:
        run_selector = "latest"

    report = run_structure_check(
        ingest_config=ingest_cfg,
        check_spec=check_cfg,
        dataset_ids=list(dataset_ids),
        processed_base=Path(processed_base),
        run_selector=run_selector,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote combined report: %s", args.out)

    # Return non-zero on any failure
    return 0 if bool(report.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())


