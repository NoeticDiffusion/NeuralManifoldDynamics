from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("vitaldb_ingest")


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    return cfg


def _contains_any_keyword(value: str, keywords: list[str]) -> bool:
    v = value.lower()
    return any(k.lower() in v for k in keywords)


def _sanitize_column_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "track"


def _select_case_ids(
    trks: pd.DataFrame,
    eeg_keywords: list[str],
    propofol_keywords: list[str],
    max_cases: int,
    random_seed: int,
) -> list[int]:
    required_cols = {"caseid", "tname"}
    missing = required_cols - set(trks.columns)
    if missing:
        raise ValueError(f"VitalDB tracks table is missing required columns: {sorted(missing)}")

    trks = trks.copy()
    trks["tname"] = trks["tname"].astype(str)
    trks["caseid"] = trks["caseid"].astype(int)

    eeg_cases = set(
        trks.loc[trks["tname"].map(lambda x: _contains_any_keyword(x, eeg_keywords)), "caseid"].unique().tolist()
    )
    prop_cases = set(
        trks.loc[trks["tname"].map(lambda x: _contains_any_keyword(x, propofol_keywords)), "caseid"].unique().tolist()
    )
    eligible = sorted(eeg_cases & prop_cases)
    if not eligible:
        raise RuntimeError("No cases found that satisfy EEG + propofol track filters.")

    rng = random.Random(random_seed)
    rng.shuffle(eligible)
    return eligible[:max_cases]


def _pick_tracks_for_case(
    trks_case: pd.DataFrame,
    eeg_keywords: list[str],
    propofol_keywords: list[str],
) -> list[str]:
    names = trks_case["tname"].astype(str).tolist()
    eeg = [n for n in names if _contains_any_keyword(n, eeg_keywords)]
    prop = [n for n in names if _contains_any_keyword(n, propofol_keywords)]
    # Keep only required tracks for a compact download artifact.
    return sorted(set(eeg + prop))


def _select_case_track_map_via_find_cases(
    vitaldb_module: Any,
    eeg_track_names: list[str],
    propofol_track_names: list[str],
    max_cases: int,
    random_seed: int,
    min_eeg_tracks_per_case: int = 1,
    include_case_ids: list[int] | None = None,
    exclude_case_ids: list[int] | None = None,
) -> dict[int, list[str]]:
    """Fallback for vitaldb versions without load_trks().

    This mode treats configured keywords as exact track names.
    """
    eeg_case_map: dict[int, set[str]] = {}
    prop_case_map: dict[int, set[str]] = {}

    for eeg_name in eeg_track_names:
        case_ids = vitaldb_module.find_cases(eeg_name)
        for case_id in case_ids or []:
            cid = int(case_id)
            eeg_case_map.setdefault(cid, set()).add(eeg_name)

    for prop_name in propofol_track_names:
        case_ids = vitaldb_module.find_cases(prop_name)
        for case_id in case_ids or []:
            cid = int(case_id)
            prop_case_map.setdefault(cid, set()).add(prop_name)

    include_set = set(include_case_ids or [])
    exclude_set = set(exclude_case_ids or [])

    eligible = sorted(set(eeg_case_map.keys()) & set(prop_case_map.keys()))
    if include_set:
        eligible = [cid for cid in eligible if cid in include_set]
    if exclude_set:
        eligible = [cid for cid in eligible if cid not in exclude_set]
    eligible = [cid for cid in eligible if len(eeg_case_map.get(cid, set())) >= int(min_eeg_tracks_per_case)]
    if not eligible:
        raise RuntimeError(
            "No cases found in find_cases() fallback mode. "
            "Use exact track names in selection.eeg_track_keywords and propofol_track_keywords."
        )
    rng = random.Random(random_seed)
    rng.shuffle(eligible)
    chosen = eligible[:max_cases]
    out: dict[int, list[str]] = {}
    for cid in chosen:
        eeg_tracks = sorted(eeg_case_map.get(cid, set()))
        prop_tracks = sorted(prop_case_map.get(cid, set()))
        # Download all matching EEG tracks and a single propofol track.
        selected = eeg_tracks + (prop_tracks[:1] if prop_tracks else [])
        out[cid] = selected
    return out


def _download_case(
    case_id: int,
    track_names: list[str],
    interval_seconds: float,
    retries: int,
    sleep_seconds: float,
    vitaldb_module: Any,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            values = vitaldb_module.load_case(case_id, track_names, interval=interval_seconds)
            arr = np.asarray(values)
            if arr.ndim != 2:
                raise RuntimeError(f"Unexpected array shape for case {case_id}: {arr.shape}")
            if arr.shape[1] != len(track_names):
                raise RuntimeError(
                    f"Returned columns ({arr.shape[1]}) do not match requested tracks ({len(track_names)}) "
                    f"for case {case_id}"
                )
            df = pd.DataFrame(arr, columns=track_names)
            df.insert(0, "time_s", np.arange(len(df)) * interval_seconds)
            return df
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Case %s attempt %s/%s failed: %s", case_id, attempt, retries, exc)
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed downloading case {case_id}: {last_error}")


def _download_raw_vital_file(
    case_id: int,
    retries: int,
    sleep_seconds: float,
    vitaldb_module: Any,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    local_name = f"{int(case_id):04d}.vital"
    out_path = out_dir / local_name
    if out_path.exists():
        return out_path

    last_error: Exception | None = None
    remote_name = local_name
    for attempt in range(1, retries + 1):
        try:
            vitaldb_module.download(remote_name, localpath=str(out_dir))
            if out_path.exists():
                return out_path
            raise RuntimeError(f"download() returned but file not found at {out_path}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Raw .vital case %s attempt %s/%s failed: %s", case_id, attempt, retries, exc)
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed raw .vital download for case {case_id}: {last_error}")


def run_download(config_path: Path) -> int:
    try:
        import vitaldb  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("vitaldb package is required. Install with: pip install vitaldb") from exc

    cfg = _load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    sel_cfg = cfg.get("selection", {})
    dl_cfg = cfg.get("download", {})
    export_cfg = cfg.get("export", {})
    auth_cfg = cfg.get("auth", {})

    received_dir = Path(paths_cfg.get("received_dir", "E:/Science_Datasets/vitaldb/received"))
    metadata_dir = Path(paths_cfg.get("metadata_dir", "E:/Science_Datasets/vitaldb/metadata"))
    cases_dir = received_dir / "cases"
    vital_files_dir = received_dir / "vital_files"
    meta_out = metadata_dir
    cases_dir.mkdir(parents=True, exist_ok=True)
    vital_files_dir.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    eeg_keywords = list(sel_cfg.get("eeg_track_keywords", ["EEG"]))
    prop_keywords = list(sel_cfg.get("propofol_track_keywords", ["PROPOFOL"]))
    max_cases = int(sel_cfg.get("max_cases", 100))
    random_seed = int(sel_cfg.get("random_seed", 42))
    min_eeg_tracks_per_case = int(sel_cfg.get("min_eeg_tracks_per_case", 1))
    include_case_ids_raw = sel_cfg.get("include_case_ids", []) if isinstance(sel_cfg, dict) else []
    exclude_case_ids_raw = sel_cfg.get("exclude_case_ids", []) if isinstance(sel_cfg, dict) else []
    include_case_ids = [int(x) for x in include_case_ids_raw] if isinstance(include_case_ids_raw, list) else []
    exclude_case_ids = [int(x) for x in exclude_case_ids_raw] if isinstance(exclude_case_ids_raw, list) else []

    interval_seconds = float(dl_cfg.get("interval_seconds", 0.5))
    retries = int(dl_cfg.get("retries", 3))
    sleep_seconds = float(dl_cfg.get("request_sleep_seconds", 0.2))

    include_case_metadata_csv = bool(export_cfg.get("include_case_metadata_csv", True))
    include_manifest_jsonl = bool(export_cfg.get("include_manifest_jsonl", True))
    file_format = str(export_cfg.get("file_format", "csv.gz")).lower()
    export_csv = file_format == "csv.gz"
    if file_format not in ("csv.gz", "none"):
        raise ValueError("Supported export.file_format values: csv.gz, none")
    raw_vital_cfg = export_cfg.get("raw_vital", {}) if isinstance(export_cfg, dict) else {}
    export_raw_vital = bool(raw_vital_cfg.get("enabled", False))
    if not export_csv and not export_raw_vital:
        raise ValueError("No output configured. Enable csv.gz and/or export.raw_vital.enabled")

    id_env = str(auth_cfg.get("id_env_var", "VITALDB_ID")) if isinstance(auth_cfg, dict) else "VITALDB_ID"
    pw_env = str(auth_cfg.get("pw_env_var", "VITALDB_PW")) if isinstance(auth_cfg, dict) else "VITALDB_PW"
    vital_id = os.getenv(id_env, "")
    vital_pw = os.getenv(pw_env, "")
    if export_raw_vital:
        if not vital_id or not vital_pw:
            raise RuntimeError(
                f"Raw .vital export requires credentials. Set environment vars: {id_env}, {pw_env}"
            )
        try:
            vitaldb.login(id=vital_id, pw=vital_pw)
            logger.info("Authenticated to VitalDB for raw .vital download.")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"VitalDB login failed: {exc}") from exc

    trks: pd.DataFrame | None = None
    fallback_track_map: dict[int, list[str]] = {}
    if hasattr(vitaldb, "load_trks"):
        logger.info("Loading VitalDB tracks metadata via load_trks()...")
        trks = vitaldb.load_trks()
        if not isinstance(trks, pd.DataFrame):
            trks = pd.DataFrame(trks)
        selected_case_ids = _select_case_ids(
            trks=trks,
            eeg_keywords=eeg_keywords,
            propofol_keywords=prop_keywords,
            max_cases=max_cases,
            random_seed=random_seed,
        )
    else:
        logger.info("load_trks() not available; using find_cases() fallback.")
        fallback_track_map = _select_case_track_map_via_find_cases(
            vitaldb_module=vitaldb,
            eeg_track_names=eeg_keywords,
            propofol_track_names=prop_keywords,
            max_cases=max_cases,
            random_seed=random_seed,
            min_eeg_tracks_per_case=min_eeg_tracks_per_case,
            include_case_ids=include_case_ids,
            exclude_case_ids=exclude_case_ids,
        )
        selected_case_ids = sorted(fallback_track_map.keys())

    logger.info("Selected %d case(s). Starting download...", len(selected_case_ids))
    manifest_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for idx, case_id in enumerate(selected_case_ids, start=1):
        if trks is not None:
            case_trks = trks.loc[trks["caseid"].astype(int) == int(case_id)].copy()
            track_names = _pick_tracks_for_case(case_trks, eeg_keywords, prop_keywords)
        else:
            track_names = fallback_track_map.get(int(case_id), [])
        if not track_names:
            logger.warning("Skipping case %s due to empty track selection.", case_id)
            continue

        logger.info("[%d/%d] Downloading case %s (%d tracks)", idx, len(selected_case_ids), case_id, len(track_names))

        row: dict[str, Any] = {
            "case_id": int(case_id),
            "track_count": len(track_names),
            "tracks": track_names,
            "status": "ok",
            "error": "",
        }
        try:
            if export_csv:
                df_case = _download_case(
                    case_id=int(case_id),
                    track_names=track_names,
                    interval_seconds=interval_seconds,
                    retries=retries,
                    sleep_seconds=sleep_seconds,
                    vitaldb_module=vitaldb,
                )
                renamed = {
                    c: _sanitize_column_name(c) if c != "time_s" else c
                    for c in df_case.columns
                }
                df_case = df_case.rename(columns=renamed)
                out_file = cases_dir / f"case-{int(case_id):06d}.csv.gz"
                df_case.to_csv(out_file, index=False, compression="gzip")
                row["output_file_csv"] = str(out_file)
                row["n_samples"] = int(len(df_case))

            if export_raw_vital:
                vital_out = _download_raw_vital_file(
                    case_id=int(case_id),
                    retries=retries,
                    sleep_seconds=sleep_seconds,
                    vitaldb_module=vitaldb,
                    out_dir=vital_files_dir,
                )
                row["output_file_vital"] = str(vital_out)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)
            logger.error("Case %s failed: %s", case_id, exc)

        manifest_rows.append(row)
        metadata_rows.append(
            {
                "case_id": int(case_id),
                "n_tracks_selected": len(track_names),
                "tracks_selected": ";".join(track_names),
            }
        )
        time.sleep(sleep_seconds)

    if include_case_metadata_csv:
        df_meta = pd.DataFrame(metadata_rows)
        df_meta.to_csv(meta_out / "case_manifest.csv", index=False)

    if include_manifest_jsonl:
        manifest_path = meta_out / "download_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for item in manifest_rows:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")

    n_ok = sum(1 for r in manifest_rows if r.get("status") == "ok")
    n_err = sum(1 for r in manifest_rows if r.get("status") == "error")
    logger.info("Finished. Success: %d, Failed: %d", n_ok, n_err)
    return 0 if n_ok > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download EEG + propofol VitalDB cases")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "vitaldb_config.yml",
        help="Path to VitalDB ingest config YAML",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_download(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
