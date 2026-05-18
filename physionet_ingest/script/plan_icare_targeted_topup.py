from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from physionet_ingest.script import download_physionet as dl

LOGGER = logging.getLogger("physionet_targeted_topup")

_CPC_RE = re.compile(r"(?im)^\s*CPC\s*:\s*([^\r\n]+)")
_OUTCOME_RE = re.compile(r"(?im)^\s*Outcome\s*:\s*([^\r\n]+)")


def _parse_patient_metadata_text(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    cpc_match = _CPC_RE.search(text)
    if cpc_match:
        raw = cpc_match.group(1).strip()
        number_match = re.search(r"\d+", raw)
        if number_match:
            value = int(number_match.group(0))
            if 1 <= value <= 5:
                parsed["cpc"] = value
                parsed["cpc_raw"] = raw
    outcome_match = _OUTCOME_RE.search(text)
    if outcome_match:
        raw = outcome_match.group(1).strip().lower()
        if "good" in raw:
            parsed["outcome"] = "good"
        elif "poor" in raw:
            parsed["outcome"] = "poor"
        else:
            parsed["outcome"] = raw
        parsed["outcome_raw"] = outcome_match.group(1).strip()
    return parsed


def _read_local_training_metadata(training_root: Path) -> dict[str, dict[str, Any]]:
    if not training_root.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for subject_dir in sorted(path for path in training_root.iterdir() if path.is_dir() and path.name.isdigit()):
        candidate_files = [subject_dir / f"{subject_dir.name}.txt"]
        candidate_files.extend(sorted(subject_dir.glob("*.txt")))
        chosen_path: Path | None = None
        chosen_meta: dict[str, Any] | None = None
        for candidate in candidate_files:
            if not candidate.exists():
                continue
            try:
                content = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            parsed = _parse_patient_metadata_text(content)
            if parsed:
                chosen_path = candidate
                chosen_meta = parsed
                break
        if chosen_path is not None and chosen_meta is not None:
            chosen_meta = dict(chosen_meta)
            chosen_meta["source"] = "local"
            chosen_meta["metadata_path"] = str(chosen_path)
            result[subject_dir.name] = chosen_meta
    return result


def _resolve_patient_metadata_remote_path(patient_id: str, checksums_lookup: dict[str, str]) -> str | None:
    if not checksums_lookup:
        return f"training/{patient_id}/{patient_id}.txt"
    exact = f"training/{patient_id}/{patient_id}.txt"
    if exact in checksums_lookup:
        return exact
    prefix = f"training/{patient_id}/"
    candidates = [
        path for path in checksums_lookup.keys() if path.startswith(prefix) and path.lower().endswith(".txt")
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda path: (
            0 if PurePosixPath(path).name.startswith(patient_id) else 1,
            len(path),
            path,
        )
    )
    return candidates[0]


def _collect_remote_candidate_metadata(
    *,
    candidate_patient_ids: list[str],
    checksums_lookup: dict[str, str],
    output_dir: Path,
    dataset_root_url: str,
    timeout_seconds: float,
    retries: int,
    retry_backoff_seconds: float,
    user_agent: str,
) -> dict[str, dict[str, Any]]:
    collected: dict[str, dict[str, Any]] = {}
    total = len(candidate_patient_ids)
    for idx, patient_id in enumerate(candidate_patient_ids, start=1):
        metadata_rel = _resolve_patient_metadata_remote_path(patient_id, checksums_lookup)
        if metadata_rel is None:
            continue
        local_candidate = output_dir.joinpath(*PurePosixPath(metadata_rel).parts)
        try:
            if local_candidate.exists():
                text = local_candidate.read_text(encoding="utf-8", errors="ignore")
            else:
                metadata_url = dl.build_remote_file_url(dataset_root_url, metadata_rel, with_download_query=True)
                text = dl.fetch_text_with_retries(
                    url=metadata_url,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    user_agent=user_agent,
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not fetch metadata sidecar for patient %s: %s", patient_id, exc)
            continue

        parsed = _parse_patient_metadata_text(text)
        if not parsed:
            continue
        parsed = dict(parsed)
        parsed["source"] = "remote_txt"
        parsed["metadata_path"] = metadata_rel
        collected[patient_id] = parsed

        if idx % 100 == 0 or idx == total:
            LOGGER.info("Remote metadata scan progress: %s/%s", idx, total)
    return collected


def _counter_from_metadata(
    metadata_by_patient: dict[str, dict[str, Any]],
) -> tuple[Counter[int], Counter[str]]:
    cpc_counts: Counter[int] = Counter()
    outcome_counts: Counter[str] = Counter()
    for meta in metadata_by_patient.values():
        cpc = meta.get("cpc")
        outcome = meta.get("outcome")
        if isinstance(cpc, int):
            cpc_counts[cpc] += 1
        if isinstance(outcome, str):
            outcome_counts[outcome] += 1
    return cpc_counts, outcome_counts


def _compute_deficits(current: Counter[Any], targets: dict[Any, int]) -> dict[Any, int]:
    deficits: dict[Any, int] = {}
    for key, target in targets.items():
        deficits[key] = max(0, int(target) - int(current.get(key, 0)))
    return deficits


def _plan_targeted_selection(
    *,
    current_cpc: Counter[int],
    current_outcome: Counter[str],
    candidates: dict[str, dict[str, Any]],
    target_cpc: dict[int, int],
    target_outcome: dict[str, int],
    max_extra_patients: int,
) -> tuple[list[str], Counter[int], Counter[str]]:
    working_cpc = Counter(current_cpc)
    working_outcome = Counter(current_outcome)
    selected: list[str] = []
    remaining_ids = sorted(candidates.keys())

    while remaining_ids and len(selected) < max_extra_patients:
        cpc_deficits = _compute_deficits(working_cpc, target_cpc)
        outcome_deficits = _compute_deficits(working_outcome, target_outcome)
        if all(value <= 0 for value in cpc_deficits.values()) and all(
            value <= 0 for value in outcome_deficits.values()
        ):
            break

        best_id: str | None = None
        best_score = 0

        for patient_id in remaining_ids:
            meta = candidates[patient_id]
            score = 0

            cpc = meta.get("cpc")
            if isinstance(cpc, int):
                deficit = target_cpc.get(cpc, 0) - int(working_cpc.get(cpc, 0))
                if deficit > 0:
                    score += 1000 + deficit * 25

            outcome = meta.get("outcome")
            if isinstance(outcome, str):
                deficit = target_outcome.get(outcome, 0) - int(working_outcome.get(outcome, 0))
                if deficit > 0:
                    score += 300 + deficit * 10

            if score > best_score:
                best_score = score
                best_id = patient_id

        if best_id is None or best_score <= 0:
            break

        selected.append(best_id)
        selected_meta = candidates[best_id]
        selected_cpc = selected_meta.get("cpc")
        selected_outcome = selected_meta.get("outcome")
        if isinstance(selected_cpc, int):
            working_cpc[selected_cpc] += 1
        if isinstance(selected_outcome, str):
            working_outcome[selected_outcome] += 1
        remaining_ids.remove(best_id)

    return selected, working_cpc, working_outcome


def _write_targeted_config(
    *,
    merged_cfg: dict[str, Any],
    selected_patient_ids: list[str],
    output_config_path: Path,
) -> None:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to write config output.") from exc

    dataset_cfg = dict(merged_cfg.get("dataset", {}))
    subset_cfg = dict(merged_cfg.get("subset", {}))
    download_cfg = dict(merged_cfg.get("download", {}))

    output_doc: dict[str, Any] = {
        "dataset": {
            "slug": dataset_cfg.get("slug", "i-care"),
            "version": dataset_cfg.get("version", "2.1"),
            "base_url": dataset_cfg.get("base_url", "https://physionet.org"),
            "remote_root": dataset_cfg.get("remote_root", "files/i-care/2.1"),
            "records_file": dataset_cfg.get("records_file", "RECORDS"),
            "checksums_file": dataset_cfg.get("checksums_file", "SHA256SUMS.txt"),
        },
        "subset": {
            "strategy": "explicit_patient_ids",
            "patient_ids": selected_patient_ids,
            "include_file_globs": subset_cfg.get("include_file_globs", ["*_EEG.hea", "*_EEG.mat", "*.txt"]),
            "include_top_level_files": subset_cfg.get(
                "include_top_level_files", ["LICENSE.txt", "RECORDS", "SHA256SUMS.txt"]
            ),
        },
        "download": {
            "output_subdir": download_cfg.get("output_subdir", "i-care_2_1_random100_longitudinal"),
            "max_parallel_downloads": download_cfg.get("max_parallel_downloads", 6),
            "dry_run": True,
        },
    }

    if "min_eeg_hours_per_patient" in subset_cfg:
        output_doc["subset"]["min_eeg_hours_per_patient"] = subset_cfg["min_eeg_hours_per_patient"]
    if "max_eeg_hours_per_patient" in subset_cfg:
        output_doc["subset"]["max_eeg_hours_per_patient"] = subset_cfg["max_eeg_hours_per_patient"]
    if "expand_record_extensions" in subset_cfg:
        output_doc["subset"]["expand_record_extensions"] = subset_cfg["expand_record_extensions"]

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(yaml.safe_dump(output_doc, sort_keys=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan a targeted I-CARE top-up cohort by CPC/outcome from remote metadata "
            "and emit an explicit-patient downloader config."
        )
    )
    parser.add_argument(
        "--config-general",
        type=Path,
        default=Path("physionet_ingest/config/config_ingest.yml"),
        help="General ingest config path.",
    )
    parser.add_argument(
        "--config-dataset",
        type=Path,
        default=Path("physionet_ingest/config/config_i-care_2_1_longitudinal.yml"),
        help="Base dataset config path to inherit settings from.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("physionet_ingest/config/config_i-care_2_1_longitudinal_targeted_topup.yml"),
        help="Generated explicit-patient top-up config path.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional JSON report path (defaults to metadata_root/<output_subdir>/targeted_topup_plan.json).",
    )
    parser.add_argument("--target-cpc1", type=int, default=60, help="Target minimum count for CPC 1.")
    parser.add_argument("--target-cpc2", type=int, default=0, help="Target minimum count for CPC 2.")
    parser.add_argument("--target-cpc3", type=int, default=0, help="Target minimum count for CPC 3.")
    parser.add_argument("--target-cpc4", type=int, default=0, help="Target minimum count for CPC 4.")
    parser.add_argument("--target-cpc5", type=int, default=66, help="Target minimum count for CPC 5.")
    parser.add_argument("--target-good", type=int, default=None, help="Optional target minimum for good outcome.")
    parser.add_argument("--target-poor", type=int, default=None, help="Optional target minimum for poor outcome.")
    parser.add_argument(
        "--max-extra-patients",
        type=int,
        default=120,
        help="Hard cap on selected extra patients.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.max_extra_patients <= 0:
        parser.error("--max-extra-patients must be > 0")

    config_general = dl.load_yaml_config(args.config_general)
    config_dataset = dl.load_yaml_config(args.config_dataset)
    cfg = dl.deep_merge_dict(config_general, config_dataset)

    paths_cfg = cfg.get("paths", {})
    network_cfg = cfg.get("network", {})
    dataset_cfg = cfg.get("dataset", {})
    download_cfg = cfg.get("download", {})

    timeout_seconds = float(network_cfg.get("timeout_seconds", 60))
    retries = int(network_cfg.get("retries", 3))
    retry_backoff_seconds = float(network_cfg.get("retry_backoff_seconds", 2.0))
    user_agent = str(network_cfg.get("user_agent", "NeuralManifoldDynamics-PhysioNetIngest/1.0"))

    base_url = str(dataset_cfg.get("base_url", "https://physionet.org"))
    remote_root = str(dataset_cfg.get("remote_root", "files/i-care/2.1"))
    records_file = str(dataset_cfg.get("records_file", "RECORDS"))
    checksums_file = str(dataset_cfg.get("checksums_file", "SHA256SUMS.txt"))
    dataset_root_url = dl.build_dataset_root_url(base_url, remote_root)

    output_subdir = str(download_cfg.get("output_subdir", "i-care_2_1_random100_longitudinal"))
    download_root = Path(str(paths_cfg.get("download_root", "E:/Science_Datasets/physionet/received")))
    metadata_root = Path(str(paths_cfg.get("metadata_root", "E:/Science_Datasets/physionet/metadata")))
    output_dir = download_root / output_subdir
    metadata_dir = metadata_root / output_subdir
    training_root = output_dir / "training"

    report_path = args.output_report or (metadata_dir / "targeted_topup_plan.json")

    LOGGER.info("Reading local downloaded metadata from: %s", training_root)
    local_metadata = _read_local_training_metadata(training_root)
    LOGGER.info("Local patients with CPC/Outcome metadata: %s", len(local_metadata))

    records_cache_path = output_dir.joinpath(*PurePosixPath(dl.normalize_relative_path(records_file)).parts)
    if records_cache_path.exists() and records_cache_path.stat().st_size > 0:
        records_text = records_cache_path.read_text(encoding="utf-8")
    else:
        records_url = dl.build_remote_file_url(dataset_root_url, records_file, with_download_query=True)
        records_text = dl.fetch_text_with_retries(
            url=records_url,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            user_agent=user_agent,
        )
    record_paths = dl.parse_records_text(records_text)
    patient_pool = dl.collect_patients_in_order(record_paths)

    checksums_lookup: dict[str, str] = {}
    checksums_cache_path = output_dir.joinpath(*PurePosixPath(dl.normalize_relative_path(checksums_file)).parts)
    if checksums_cache_path.exists() and checksums_cache_path.stat().st_size > 0:
        checksums_text = checksums_cache_path.read_text(encoding="utf-8")
        checksums_lookup = dl.build_checksum_lookup(dl.parse_sha256sums_text(checksums_text), remote_root=remote_root)
        LOGGER.info("Using local checksum cache for metadata path resolution.")
    else:
        LOGGER.warning(
            "No local checksum cache found at %s. Falling back to default sidecar path pattern training/<id>/<id>.txt.",
            checksums_cache_path,
        )

    missing_pool_ids = [patient_id for patient_id in patient_pool if patient_id not in local_metadata]
    LOGGER.info("Patients in pool not currently represented locally: %s", len(missing_pool_ids))
    remote_metadata = _collect_remote_candidate_metadata(
        candidate_patient_ids=missing_pool_ids,
        checksums_lookup=checksums_lookup,
        output_dir=output_dir,
        dataset_root_url=dataset_root_url,
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        user_agent=user_agent,
    )
    LOGGER.info("Remote candidates with parsable CPC/Outcome metadata: %s", len(remote_metadata))

    current_cpc, current_outcome = _counter_from_metadata(local_metadata)
    target_cpc = {
        1: int(args.target_cpc1),
        2: int(args.target_cpc2),
        3: int(args.target_cpc3),
        4: int(args.target_cpc4),
        5: int(args.target_cpc5),
    }
    target_outcome: dict[str, int] = {}
    if args.target_good is not None:
        target_outcome["good"] = int(args.target_good)
    if args.target_poor is not None:
        target_outcome["poor"] = int(args.target_poor)

    selected_ids, projected_cpc, projected_outcome = _plan_targeted_selection(
        current_cpc=current_cpc,
        current_outcome=current_outcome,
        candidates=remote_metadata,
        target_cpc=target_cpc,
        target_outcome=target_outcome,
        max_extra_patients=int(args.max_extra_patients),
    )

    _write_targeted_config(
        merged_cfg=cfg,
        selected_patient_ids=selected_ids,
        output_config_path=args.output_config,
    )

    report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_output_subdir": output_subdir,
        "pool_patient_count": len(patient_pool),
        "local_patient_count_with_metadata": len(local_metadata),
        "missing_pool_patient_count": len(missing_pool_ids),
        "remote_candidate_count": len(remote_metadata),
        "targets": {
            "cpc": target_cpc,
            "outcome": target_outcome,
            "max_extra_patients": int(args.max_extra_patients),
        },
        "current_counts": {
            "cpc": {str(key): int(value) for key, value in sorted(current_cpc.items())},
            "outcome": {str(key): int(value) for key, value in sorted(current_outcome.items())},
        },
        "projected_counts": {
            "cpc": {str(key): int(value) for key, value in sorted(projected_cpc.items())},
            "outcome": {str(key): int(value) for key, value in sorted(projected_outcome.items())},
        },
        "deficits_before": {
            "cpc": {str(key): int(value) for key, value in _compute_deficits(current_cpc, target_cpc).items()},
            "outcome": {
                str(key): int(value) for key, value in _compute_deficits(current_outcome, target_outcome).items()
            },
        },
        "deficits_after": {
            "cpc": {str(key): int(value) for key, value in _compute_deficits(projected_cpc, target_cpc).items()},
            "outcome": {
                str(key): int(value) for key, value in _compute_deficits(projected_outcome, target_outcome).items()
            },
        },
        "selected_patient_ids_count": len(selected_ids),
        "selected_patient_ids_preview": selected_ids[:50],
        "selected_patient_ids": selected_ids,
        "output_config_path": str(args.output_config),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Targeted config written: {args.output_config}")
    print(f"Plan report written: {report_path}")
    print(f"Selected extra patients: {len(selected_ids)}")
    if selected_ids:
        print(f"First selected IDs: {', '.join(selected_ids[:15])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
