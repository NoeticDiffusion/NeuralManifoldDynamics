from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from physionet_ingest.script import download_physionet as dl

LOGGER = logging.getLogger("physionet_balanced_cpc_cohort")

_KV_RE = re.compile(r"^\s*([^:#]+?)\s*:\s*(.*?)\s*$")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "none", "null"}:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _to_bool_token(value: Any) -> str:
    if value is None:
        return "missing"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return "true"
    if text in {"false", "0", "no", "n"}:
        return "false"
    return "missing"


def _normalize_sex(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if text.startswith("m"):
        return "male"
    if text.startswith("f"):
        return "female"
    return "unknown"


def _normalize_hospital(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().upper()
    return text or "unknown"


def _normalize_ttm(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        text = str(value).strip().lower() if value is not None else ""
        if not text or text in {"nan", "na", "none", "null"}:
            return "none"
        return "other"
    if abs(number - 33.0) < 0.2:
        return "33"
    if abs(number - 36.0) < 0.2:
        return "36"
    return "other"


def _parse_metadata_text(text: str) -> dict[str, Any]:
    raw: dict[str, str] = {}
    for line in text.splitlines():
        match = _KV_RE.match(line)
        if match is None:
            continue
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        raw[key] = value

    cpc = _to_int(raw.get("cpc"))
    parsed: dict[str, Any] = {
        "cpc": cpc if cpc in {1, 2, 3, 4, 5} else None,
        "age": _to_float(raw.get("age")),
        "sex": _normalize_sex(raw.get("sex")),
        "hospital": _normalize_hospital(raw.get("hospital")),
        "ohca": _to_bool_token(raw.get("ohca")),
        "shockable": _to_bool_token(raw.get("shockable rhythm")),
        "ttm": _normalize_ttm(raw.get("ttm")),
        "rosc_min": _to_float(raw.get("rosc")),
        "outcome_raw": raw.get("outcome"),
    }
    return parsed


def _read_local_metadata(training_root: Path) -> dict[str, dict[str, Any]]:
    if not training_root.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    subject_dirs = sorted(path for path in training_root.iterdir() if path.is_dir() and path.name.isdigit())
    for subject_dir in subject_dirs:
        candidates = [subject_dir / f"{subject_dir.name}.txt"]
        candidates.extend(sorted(subject_dir.glob("*.txt")))
        chosen: dict[str, Any] | None = None
        chosen_path: Path | None = None
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            parsed = _parse_metadata_text(text)
            if parsed.get("cpc") is None:
                continue
            chosen = parsed
            chosen_path = candidate
            break
        if chosen is None or chosen_path is None:
            continue
        chosen = dict(chosen)
        chosen["source"] = "local"
        chosen["metadata_path"] = str(chosen_path)
        out[subject_dir.name] = chosen
    return out


def _resolve_metadata_path(patient_id: str, checksums_lookup: dict[str, str]) -> str | None:
    if not checksums_lookup:
        return f"training/{patient_id}/{patient_id}.txt"
    exact = f"training/{patient_id}/{patient_id}.txt"
    if exact in checksums_lookup:
        return exact
    prefix = f"training/{patient_id}/"
    candidates = [path for path in checksums_lookup.keys() if path.startswith(prefix) and path.lower().endswith(".txt")]
    if not candidates:
        return None
    candidates.sort(
        key=lambda value: (
            0 if PurePosixPath(value).name.startswith(patient_id) else 1,
            len(value),
            value,
        )
    )
    return candidates[0]


def _collect_remote_metadata(
    *,
    candidate_ids: list[str],
    checksums_lookup: dict[str, str],
    output_dir: Path,
    dataset_root_url: str,
    timeout_seconds: float,
    retries: int,
    retry_backoff_seconds: float,
    user_agent: str,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    total = len(candidate_ids)
    for idx, patient_id in enumerate(candidate_ids, start=1):
        metadata_rel = _resolve_metadata_path(patient_id, checksums_lookup)
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
            LOGGER.warning("Could not fetch metadata for patient %s: %s", patient_id, exc)
            continue

        parsed = _parse_metadata_text(text)
        if parsed.get("cpc") not in {1, 2, 3, 4, 5}:
            continue
        parsed = dict(parsed)
        parsed["source"] = "remote_txt"
        parsed["metadata_path"] = metadata_rel
        out[patient_id] = parsed

        if idx % 100 == 0 or idx == total:
            LOGGER.info("Remote metadata scan progress: %s/%s", idx, total)
    return out


def _l1_category_distance(values_a: list[str], values_b: list[str]) -> float:
    if not values_a or not values_b:
        return 1.0
    counter_a = Counter(values_a)
    counter_b = Counter(values_b)
    denom_a = float(len(values_a))
    denom_b = float(len(values_b))
    keys = set(counter_a.keys()) | set(counter_b.keys())
    return float(
        sum(abs(counter_a.get(key, 0) / denom_a - counter_b.get(key, 0) / denom_b) for key in sorted(keys))
    )


def _mean(values: list[float | None]) -> tuple[float | None, int]:
    finite = [value for value in values if value is not None]
    if not finite:
        return None, 0
    return float(sum(finite) / len(finite)), len(finite)


def _std(values: list[float | None]) -> float:
    finite = [value for value in values if value is not None]
    if len(finite) < 2:
        return 1.0
    mu = sum(finite) / len(finite)
    var = sum((value - mu) ** 2 for value in finite) / (len(finite) - 1)
    return max(1.0, math.sqrt(var))


def _balance_score(
    cpc1_rows: list[dict[str, Any]],
    cpc5_rows: list[dict[str, Any]],
    *,
    age_scale: float,
    rosc_scale: float,
) -> tuple[float, dict[str, float]]:
    age1, _ = _mean([row.get("age") for row in cpc1_rows])
    age5, _ = _mean([row.get("age") for row in cpc5_rows])
    age_term = abs((age1 or 0.0) - (age5 or 0.0)) / max(1.0, age_scale)

    rosc1, rosc_n1 = _mean([row.get("rosc_min") for row in cpc1_rows])
    rosc5, rosc_n5 = _mean([row.get("rosc_min") for row in cpc5_rows])
    rosc_term = abs((rosc1 or 0.0) - (rosc5 or 0.0)) / max(1.0, rosc_scale)
    rosc_missing_term = abs(1.0 - rosc_n1 / max(1, len(cpc1_rows)) - (1.0 - rosc_n5 / max(1, len(cpc5_rows))))

    sex_term = _l1_category_distance(
        [str(row.get("sex", "unknown")) for row in cpc1_rows],
        [str(row.get("sex", "unknown")) for row in cpc5_rows],
    )
    site_term = _l1_category_distance(
        [str(row.get("hospital", "unknown")) for row in cpc1_rows],
        [str(row.get("hospital", "unknown")) for row in cpc5_rows],
    )
    ohca_term = _l1_category_distance(
        [str(row.get("ohca", "missing")) for row in cpc1_rows],
        [str(row.get("ohca", "missing")) for row in cpc5_rows],
    )
    shock_term = _l1_category_distance(
        [str(row.get("shockable", "missing")) for row in cpc1_rows],
        [str(row.get("shockable", "missing")) for row in cpc5_rows],
    )
    ttm_term = _l1_category_distance(
        [str(row.get("ttm", "none")) for row in cpc1_rows],
        [str(row.get("ttm", "none")) for row in cpc5_rows],
    )

    score = (
        2.0 * age_term
        + 1.8 * site_term
        + 1.3 * sex_term
        + 1.2 * ohca_term
        + 1.2 * shock_term
        + 1.1 * ttm_term
        + 0.9 * rosc_term
        + 0.6 * rosc_missing_term
    )
    diagnostics = {
        "score_total": float(score),
        "age_term": float(age_term),
        "site_term": float(site_term),
        "sex_term": float(sex_term),
        "ohca_term": float(ohca_term),
        "shockable_term": float(shock_term),
        "ttm_term": float(ttm_term),
        "rosc_term": float(rosc_term),
        "rosc_missing_term": float(rosc_missing_term),
    }
    return float(score), diagnostics


def _sample_rows(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], set[str]]:
    chosen = rng.sample(rows, n)
    chosen_ids = {str(row["patient_id"]) for row in chosen}
    return chosen, chosen_ids


def _hill_climb_once(
    *,
    rows1_all: list[dict[str, Any]],
    rows5_all: list[dict[str, Any]],
    selected1: list[dict[str, Any]],
    selected5: list[dict[str, Any]],
    selected1_ids: set[str],
    selected5_ids: set[str],
    age_scale: float,
    rosc_scale: float,
    steps: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    current_score, current_diag = _balance_score(selected1, selected5, age_scale=age_scale, rosc_scale=rosc_scale)
    rows1_by_id = {str(row["patient_id"]): row for row in rows1_all}
    rows5_by_id = {str(row["patient_id"]): row for row in rows5_all}
    all1_ids = [str(row["patient_id"]) for row in rows1_all]
    all5_ids = [str(row["patient_id"]) for row in rows5_all]

    for _ in range(max(0, steps)):
        if rng.random() < 0.5:
            remove_id = rng.choice(tuple(selected1_ids))
            candidate_pool = [pid for pid in all1_ids if pid not in selected1_ids]
            if not candidate_pool:
                continue
            add_id = rng.choice(candidate_pool)

            trial1_ids = set(selected1_ids)
            trial1_ids.remove(remove_id)
            trial1_ids.add(add_id)
            trial1 = [rows1_by_id[pid] for pid in sorted(trial1_ids)]
            trial_score, trial_diag = _balance_score(trial1, selected5, age_scale=age_scale, rosc_scale=rosc_scale)
            if trial_score < current_score:
                selected1_ids = trial1_ids
                selected1 = trial1
                current_score = trial_score
                current_diag = trial_diag
        else:
            remove_id = rng.choice(tuple(selected5_ids))
            candidate_pool = [pid for pid in all5_ids if pid not in selected5_ids]
            if not candidate_pool:
                continue
            add_id = rng.choice(candidate_pool)

            trial5_ids = set(selected5_ids)
            trial5_ids.remove(remove_id)
            trial5_ids.add(add_id)
            trial5 = [rows5_by_id[pid] for pid in sorted(trial5_ids)]
            trial_score, trial_diag = _balance_score(selected1, trial5, age_scale=age_scale, rosc_scale=rosc_scale)
            if trial_score < current_score:
                selected5_ids = trial5_ids
                selected5 = trial5
                current_score = trial_score
                current_diag = trial_diag

    return selected1, selected5, current_diag


def _select_balanced_groups(
    *,
    rows1_all: list[dict[str, Any]],
    rows5_all: list[dict[str, Any]],
    n1: int,
    n5: int,
    seed: int,
    random_samples: int,
    hill_steps: int,
) -> tuple[list[str], list[str], dict[str, float]]:
    if len(rows1_all) < n1:
        raise ValueError(f"Not enough CPC1 candidates: need {n1}, available {len(rows1_all)}")
    if len(rows5_all) < n5:
        raise ValueError(f"Not enough CPC5 candidates: need {n5}, available {len(rows5_all)}")

    age_scale = _std([row.get("age") for row in rows1_all + rows5_all])
    rosc_scale = _std([row.get("rosc_min") for row in rows1_all + rows5_all])
    rng = random.Random(seed)

    best_score = float("inf")
    best_selection_1: list[dict[str, Any]] | None = None
    best_selection_5: list[dict[str, Any]] | None = None
    best_diag: dict[str, float] = {}

    for idx in range(1, max(1, random_samples) + 1):
        selected1, selected1_ids = _sample_rows(rows1_all, n1, rng)
        selected5, selected5_ids = _sample_rows(rows5_all, n5, rng)

        selected1, selected5, diag = _hill_climb_once(
            rows1_all=rows1_all,
            rows5_all=rows5_all,
            selected1=selected1,
            selected5=selected5,
            selected1_ids=selected1_ids,
            selected5_ids=selected5_ids,
            age_scale=age_scale,
            rosc_scale=rosc_scale,
            steps=hill_steps,
            rng=rng,
        )

        score = float(diag.get("score_total", float("inf")))
        if score < best_score:
            best_score = score
            best_selection_1 = selected1
            best_selection_5 = selected5
            best_diag = diag

        if idx % 25 == 0 or idx == random_samples:
            LOGGER.info("Balance search progress: %s/%s (best score %.5f)", idx, random_samples, best_score)

    if best_selection_1 is None or best_selection_5 is None:
        raise RuntimeError("Failed to select a balanced cohort.")

    ids1 = sorted(str(row["patient_id"]) for row in best_selection_1)
    ids5 = sorted(str(row["patient_id"]) for row in best_selection_5)
    return ids1, ids5, best_diag


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ages = [row.get("age") for row in rows if row.get("age") is not None]
    rosc = [row.get("rosc_min") for row in rows if row.get("rosc_min") is not None]
    return {
        "n": len(rows),
        "age_mean": float(sum(ages) / len(ages)) if ages else None,
        "age_median": float(sorted(ages)[len(ages) // 2]) if ages else None,
        "sex_counts": dict(Counter(str(row.get("sex", "unknown")) for row in rows)),
        "hospital_counts": dict(Counter(str(row.get("hospital", "unknown")) for row in rows)),
        "ohca_counts": dict(Counter(str(row.get("ohca", "missing")) for row in rows)),
        "shockable_counts": dict(Counter(str(row.get("shockable", "missing")) for row in rows)),
        "ttm_counts": dict(Counter(str(row.get("ttm", "none")) for row in rows)),
        "rosc_median_min": float(sorted(rosc)[len(rosc) // 2]) if rosc else None,
        "rosc_known_n": len(rosc),
    }


def _write_config(
    *,
    merged_cfg: dict[str, Any],
    patient_ids: list[str],
    output_subdir: str,
    output_config_path: Path,
    max_eeg_hours_per_patient: float,
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
            "patient_ids": patient_ids,
            "include_file_globs": subset_cfg.get("include_file_globs", ["*_EEG.hea", "*_EEG.mat", "*.txt"]),
            "include_top_level_files": subset_cfg.get(
                "include_top_level_files", ["LICENSE.txt", "RECORDS", "SHA256SUMS.txt"]
            ),
            "max_eeg_hours_per_patient": float(max_eeg_hours_per_patient),
        },
        "download": {
            "output_subdir": output_subdir,
            "max_parallel_downloads": int(download_cfg.get("max_parallel_downloads", 6)),
            "dry_run": False,
            "verify_checksum": False,
            "verify_existing_size": True,
        },
    }

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(yaml.safe_dump(output_doc, sort_keys=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-overlapping balanced I-CARE CPC1/CPC5 cohort and emit an "
            "explicit-patient downloader config."
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
        help="Base dataset config path.",
    )
    parser.add_argument(
        "--exclude-output-subdir",
        type=str,
        default="i-care_2_1_random100_longitudinal",
        help="Existing output_subdir used as non-overlap exclusion source.",
    )
    parser.add_argument(
        "--target-cpc1-count",
        type=int,
        default=70,
        help="Target number of CPC1 patients.",
    )
    parser.add_argument(
        "--target-cpc5-count",
        type=int,
        default=70,
        help="Target number of CPC5 patients.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="i-care_2_1_next_140_longitudinal_0_12h",
        help="Downloader output_subdir for selected cohort.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("physionet_ingest/config/config_i-care_2_1_next_140_longitudinal_0_12h.yml"),
        help="Generated explicit-patient config path.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional JSON plan report path.",
    )
    parser.add_argument(
        "--max-eeg-hours",
        type=float,
        default=12.0,
        help="Max cumulative EEG hours per patient in downloader config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260518,
        help="Random seed for balanced selection search.",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=150,
        help="Number of random restarts for balance optimization.",
    )
    parser.add_argument(
        "--hill-steps",
        type=int,
        default=700,
        help="Local swap-improvement steps per random restart.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.target_cpc1_count <= 0 or args.target_cpc5_count <= 0:
        parser.error("Both target counts must be > 0.")
    if args.max_eeg_hours <= 0:
        parser.error("--max-eeg-hours must be > 0.")
    if args.random_samples <= 0:
        parser.error("--random-samples must be > 0.")
    if args.hill_steps < 0:
        parser.error("--hill-steps must be >= 0.")

    config_general = dl.load_yaml_config(args.config_general)
    config_dataset = dl.load_yaml_config(args.config_dataset)
    cfg = dl.deep_merge_dict(config_general, config_dataset)

    paths_cfg = cfg.get("paths", {})
    network_cfg = cfg.get("network", {})
    dataset_cfg = cfg.get("dataset", {})

    timeout_seconds = float(network_cfg.get("timeout_seconds", 60))
    retries = int(network_cfg.get("retries", 3))
    retry_backoff_seconds = float(network_cfg.get("retry_backoff_seconds", 2.0))
    user_agent = str(network_cfg.get("user_agent", "NeuralManifoldDynamics-PhysioNetIngest/1.0"))

    base_url = str(dataset_cfg.get("base_url", "https://physionet.org"))
    remote_root = str(dataset_cfg.get("remote_root", "files/i-care/2.1"))
    records_file = str(dataset_cfg.get("records_file", "RECORDS"))
    checksums_file = str(dataset_cfg.get("checksums_file", "SHA256SUMS.txt"))
    dataset_root_url = dl.build_dataset_root_url(base_url, remote_root)

    download_root = Path(str(paths_cfg.get("download_root", "E:/Science_Datasets/physionet/received")))
    metadata_root = Path(str(paths_cfg.get("metadata_root", "E:/Science_Datasets/physionet/metadata")))
    exclude_output_dir = download_root / str(args.exclude_output_subdir)
    training_root = exclude_output_dir / "training"

    LOGGER.info("Reading local exclusion cohort metadata from: %s", training_root)
    local_metadata = _read_local_metadata(training_root)
    excluded_ids = set(local_metadata.keys())
    LOGGER.info("Local exclusion set size: %s", len(excluded_ids))

    records_cache_path = exclude_output_dir.joinpath(*PurePosixPath(dl.normalize_relative_path(records_file)).parts)
    if records_cache_path.exists() and records_cache_path.stat().st_size > 0:
        records_text = records_cache_path.read_text(encoding="utf-8")
        LOGGER.info("Using cached records file: %s", records_cache_path)
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
    checksums_cache_path = exclude_output_dir.joinpath(*PurePosixPath(dl.normalize_relative_path(checksums_file)).parts)
    if checksums_cache_path.exists() and checksums_cache_path.stat().st_size > 0:
        checksums_text = checksums_cache_path.read_text(encoding="utf-8")
        checksums_lookup = dl.build_checksum_lookup(dl.parse_sha256sums_text(checksums_text), remote_root=remote_root)
        LOGGER.info("Using cached checksums file: %s", checksums_cache_path)
    else:
        LOGGER.warning(
            "No local checksum cache found at %s; metadata path resolution will use default sidecar pattern.",
            checksums_cache_path,
        )

    candidate_ids = [patient_id for patient_id in patient_pool if patient_id not in excluded_ids]
    LOGGER.info("Candidate non-overlap patients in pool: %s", len(candidate_ids))

    remote_metadata = _collect_remote_metadata(
        candidate_ids=candidate_ids,
        checksums_lookup=checksums_lookup,
        output_dir=exclude_output_dir,
        dataset_root_url=dataset_root_url,
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        user_agent=user_agent,
    )

    cpc1_rows = [
        {"patient_id": patient_id, **meta}
        for patient_id, meta in sorted(remote_metadata.items())
        if int(meta.get("cpc", 0)) == 1
    ]
    cpc5_rows = [
        {"patient_id": patient_id, **meta}
        for patient_id, meta in sorted(remote_metadata.items())
        if int(meta.get("cpc", 0)) == 5
    ]
    LOGGER.info("Remote candidate counts: CPC1=%s, CPC5=%s", len(cpc1_rows), len(cpc5_rows))

    selected_cpc1, selected_cpc5, balance_diag = _select_balanced_groups(
        rows1_all=cpc1_rows,
        rows5_all=cpc5_rows,
        n1=int(args.target_cpc1_count),
        n5=int(args.target_cpc5_count),
        seed=int(args.seed),
        random_samples=int(args.random_samples),
        hill_steps=int(args.hill_steps),
    )

    selected_all = sorted(selected_cpc1 + selected_cpc5)
    _write_config(
        merged_cfg=cfg,
        patient_ids=selected_all,
        output_subdir=str(args.output_subdir),
        output_config_path=args.output_config,
        max_eeg_hours_per_patient=float(args.max_eeg_hours),
    )

    selected_rows_1 = [next(row for row in cpc1_rows if row["patient_id"] == patient_id) for patient_id in selected_cpc1]
    selected_rows_5 = [next(row for row in cpc5_rows if row["patient_id"] == patient_id) for patient_id in selected_cpc5]
    summary_1 = _summarize_group(selected_rows_1)
    summary_5 = _summarize_group(selected_rows_5)

    report_path = args.output_report
    if report_path is None:
        report_path = metadata_root / str(args.output_subdir) / "balanced_cpc15_plan.json"
    report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "search": {
            "random_samples": int(args.random_samples),
            "hill_steps": int(args.hill_steps),
        },
        "targets": {
            "cpc1_count": int(args.target_cpc1_count),
            "cpc5_count": int(args.target_cpc5_count),
            "output_subdir": str(args.output_subdir),
            "max_eeg_hours": float(args.max_eeg_hours),
        },
        "candidate_counts": {
            "pool_total": len(patient_pool),
            "excluded_local_count": len(excluded_ids),
            "non_overlap_candidates": len(candidate_ids),
            "remote_metadata_with_cpc": len(remote_metadata),
            "cpc1_candidates": len(cpc1_rows),
            "cpc5_candidates": len(cpc5_rows),
        },
        "selected_counts": {
            "cpc1": len(selected_cpc1),
            "cpc5": len(selected_cpc5),
            "total": len(selected_all),
        },
        "balance_diagnostics": balance_diag,
        "selected_group_summaries": {
            "cpc1": summary_1,
            "cpc5": summary_5,
        },
        "selected_patient_ids": {
            "cpc1": selected_cpc1,
            "cpc5": selected_cpc5,
            "all_sorted": selected_all,
        },
        "output_config_path": str(args.output_config),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Balanced config written: {args.output_config}")
    print(f"Balanced report written: {report_path}")
    print(f"Selected patients: {len(selected_all)} (CPC1={len(selected_cpc1)}, CPC5={len(selected_cpc5)})")
    print(f"Best balance score: {balance_diag.get('score_total')}")
    print(f"CPC1 preview: {', '.join(selected_cpc1[:10])}")
    print(f"CPC5 preview: {', '.join(selected_cpc5[:10])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
