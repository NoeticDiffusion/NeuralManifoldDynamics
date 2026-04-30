from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dandi_io.contracts import (
    DandiIngestionConfig,
    DatasetSpec,
    ExecutionSpec,
    OutputSpec,
    SelectionSpec,
    StorageSpec,
)


DEFAULT_DANDI_CONFIG: dict[str, Any] = {
    "dataset": {
        "config_id": "base",
        "adapter": "generic",
        "dandiset_id": "",
        "version": "draft",
    },
    "storage": {
        "output_root": "data/dandi",
        "cache_root": "data/dandi/cache",
    },
    "selection": {
        "path_filters": [],
        "subject_filters": [],
        "session_filters": [],
        "asset_limit": None,
    },
    "execution": {
        "metadata_only": True,
        "streaming_allowed": True,
    },
    "outputs": {
        "manifest_json": None,
        "manifest_csv": None,
        "triage_markdown": None,
        "probe_json": None,
    },
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at config root in {config_path}.")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_dandi_config(path: str | Path) -> tuple[DandiIngestionConfig, Path]:
    config_path = Path(path)
    loaded = load_yaml_config(config_path)
    merged = deep_merge(DEFAULT_DANDI_CONFIG, loaded)
    config = _coerce_config(merged)
    return config, config_path


def _coerce_config(data: dict[str, Any]) -> DandiIngestionConfig:
    dataset_block = _ensure_mapping(data.get("dataset"), "dataset")
    storage_block = _ensure_mapping(data.get("storage"), "storage")
    selection_block = _ensure_mapping(data.get("selection"), "selection")
    execution_block = _ensure_mapping(data.get("execution"), "execution")
    outputs_block = _ensure_mapping(data.get("outputs"), "outputs")

    dandiset_id = str(dataset_block.get("dandiset_id", "")).strip()
    if not dandiset_id:
        raise ValueError("Config must define `dataset.dandiset_id`.")

    config_id = str(dataset_block.get("config_id") or f"dandiset_{dandiset_id}").strip()
    adapter = str(dataset_block.get("adapter") or "generic").strip()
    version = str(dataset_block.get("version") or "draft").strip()

    output_root = Path(str(storage_block.get("output_root") or "data/dandi"))
    cache_root = Path(str(storage_block.get("cache_root") or output_root / "cache"))
    raw_root = Path(str(storage_block.get("raw_root") or output_root / "raw" / dandiset_id))
    manifest_root = Path(
        str(storage_block.get("manifest_root") or output_root / "manifests" / dandiset_id)
    )
    triage_root = Path(str(storage_block.get("triage_root") or output_root / "triage" / dandiset_id))

    manifest_json = Path(
        str(outputs_block.get("manifest_json") or manifest_root / f"{config_id}_manifest.json")
    )
    manifest_csv = Path(
        str(outputs_block.get("manifest_csv") or manifest_root / f"{config_id}_manifest.csv")
    )
    triage_markdown = Path(
        str(outputs_block.get("triage_markdown") or triage_root / f"{config_id}_triage.md")
    )
    probe_json = Path(str(outputs_block.get("probe_json") or triage_root / f"{config_id}_probe.json"))

    dataset = DatasetSpec(
        config_id=config_id,
        adapter=adapter,
        dandiset_id=dandiset_id,
        version=version,
        metadata={k: v for k, v in dataset_block.items() if k not in {"config_id", "adapter", "dandiset_id", "version"}},
    )
    storage = StorageSpec(
        output_root=output_root,
        cache_root=cache_root,
        raw_root=raw_root,
        manifest_root=manifest_root,
        triage_root=triage_root,
        metadata={
            k: v
            for k, v in storage_block.items()
            if k not in {"output_root", "cache_root", "raw_root", "manifest_root", "triage_root"}
        },
    )
    selection = SelectionSpec(
        path_filters=_coerce_string_tuple(selection_block.get("path_filters")),
        subject_filters=_coerce_string_tuple(selection_block.get("subject_filters")),
        session_filters=_coerce_string_tuple(selection_block.get("session_filters")),
        asset_limit=_coerce_optional_int(selection_block.get("asset_limit")),
        metadata={
            k: v
            for k, v in selection_block.items()
            if k not in {"path_filters", "subject_filters", "session_filters", "asset_limit"}
        },
    )
    execution = ExecutionSpec(
        metadata_only=bool(execution_block.get("metadata_only", True)),
        streaming_allowed=bool(execution_block.get("streaming_allowed", True)),
        metadata={
            k: v
            for k, v in execution_block.items()
            if k not in {"metadata_only", "streaming_allowed"}
        },
    )
    outputs = OutputSpec(
        manifest_json=manifest_json,
        manifest_csv=manifest_csv,
        triage_markdown=triage_markdown,
        probe_json=probe_json,
        metadata={
            k: v
            for k, v in outputs_block.items()
            if k not in {"manifest_json", "manifest_csv", "triage_markdown", "probe_json"}
        },
    )
    return DandiIngestionConfig(
        dataset=dataset,
        storage=storage,
        selection=selection,
        execution=execution,
        outputs=outputs,
        metadata={
            k: v
            for k, v in data.items()
            if k not in {"dataset", "storage", "selection", "execution", "outputs"}
        },
    )


def ensure_storage_roots(config: DandiIngestionConfig) -> None:
    config.storage.cache_root.mkdir(parents=True, exist_ok=True)
    config.storage.raw_root.mkdir(parents=True, exist_ok=True)
    config.storage.manifest_root.mkdir(parents=True, exist_ok=True)
    config.storage.triage_root.mkdir(parents=True, exist_ok=True)
    config.outputs.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    config.outputs.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    config.outputs.triage_markdown.parent.mkdir(parents=True, exist_ok=True)
    config.outputs.probe_json.parent.mkdir(parents=True, exist_ok=True)


def _ensure_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping at config section `{label}`.")
    return value


def _coerce_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError("Expected list/tuple of strings in selection filters.")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
