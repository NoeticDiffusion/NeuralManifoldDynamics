"""Dataset registry and path helpers.

Lists dataset IDs from ingest configuration and resolves per-dataset output
directories. Expects a config dict as returned by :func:`core.config_loader.load_config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple


def _normalize_dataset_entry(entry) -> Tuple[str | None, bool]:
    """Parse a ``datasets`` list entry into ``(dataset_id, pca_results_flag)``."""
    if isinstance(entry, str):
        return entry, False
    if isinstance(entry, Mapping):
        dataset_id = entry.get("id") or entry.get("dataset") or entry.get("name")
        if dataset_id:
            flagged = entry.get("pca_results")
            if flagged is None:
                # Backward compatibility for earlier "analysis" key
                flagged = entry.get("analysis", False)
            return str(dataset_id), bool(flagged)
    return None, False


def list_datasets(config: Mapping[str, object], include_pca_results: bool = False) -> List[str]:
    """Return dataset ids to process (for example ``["ds003490", ...]``).

    Args:
        config: Configuration mapping with a ``datasets`` key.
        include_pca_results: If True, include entries flagged with
            ``pca_results: true`` in YAML.

    Returns:
        List of dataset ID strings.
    """
    datasets = config.get("datasets") if isinstance(config, Mapping) else None
    if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)):
        return []

    resolved: List[str] = []
    for item in datasets:
        dataset_id, flagged_pca = _normalize_dataset_entry(item)
        if not dataset_id:
            continue
        if flagged_pca and not include_pca_results:
            continue
        resolved.append(dataset_id)
    return resolved


def dataset_output_dir(base_out: Path, dataset_id: str) -> Path:
    """Return ``base_out / dataset_id`` as the output directory for one dataset.

    Args:
        base_out: Base output directory from config.
        dataset_id: Dataset identifier (for example ``ds003490``).

    Returns:
        Resolved filesystem path for that dataset's outputs.
    """
    return Path(base_out) / dataset_id


