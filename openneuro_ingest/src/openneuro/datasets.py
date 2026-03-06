"""Dataset registry and path helpers.

Responsibilities
----------------
- Provide a registry of target OpenNeuro dataset ids.
- Resolve output directories and per-dataset paths based on config.

Inputs
------
- config: dict returned by `config_loader.load_config`.

Outputs
-------
- Simple helpers: list of dataset ids; computed paths.

Dependencies
------------
- None at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple


def _normalize_dataset_entry(entry) -> Tuple[str | None, bool]:
    """Return (dataset_id, analysis_flag) tuple for a list entry."""
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
    """Return the dataset ids to process, e.g., ["ds003490", ...].
    
    Parameters
    ----------
    config
        Configuration dict with "datasets" key.
    include_pca_results
        Include datasets flagged with ``pca_results: true`` if set.
    
    Returns
    -------
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
    """Compute output directory for a dataset.
    
    Parameters
    ----------
    base_out
        Base output directory from config.
    dataset_id
        Dataset identifier (e.g., "ds003490").
    
    Returns
    -------
    Resolved path for dataset output.
    """
    return Path(base_out) / dataset_id


