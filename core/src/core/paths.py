"""Path resolution helpers shared across packages.

Chooses ``received_dir`` (raw data root) and ``processed_dir`` from CLI overrides
or config, with optional ``received_dir_fallbacks`` and scoring by how many
configured dataset folders exist under each candidate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Tuple


def resolve_paths(
    config: Mapping[str, Any],
    cli_out_dir: Path | None,
    cli_data_dir: Path | None = None,
) -> Tuple[Path, Path]:
    """Resolve dataset ``received`` (raw) and ``processed`` output directories.

    Args:
        config: Ingest configuration containing a ``paths`` section.
        cli_out_dir: If set, overrides ``paths.processed_dir``.
        cli_data_dir: If set, overrides ``paths.received_dir`` (highest priority).

    Returns:
        ``(received_dir, processed_dir)`` as resolved :class:`pathlib.Path` objects.
    """
    paths_cfg = config.get("paths", {}) if isinstance(config, Mapping) else {}

    def _dataset_ids_from_config(cfg: Mapping[str, Any]) -> list[str]:
        """Internal helper: dataset ids from config."""
        raw = cfg.get("datasets", []) if isinstance(cfg, Mapping) else []
        out: list[str] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, Mapping) and "id" in item:
                try:
                    out.append(str(item.get("id")))
                except Exception:
                    continue
        return [x for x in out if x]

    def _score_received_root(root: Path, ds_ids: list[str]) -> int:
        # Score by how many dataset directories exist under this root.
        """Internal helper: score received root."""
        try:
            if not root.exists() or not root.is_dir():
                return -1
        except Exception:
            return -1
        if not ds_ids:
            return 0
        score = 0
        for ds in ds_ids:
            try:
                if (root / ds).exists():
                    score += 1
            except Exception:
                continue
        return score

    # Build received_dir candidates:
    # - CLI override wins
    # - otherwise use config paths.received_dir
    # - plus optional fallbacks: paths.received_dir_fallbacks
    received_candidates: list[Path] = []
    if cli_data_dir:
        received_candidates.append(Path(cli_data_dir))
    else:
        received_candidates.append(Path(paths_cfg.get("received_dir", "E:/Science_Datasets/openneuro/received")))
        fallbacks = paths_cfg.get("received_dir_fallbacks", []) if isinstance(paths_cfg, Mapping) else []
        if isinstance(fallbacks, (list, tuple)):
            for fb in fallbacks:
                if fb:
                    received_candidates.append(Path(str(fb)))

    # Choose best candidate (prefer existing root that contains the most dataset dirs).
    ds_ids = _dataset_ids_from_config(config)
    best = None
    best_score = -2
    for cand in received_candidates:
        score = _score_received_root(cand, ds_ids)
        if score > best_score:
            best = cand
            best_score = score
    received_dir = best if best is not None else received_candidates[0]

    processed_dir = Path(cli_out_dir) if cli_out_dir else Path(
        paths_cfg.get("processed_dir", "E:/Science_Datasets/openneuro/processed")
    )
    return received_dir, processed_dir

