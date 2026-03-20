"""Config-driven file exclusion helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Mapping

import pandas as pd


def resolve_exclude_file_patterns(config: Mapping[str, Any] | None) -> list[str]:
    """Return configured file-exclusion regex patterns.

    Supports both ``exclude-files`` and ``exclude_files`` for YAML convenience.
    """

    if not isinstance(config, Mapping):
        return []
    raw = config.get("exclude-files", config.get("exclude_files"))
    if raw is None:
        return []
    if isinstance(raw, str):
        pattern = raw.strip()
        return [pattern] if pattern else []
    if isinstance(raw, Iterable):
        out: list[str] = []
        for item in raw:
            pattern = str(item).strip()
            if pattern:
                out.append(pattern)
        return out
    return []


def apply_exclude_file_filters(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, Any] | None,
    candidate_columns: Sequence[str] = ("path", "file"),
) -> tuple[pd.DataFrame, int, list[str]]:
    """Filter rows whose file/path columns match configured exclusion regexes."""

    patterns = resolve_exclude_file_patterns(config)
    if frame.empty or not patterns:
        return frame, 0, patterns

    present_columns = [str(col) for col in candidate_columns if str(col) in frame.columns]
    if not present_columns:
        return frame, 0, patterns

    mask = pd.Series(False, index=frame.index, dtype=bool)
    for col in present_columns:
        series = frame[col].astype(str)
        for pattern in patterns:
            mask |= series.str.contains(pattern, regex=True, na=False)

    excluded = int(mask.sum())
    if excluded <= 0:
        return frame, 0, patterns
    return frame.loc[~mask].copy(), excluded, patterns
