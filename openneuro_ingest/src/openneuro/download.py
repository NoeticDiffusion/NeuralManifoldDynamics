"""Download wrappers for OpenNeuro datasets and basic integrity checks.

Responsibilities
----------------
- Wrap `openneuro-py` to download BIDS datasets with include/exclude patterns.
- Provide simple retry logic and return local dataset root paths.

Inputs
------
- dataset_ids: iterable of OpenNeuro dataset ids.
- config: dict with download-related settings (patterns, retries, cache dir).
- out_dir: base output directory.

Outputs
-------
- Mapping dataset_id -> local path (as `Path`).

Dependencies
------------
- Optional: `openneuro` at runtime when downloads are invoked.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

logger = logging.getLogger(__name__)


class MissingIncludePatternError(RuntimeError):
    """Raised when openneuro reports include glob(s) that are absent in a dataset."""

    def __init__(self, patterns: Sequence[str], source: str) -> None:
        self.patterns: List[str] = [p for p in patterns if p]
        self.source = source
        msg_patterns = ", ".join(self.patterns) if self.patterns else "<unknown>"
        super().__init__(f"Include pattern(s) not found ({source}): {msg_patterns}")


_MISSING_INCLUDE_RE = re.compile(
    r"Could not find path in the dataset:\s*(?:-\s*(?P<pattern>.+))",
    re.IGNORECASE,
)


def _extract_missing_patterns(message: str) -> List[str]:
    return [m.strip() for m in _MISSING_INCLUDE_RE.findall(message or "") if m]


def _resolve_patterns_for_dataset(
    config: Mapping[str, object] | None,
    dataset_id: str,
) -> tuple[List[str], List[str]]:
    include_patterns: List[str] = []
    exclude_patterns: List[str] = []

    if isinstance(config, Mapping):
        dl_cfg = config.get("download", {}) or {}
        if isinstance(dl_cfg, Mapping):
            include_patterns = list(dl_cfg.get("include_patterns", []) or [])
            exclude_patterns = list(dl_cfg.get("exclude_patterns", []) or [])

            ds_overrides = (dl_cfg.get("datasets", {}) or {}).get(dataset_id, {})
            if isinstance(ds_overrides, Mapping):
                if "include_patterns" in ds_overrides:
                    include_patterns = list(ds_overrides.get("include_patterns") or [])
                if "exclude_patterns" in ds_overrides:
                    exclude_patterns = list(ds_overrides.get("exclude_patterns") or [])

                extra_includes = ds_overrides.get("include_patterns_extra")
                if extra_includes:
                    include_patterns.extend(
                        [p for p in extra_includes if p not in include_patterns]
                    )
                extra_excludes = ds_overrides.get("exclude_patterns_extra")
                if extra_excludes:
                    exclude_patterns.extend(
                        [p for p in extra_excludes if p not in exclude_patterns]
                    )

    return include_patterns, exclude_patterns


def _perform_openneuro_download(
    dataset_id: str,
    dataset_path: Path,
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
) -> None:
    """Attempt download via Python API, falling back to CLI."""
    api_error: Exception | None = None

    try:
        import openneuro as on  # type: ignore

        logger.info("Using openneuro-py Python API")
        kwargs = {
            "dataset": dataset_id,
            "target_dir": str(dataset_path),
        }
        if include_patterns:
            kwargs["include"] = list(include_patterns)
        if exclude_patterns:
            kwargs["exclude"] = list(exclude_patterns)

        on.download(**kwargs)
        return
    except Exception as exc:  # noqa: BLE001 - capture for fallback analysis
        api_error = exc
        missing = _extract_missing_patterns(str(exc))
        if missing:
            raise MissingIncludePatternError(missing, source="python_api") from exc
        logger.info("Falling back to openneuro-py CLI due to: %s", exc)

    cli = _resolve_openneuro_cli()
    cmd: List[str] = [
        cli,
        "download",
        f"--dataset={dataset_id}",
        f"--target-dir={str(dataset_path)}",
    ]
    for inc in include_patterns:
        cmd.extend(["--include", inc])
    for exc in exclude_patterns:
        cmd.extend(["--exclude", exc])

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except subprocess.CalledProcessError as proc_err:
        stdout = proc_err.stdout or ""
        stderr = proc_err.stderr or ""
        missing = _extract_missing_patterns("\n".join([stderr, stdout]))
        if missing:
            raise MissingIncludePatternError(missing, source="cli") from proc_err
        msg = stderr.strip() or stdout.strip() or str(proc_err)
        if api_error is not None:
            msg = f"Python API error: {api_error}; CLI error: {msg}"
        raise RuntimeError(f"openneuro download failed: {msg}") from proc_err


def download_datasets(dataset_ids: Iterable[str], config: Mapping[str, object], out_dir: Path) -> Dict[str, Path]:
    """Download datasets and return local root paths.
    
    Parameters
    ----------
    dataset_ids
        List of OpenNeuro dataset IDs to download.
    config
        Configuration dict with download settings.
    out_dir
        Base output directory for downloads.
    
    Returns
    -------
    Mapping from dataset_id to local path.
    
    Raises
    ------
    RuntimeError
        If download fails after retries.
    """
    result: Dict[str, Path] = {}
    retries = config.get("download", {}).get("retries", 3) if isinstance(config, dict) else 3
    
    for ds_id in dataset_ids:
        logger.info(f"Downloading dataset {ds_id}")
        dataset_path = Path(out_dir) / ds_id
        dataset_path.mkdir(parents=True, exist_ok=True)

        include_patterns, exclude_patterns = _resolve_patterns_for_dataset(config, ds_id)
        sanitized_includes = list(include_patterns)

        success = False
        attempts = 0
        last_error: Exception | None = None
        while True:
            try:
                _perform_openneuro_download(ds_id, dataset_path, sanitized_includes, exclude_patterns)
                success = True
                logger.info(f"Successfully downloaded {ds_id}")
                break
            except MissingIncludePatternError as missing_err:
                removed = False
                for pattern in missing_err.patterns:
                    if pattern in sanitized_includes:
                        sanitized_includes = [p for p in sanitized_includes if p != pattern]
                        logger.warning(
                            "Include pattern '%s' not present in %s (reported by %s); removing and retrying",
                            pattern,
                            ds_id,
                            missing_err.source,
                        )
                        removed = True
                if removed:
                    if not sanitized_includes and include_patterns:
                        logger.warning(
                            "All include patterns removed for %s; retrying download without include filters",
                            ds_id,
                        )
                    continue
                # Nothing removed—treat as a failed attempt.
                attempts += 1
                last_error = missing_err
            except Exception as exc:  # noqa: BLE001
                attempts += 1
                last_error = exc

            if attempts >= retries:
                msg = str(last_error) if last_error else "unknown error"
                raise RuntimeError(f"Failed to download {ds_id} after {retries} attempts ({msg})") from last_error

        result[ds_id] = dataset_path
    
    return result



def _resolve_openneuro_cli() -> str:
    """Return the path/name of the openneuro-py CLI executable.

    On Windows within a venv, this is typically `openneuro-py.exe` under Scripts.
    Otherwise, rely on PATH to resolve `openneuro-py`.
    """
    # Try venv Scripts first
    venv_scripts = Path(sys.executable).parent
    candidate = venv_scripts / ("openneuro-py.exe" if sys.platform.startswith("win") else "openneuro-py")
    if candidate.exists():
        return str(candidate)
    return "openneuro-py"

