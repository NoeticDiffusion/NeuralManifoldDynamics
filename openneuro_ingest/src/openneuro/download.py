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
- Optional: `openneuro` at runtime when downloads are invoked via the Python
  API (only used when ``download.use_uvx`` is false).
- `uvx` (from `uv`) on PATH by default; the download is run through
  ``uvx openneuro-py@latest`` so the latest released openneuro-py is used
  in an isolated environment. Override with ``download.use_uvx: false`` in
  config or the ``OPENNEURO_PREFER_UVX=0`` env var.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
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


def _user_script_dirs() -> List[Path]:
    """Per-user script directories where pip `--user` installs CLI entry points."""
    dirs: List[Path] = []
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if appdata:
            pyxy = f"Python{sys.version_info.major}{sys.version_info.minor}"
            dirs.append(Path(appdata) / "Python" / pyxy / "Scripts")
    else:
        home = os.environ.get("HOME")
        if home:
            dirs.append(Path(home) / ".local" / "bin")
    return dirs


def _perform_openneuro_download(
    dataset_id: str,
    dataset_path: Path,
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
    *,
    use_uvx: bool = True,
) -> None:
    """Attempt download via the openneuro-py CLI, optionally via the Python API.

    By default (``use_uvx=True``) the download is run through
    ``uvx openneuro-py@latest`` so the pipeline uses the latest released
    openneuro-py in an isolated environment. This avoids breakage from an
    installed openneuro-py whose GraphQL metadata query is incompatible
    with the current OpenNeuro server schema — e.g. 2026.3.0 querying the
    removed ``key`` field on ``DatasetFile`` raises
    ``Cannot query field "key" on type "DatasetFile"`` and never reaches
    the file-transfer stage. ``uvx`` fetches a working latest version
    without modifying the current project.

    When ``use_uvx=False`` the function first tries the locally installed
    ``openneuro`` Python API, and on failure falls back to a locally
    installed ``openneuro-py`` CLI executable (or uvx if none is found).
    """
    api_error: Exception | None = None

    if not use_uvx:
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

    cli_prefix = _resolve_openneuro_cli(prefer_uvx=use_uvx)
    cmd: List[str] = cli_prefix + [
        "download",
        f"--dataset={dataset_id}",
        f"--target-dir={str(dataset_path)}",
    ]
    for inc in include_patterns:
        # On Windows, passing wildcard includes as separate argv entries lets
        # uvx/native argument handling expand them against the caller's local
        # working tree before openneuro-py sees them.  The equals form keeps
        # each include as one option-value token.
        cmd.append(f"--include={inc}")
    for exc in exclude_patterns:
        cmd.append(f"--exclude={exc}")

    logger.info("Using openneuro-py CLI via: %s", " ".join(cli_prefix))
    try:
        # The ingest command is normally launched with openneuro_ingest/src on
        # PYTHONPATH so that this package can import `core`.  If that variable
        # leaks into uvx, the child executable imports this repository's local
        # `openneuro` package instead of the isolated openneuro-py package and
        # fails with `No module named openneuro._cli`.
        child_env = os.environ.copy()
        if use_uvx:
            child_env.pop("PYTHONPATH", None)
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            env=child_env,
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


_METADATA_ONLY_MARKERS = (
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "events.tsv",
    "events.json",
    "channels.tsv",
    "channels.json",
    "electrodes.tsv",
    "coordsystem.json",
    "README",
    "_eeg.json",
    "_ieeg.json",
    "_meg.json",
)
_ALLOWED_METADATA_ONLY_PATTERNS = {
    "**/dataset_description.json",
    "**/participants.tsv",
    "**/participants.json",
    "**/*_events.tsv",
    "**/*_events.json",
    "**/*_channels.tsv",
    "**/*_channels.json",
    "**/*_electrodes.tsv",
    "**/*_coordsystem.json",
    "**/*_eeg.json",
    "**/*_ieeg.json",
    "**/*_meg.json",
}
_RAW_SIGNAL_SUFFIXES = (
    ".edf",
    ".edf.gz",
    ".eeg",
    ".fif",
    ".fif.gz",
    ".mat",
    ".nii",
    ".nii.gz",
    ".set",
    ".fdt",
    ".bdf",
    ".vhdr",
    ".vmrk",
    ".dat",
    ".npy",
    ".npz",
    ".nwb",
    ".h5",
    ".hdf5",
    ".bin",
    ".raw",
    ".mff",
    ".mef",
    ".nev",
    ".ncs",
    ".ns1",
    ".ns2",
    ".ns3",
    ".ns4",
    ".ns5",
    ".ns6",
    ".sif",
    ".trc",
    ".cnt",
    ".con",
    ".wav",
    ".mp4",
)
_EXPLICIT_DOWNLOAD_BOOKKEEPING_NAMES = {
    "README",
    "README.md",
    "CHANGES",
    "LICENSE",
    "dataset_description.json",
    "participants.json",
    "participants.tsv",
}


def _validate_metadata_only_patterns(patterns: Sequence[str]) -> None:
    """Reject include globs that could fetch neural signal payloads."""
    if not patterns:
        raise ValueError("metadata_only_patterns_required")
    for pattern in patterns:
        normalized = str(pattern).replace("\\", "/").lower()
        if normalized not in _ALLOWED_METADATA_ONLY_PATTERNS:
            raise ValueError(f"metadata_only_pattern_not_allowlisted:{pattern}")
        if any(normalized.endswith(suffix) for suffix in _RAW_SIGNAL_SUFFIXES):
            raise ValueError(f"raw_signal_pattern_not_allowed:{pattern}")
        if not any(marker.lower() in normalized for marker in _METADATA_ONLY_MARKERS):
            raise ValueError(f"non_metadata_pattern_not_allowed:{pattern}")


def _payload_files(root: Path) -> List[Path]:
    """Return known neural/media payloads beneath a download root."""
    payloads: List[Path] = []
    if not root.exists():
        return payloads
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in _RAW_SIGNAL_SUFFIXES):
            payloads.append(path)
    return payloads


def _resolve_metadata_only_patterns(
    config: Mapping[str, object] | None,
    dataset_id: str,
) -> tuple[List[str], List[str]]:
    """Resolve strict metadata-only patterns without generic download fallbacks."""
    required: List[str] = []
    optional: List[str] = []
    if isinstance(config, Mapping):
        dl_cfg = config.get("download", {}) or {}
        if isinstance(dl_cfg, Mapping):
            required = list(dl_cfg.get("metadata_only_patterns", []) or [])
            optional = list(dl_cfg.get("metadata_only_optional_patterns", []) or [])
            overrides = (dl_cfg.get("datasets", {}) or {}).get(dataset_id, {})
            if isinstance(overrides, Mapping) and "metadata_only_patterns" in overrides:
                required = list(overrides.get("metadata_only_patterns") or [])
            if isinstance(overrides, Mapping) and "metadata_only_optional_patterns" in overrides:
                optional = list(overrides.get("metadata_only_optional_patterns") or [])
    return required, optional


def download_metadata_only(
    dataset_ids: Iterable[str],
    config: Mapping[str, object],
    out_dir: Path,
) -> Dict[str, Path]:
    """Download only explicitly allowlisted BIDS metadata/event files.

    Unlike :func:`download_datasets`, this function never removes missing
    include patterns and never retries with an unfiltered dataset download.
    Missing patterns are a source-schema failure and remain fail-closed.
    """
    dl_cfg = config.get("download", {}) if isinstance(config, Mapping) else {}
    retries = int(dl_cfg.get("retries", 3)) if isinstance(dl_cfg, Mapping) else 3
    use_uvx = bool(dl_cfg.get("use_uvx", True)) if isinstance(dl_cfg, Mapping) else True
    env_override = os.environ.get("OPENNEURO_PREFER_UVX")
    if env_override is not None and env_override.strip():
        use_uvx = env_override.strip().lower() in ("1", "true", "yes", "on")

    result: Dict[str, Path] = {}
    for dataset_id in dataset_ids:
        required_patterns, optional_patterns = _resolve_metadata_only_patterns(config, dataset_id)
        if not required_patterns:
            raise ValueError(f"metadata_only_required_patterns_required:{dataset_id}")
        _validate_metadata_only_patterns(required_patterns + optional_patterns)
        include_patterns = list(required_patterns) + list(optional_patterns)
        dataset_path = Path(out_dir) / dataset_id
        dataset_path.mkdir(parents=True, exist_ok=True)
        existing_payloads = _payload_files(dataset_path)
        if existing_payloads:
            raise RuntimeError(
                f"metadata_only_target_contains_payload:{dataset_id}:"
                f"{existing_payloads[0]}"
            )

        success = False
        attempts = 0
        while attempts < retries and not success:
            try:
                _perform_openneuro_download(
                    dataset_id,
                    dataset_path,
                    include_patterns,
                    [],
                    use_uvx=use_uvx,
                )
                success = True
                break
            except MissingIncludePatternError as missing_err:
                missing_optional = [
                    pattern
                    for pattern in missing_err.patterns
                    if pattern in optional_patterns
                ]
                missing_required = [
                    pattern
                    for pattern in missing_err.patterns
                    if pattern not in optional_patterns
                ]
                if missing_required:
                    raise
                if missing_optional:
                    optional_patterns = [
                        pattern for pattern in optional_patterns if pattern not in missing_optional
                    ]
                    include_patterns = list(required_patterns) + list(optional_patterns)
                    logger.warning(
                        "Optional metadata pattern(s) not present in %s; continuing with "
                        "remaining explicit metadata allowlist: %s",
                        dataset_id,
                        ", ".join(missing_optional),
                    )
                    continue
                raise
            except Exception as exc:  # noqa: BLE001 - preserve retry boundary
                attempts += 1
                if attempts >= retries:
                    raise RuntimeError(
                        f"Failed metadata-only download for {dataset_id} "
                        f"after {retries} attempts ({exc})"
                    ) from exc
                logger.warning(
                    "Metadata-only download attempt %s/%s failed for %s: %s",
                    attempts,
                    retries,
                    dataset_id,
                    exc,
                )
        if not success:
            raise RuntimeError(f"metadata_only_download_exhausted:{dataset_id}")
        downloaded_payloads = _payload_files(dataset_path)
        if downloaded_payloads:
            raise RuntimeError(
                f"metadata_only_payload_detected:{dataset_id}:{downloaded_payloads[0]}"
            )
        if not any(
            path.is_file() and path.name.lower() == "dataset_description.json"
            for path in dataset_path.rglob("*")
        ):
            raise RuntimeError(f"metadata_only_required_file_missing:{dataset_id}:dataset_description.json")
        if not any(
            path.is_file() and path.name.lower().endswith("_events.tsv")
            for path in dataset_path.rglob("*")
        ):
            raise RuntimeError(f"metadata_only_required_file_missing:{dataset_id}:*_events.tsv")
        result[dataset_id] = dataset_path
    return result


def _validate_explicit_payload_paths(paths: Sequence[str]) -> List[str]:
    """Validate exact OpenNeuro paths for a signal-payload download."""
    normalized: List[str] = []
    for raw in paths:
        value = str(raw).replace("\\", "/").strip()
        path = PurePosixPath(value)
        if (
            not value
            or path.is_absolute()
            or ".." in path.parts
            or any(token in value for token in ("*", "?", "[", "]"))
            or path.suffix.lower() not in {".set", ".fdt"}
        ):
            raise ValueError(f"explicit_payload_path_not_allowed:{raw}")
        normalized.append(str(path))
    if not normalized:
        raise ValueError("explicit_payload_paths_required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("explicit_payload_paths_duplicate")
    return sorted(normalized)


def download_explicit_allowlist(
    dataset_id: str,
    payload_paths: Sequence[str],
    out_dir: Path,
    *,
    existing_paths: Sequence[str] = (),
    retries: int = 3,
    use_uvx: bool = True,
) -> Path:
    """Download only an exact `.set`/`.fdt` allowlist.

    This helper is intentionally separate from :func:`download_datasets`.
    It never removes missing include paths and never retries without include
    filters.  ``existing_paths`` are already downloaded paths that must remain
    in the final exact inventory, which allows a caller to download `.set`
    headers first and their declared `.fdt` companions second.
    """
    requested = _validate_explicit_payload_paths(payload_paths)
    existing = _validate_explicit_payload_paths(existing_paths) if existing_paths else []
    required = set(requested) | set(existing)
    dataset_path = Path(out_dir) / dataset_id
    dataset_path.mkdir(parents=True, exist_ok=True)
    actual_before, unexpected_before = _inventory_explicit_payload_files(dataset_path)
    if unexpected_before:
        raise RuntimeError(
            f"explicit_payload_target_contains_unexpected_files:{dataset_id}:"
            f"{unexpected_before[0]}"
        )
    actual_before_set = set(actual_before)
    unexpected_payloads = sorted(actual_before_set - required)
    if unexpected_payloads:
        companion_payloads = [
            path
            for path in unexpected_payloads
            if path.lower().endswith(".fdt")
            and path[:-4] + ".set" in required
        ]
        if len(companion_payloads) == len(unexpected_payloads):
            required.update(companion_payloads)
        else:
            raise RuntimeError(
                f"explicit_payload_target_contains_unallowlisted_payload:{dataset_id}:"
                f"{unexpected_payloads[0]}"
            )
    missing = sorted(required - actual_before_set)
    attempts = 0
    while missing:
        try:
            _perform_openneuro_download(
                dataset_id,
                dataset_path,
                missing,
                [],
                use_uvx=use_uvx,
            )
        except MissingIncludePatternError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve retry boundary
            attempts += 1
            if attempts >= max(1, int(retries)):
                raise RuntimeError(
                    f"Failed explicit payload download for {dataset_id} "
                    f"after {retries} attempts ({exc})"
                ) from exc
            logger.warning(
                "Explicit payload download attempt %s/%s failed for %s: %s",
                attempts,
                retries,
                dataset_id,
                exc,
            )
        actual_after, unexpected_after = _inventory_explicit_payload_files(dataset_path)
        if unexpected_after:
            raise RuntimeError(
                f"explicit_payload_download_created_unexpected_files:{dataset_id}:"
                f"{unexpected_after[0]}"
            )
        missing = sorted(required - set(actual_after))
        if missing:
            attempts += 1
            if attempts >= max(1, int(retries)):
                raise RuntimeError(
                    f"explicit_payload_download_incomplete:{dataset_id}:"
                    f"{missing}"
                )
    actual, unexpected = _inventory_explicit_payload_files(dataset_path)
    actual_set = set(actual)
    if unexpected:
        raise RuntimeError(
            f"explicit_payload_download_created_unexpected_files:{dataset_id}:"
            f"{unexpected[0]}"
        )
    missing = sorted(required - actual_set)
    extra = sorted(actual_set - required)
    if missing or extra:
        raise RuntimeError(
            f"explicit_payload_inventory_mismatch:{dataset_id}:"
            f"missing={missing}:extra={extra}"
        )
    return dataset_path


def _inventory_explicit_payload_files(root: Path) -> tuple[List[str], List[str]]:
    """Inventory a strict payload-only target directory."""
    payloads: List[str] = []
    unexpected: List[str] = []
    if not root.exists():
        return payloads, unexpected
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root)).replace("\\", "/")
        if path.suffix.lower() in {".set", ".fdt"}:
            payloads.append(relative)
        elif path.name not in _EXPLICIT_DOWNLOAD_BOOKKEEPING_NAMES:
            unexpected.append(relative)
    return payloads, unexpected


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
    dl_cfg = config.get("download", {}) if isinstance(config, dict) else {}
    retries = dl_cfg.get("retries", 3)
    use_uvx = bool(dl_cfg.get("use_uvx", True))
    env_override = os.environ.get("OPENNEURO_PREFER_UVX")
    if env_override is not None and env_override.strip():
        use_uvx = env_override.strip().lower() in ("1", "true", "yes", "on")

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
                _perform_openneuro_download(
                    ds_id,
                    dataset_path,
                    sanitized_includes,
                    exclude_patterns,
                    use_uvx=use_uvx,
                )
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



def _resolve_openneuro_cli(prefer_uvx: bool = True) -> List[str]:
    """Return the command prefix for invoking the openneuro-py CLI.

    When ``prefer_uvx`` is True (default), return ``["uvx", "openneuro-py@latest"]``
    so the pipeline runs the latest released openneuro-py in an isolated
    environment. This sidesteps installed versions whose GraphQL metadata
    query is incompatible with the current OpenNeuro server schema.

    When ``prefer_uvx`` is False, resolve a locally installed
    ``openneuro-py`` executable in this order:
      1. interpreter Scripts dir (venv) — ``openneuro-py(.exe)``
      2. per-user Scripts dir (pip ``--user`` installs)
      3. ``shutil.which("openneuro-py")`` (PATH)
      4. ``uvx openneuro-py@latest`` (always-available fallback)
    """
    if not prefer_uvx:
        exe_name = "openneuro-py.exe" if sys.platform.startswith("win") else "openneuro-py"
        # 1. venv / interpreter Scripts
        cand = Path(sys.executable).parent / exe_name
        if cand.exists():
            return [str(cand)]
        # 2. per-user Scripts
        for base in _user_script_dirs():
            c = base / exe_name
            if c.exists():
                return [str(c)]
        # 3. PATH
        which = shutil.which("openneuro-py")
        if which:
            return [which]

    # 4. uvx (preferred, or last-resort fallback)
    if shutil.which("uvx") is None:
        raise RuntimeError(
            "uvx not found on PATH. Install uv (https://docs.astral.sh/uv/) or set "
            "download.use_uvx: false after installing a working openneuro-py."
        )
    return ["uvx", "openneuro-py@latest"]

