"""Fallback downloader for OpenNeuro presigned S3 URL scripts.

Why this exists
---------------
Sometimes OpenNeuro support provides a "download script" containing many `curl`
commands with presigned S3 URLs, e.g.:

  curl --create-dirs https://s3.amazonaws.com/openneuro.org/dsXXXXXX/... -o dsXXXXXX-<version>/path/to/file

Those scripts are often awkward to run on Windows (and may contain characters
like `&` that can be misinterpreted by shells when unquoted). This tool parses
the script and downloads the referenced files directly via Python.

Default behaviour is designed to match the rest of this repo:
- write into config `paths.received_dir` (or `--data-dir`)
- place files under `<received_dir>/<dataset_id>/...` when dataset id is known
- optionally build `file_index.csv` into `<processed_dir>/<dataset_id>/...`

New in v1.1: "SourceTree → curl list → download" mode
-----------------------------------------------------
Some datasets fail with openneuro-py (include filters, git-annex issues).
When you can obtain a lightweight dataset tree (e.g. via a Git clone of
`https://github.com/OpenNeuroDatasets/<ds>.git` or a DataLad clone),
you can generate a public S3 curl list from that tree (equivalent to
`openneuro/scripts/make_s3_curl_list_from_tree.ps1`) and download it
with the same retry/parallel logic.

This tool can now:
- parse existing `curl ... -o ...` scripts (presigned or public S3)
- OR generate such a list from a local SourceTree (optionally cloning it)
  and download in one shot.

This is intentionally separate from `noetic_ingest.download` because the
presigned script is an external artifact (not something we can always generate
programmatically from the OpenNeuro API).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

from filelock import FileLock

logger = logging.getLogger("presigned_fallback")


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "config_ingest.yaml"


def _load_config(config_path: Path) -> dict:
    try:
        from core import config_loader

        return config_loader.load_config(config_path)
    except Exception as exc:
        logger.warning("Failed to load config %s: %s", config_path, exc)
        return {}


def _resolve_data_dir(config: dict, override: Path | None) -> Path:
    if override is not None:
        return override
    paths_cfg = config.get("paths", {}) if isinstance(config, dict) else {}
    data_dir = paths_cfg.get("received_dir")
    if data_dir:
        return Path(str(data_dir))
    return Path.cwd()


def _resolve_processed_dir(config: dict, override: Path | None, default_base: Path) -> Path:
    if override is not None:
        return override
    paths_cfg = config.get("paths", {}) if isinstance(config, dict) else {}
    processed_dir = paths_cfg.get("processed_dir")
    if processed_dir:
        return Path(str(processed_dir))
    return default_base.parent / "processed"


@dataclass(frozen=True)
class _DownloadSpec:
    url: str
    out_rel_posix: str


def _encode_s3_path(rel_posix: str) -> str:
    # Preserve '/' but escape any other reserved characters.
    return urllib.parse.quote(rel_posix, safe="/-_.~")


def _list_tree_files(
    source_root: Path,
    *,
    include_derivatives: bool,
    include_regex: str | None,
    exclude_regex: str | None,
) -> list[str]:
    """Return POSIX-style relative file paths under source_root.

    Mirrors `openneuro/scripts/make_s3_curl_list_from_tree.ps1`.
    """
    root = Path(source_root)
    if not root.exists():
        raise FileNotFoundError(f"SourceTree not found: {root}")

    rx_inc = re.compile(include_regex) if include_regex else None
    rx_exc = re.compile(exclude_regex) if exclude_regex else None

    rels: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(root)
        except Exception:
            continue

        # Drop internal dirs
        parts_lower = [str(x).lower() for x in rel.parts]
        if ".git" in parts_lower or ".datalad" in parts_lower:
            continue
        if not include_derivatives and any(str(x).lower() == "derivatives" for x in rel.parts):
            continue

        rel_posix = PurePosixPath(*rel.parts).as_posix()
        if rx_inc and not rx_inc.search(rel_posix):
            continue
        if rx_exc and rx_exc.search(rel_posix):
            continue
        rels.append(rel_posix)

    rels = sorted(set(rels))
    if not rels:
        raise RuntimeError("No files matched. Check SourceTree / filters.")
    return rels


def _generate_specs_from_tree(
    *,
    dataset_id: str,
    source_root: Path,
    s3_base: str,
    include_derivatives: bool,
    include_regex: str | None,
    exclude_regex: str | None,
) -> list[_DownloadSpec]:
    rels = _list_tree_files(
        source_root,
        include_derivatives=include_derivatives,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
    )
    base = str(s3_base).rstrip("/")
    specs: list[_DownloadSpec] = []
    for rel_posix in rels:
        url = f"{base}/{dataset_id}/{_encode_s3_path(rel_posix)}"
        specs.append(_DownloadSpec(url=url, out_rel_posix=rel_posix))
    return specs


def _ensure_git_available() -> None:
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True, text=True)
    except Exception as exc:
        raise RuntimeError("git is required for --clone-source-tree (install Git and ensure it's on PATH).") from exc


def _clone_source_tree_if_needed(
    *,
    dataset_id: str,
    dest_dir: Path,
    github_repo_base: str,
    update: bool,
) -> Path:
    """Clone https://github.com/OpenNeuroDatasets/<dataset_id>.git into dest_dir/<dataset_id>.

    This is a lightweight clone used only as a file manifest. It does NOT download annex content.
    """
    _ensure_git_available()

    repo = f"{str(github_repo_base).rstrip('/')}/{dataset_id}.git"
    dest = Path(dest_dir) / dataset_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dest.exists() and (dest / ".git").exists():
        if update:
            # best-effort, keep it simple and safe
            try:
                subprocess.run(["git", "-C", str(dest), "fetch", "--all", "--prune"], check=True)
                subprocess.run(["git", "-C", str(dest), "reset", "--hard", "origin/master"], check=False)
                subprocess.run(["git", "-C", str(dest), "reset", "--hard", "origin/main"], check=False)
            except Exception:
                # Don't fail the whole run if update fails; tree still usable.
                logger.warning("Failed to update existing SourceTree at %s; using as-is.", dest)
        return dest

    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(f"SourceTree dest exists but is not a git repo: {dest}")

    # Shallow clone; large repos still provide full path listing for current HEAD.
    cmd = ["git", "clone", "--depth", "1", repo, str(dest)]
    logger.info("Cloning SourceTree: %s -> %s", repo, dest)
    subprocess.run(cmd, check=True)
    return dest


def _redact_url(url: str) -> str:
    """Remove sensitive query params from logs (e.g., Signature)."""
    try:
        parsed = urllib.parse.urlsplit(url)
    except Exception:
        return "<invalid-url>"
    # Keep scheme/netloc/path; drop query/fragment
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _join_continued_lines(lines: Iterable[str]) -> list[str]:
    """Join lines ending with backslash as shell-style continuations."""
    out: list[str] = []
    buf: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.rstrip()
        if not stripped:
            # flush buffer if any
            if buf:
                out.append(" ".join(buf).strip())
                buf = []
            continue
        if stripped.endswith("\\"):
            buf.append(stripped[:-1].rstrip())
            continue
        if buf:
            buf.append(stripped)
            out.append(" ".join(buf).strip())
            buf = []
        else:
            out.append(stripped)
    if buf:
        out.append(" ".join(buf).strip())
    return out


def _parse_curl_line(line: str) -> _DownloadSpec | None:
    s = (line or "").strip()
    if not s:
        return None
    if s.startswith("#!"):
        return None
    if s.startswith("#"):
        return None
    if "curl" not in s:
        return None

    try:
        tokens = shlex.split(s, posix=True)
    except Exception:
        # If parsing fails, ignore the line but log once at debug level.
        logger.debug("Unable to parse line: %s", s)
        return None
    if not tokens:
        return None
    # Allow: "curl ..." or "/usr/bin/curl ..."
    if Path(tokens[0]).name != "curl":
        if "curl" not in tokens[0]:
            return None

    url: str | None = None
    for tok in tokens:
        if tok.startswith("https://") or tok.startswith("http://"):
            url = tok
            break
    if not url:
        return None

    out_path: str | None = None
    # Support: -o <path> and --output <path>
    for i, tok in enumerate(tokens):
        if tok == "-o" and i + 1 < len(tokens):
            out_path = tokens[i + 1]
            break
        if tok == "--output" and i + 1 < len(tokens):
            out_path = tokens[i + 1]
            break
    # Support: -O (remote name)
    if out_path is None and "-O" in tokens:
        try:
            parsed = urllib.parse.urlsplit(url)
            out_path = Path(parsed.path).name or None
        except Exception:
            out_path = None

    if not out_path:
        return None

    return _DownloadSpec(url=url, out_rel_posix=str(out_path))


def _parse_script(script_path: Path) -> list[_DownloadSpec]:
    text = script_path.read_text(encoding="utf-8", errors="replace")
    lines = _join_continued_lines(text.splitlines())
    specs: list[_DownloadSpec] = []
    for line in lines:
        spec = _parse_curl_line(line)
        if spec is not None:
            specs.append(spec)
    return specs


def _safe_rel_path(posix_rel: str) -> Path:
    """Convert a POSIX-ish relative path from the script to a safe Path.

    Reject absolute paths and any path that escapes via '..'.
    """
    rel = PurePosixPath(str(posix_rel).strip())
    if not rel.parts:
        raise ValueError("Empty output path in script")
    if rel.is_absolute():
        raise ValueError(f"Absolute output paths are not allowed: {posix_rel}")
    if any(p == ".." for p in rel.parts):
        raise ValueError(f"Parent traversal is not allowed in output paths: {posix_rel}")
    # Also reject Windows drive syntax sneaking into a string.
    if ":" in rel.parts[0]:
        raise ValueError(f"Suspicious output path (drive letter?): {posix_rel}")
    return Path(*rel.parts)


def _common_first_component(paths: Sequence[Path]) -> str | None:
    if not paths:
        return None
    first = paths[0].parts[0] if paths[0].parts else None
    if not first:
        return None
    for p in paths[1:]:
        if not p.parts or p.parts[0] != first:
            return None
    return first


def _is_derivatives_path(rel: Path) -> bool:
    return any(str(p).lower() == "derivatives" for p in rel.parts)


def _infer_dataset_id_from_component(comp: str) -> str | None:
    # Common pattern: ds004504-1.0.8 or ds004504
    if not comp.startswith("ds"):
        return None
    # Keep only leading ds + digits
    i = 2
    while i < len(comp) and comp[i].isdigit():
        i += 1
    ds = comp[:i]
    if ds == "ds" or len(ds) < 4:
        return None
    return ds


def _download_to_path(
    url: str,
    dest_path: Path,
    *,
    timeout_s: float,
    retries: int,
    backoff_s: float,
    overwrite: bool,
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a lock to avoid two workers writing the same file if scripts overlap.
    lock_path = dest_path.with_suffix(dest_path.suffix + ".lock")
    with FileLock(str(lock_path)):
        if dest_path.exists() and not overwrite:
            return

        tmp_path = dest_path.with_name(dest_path.name + ".part")
        if tmp_path.exists() and overwrite:
            try:
                tmp_path.unlink()
            except Exception:
                pass

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "noetic-ingest-presigned-fallback/1.0",
                        "Accept": "*/*",
                    },
                )
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    # Stream to disk
                    with tmp_path.open("wb") as f:
                        while True:
                            chunk = resp.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                tmp_path.replace(dest_path)
                return
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = exc
                # Retry only when attempts remain
                if attempt >= retries:
                    break
                sleep_s = backoff_s * (2**attempt)
                time.sleep(sleep_s)
        raise RuntimeError(f"Download failed for {_redact_url(url)} -> {dest_path}: {last_err}") from last_err


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download OpenNeuro presigned S3 URL scripts (curl list) via Python.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--script",
        type=Path,
        action="append",
        default=None,
        help="Path to a shell script (or text file) containing `curl ... -o ...` lines. Can be given multiple times.",
    )
    mode.add_argument(
        "--generate-from-tree",
        action="store_true",
        help="Generate a public S3 curl list from a local SourceTree (Python equivalent of make_s3_curl_list_from_tree.ps1) and download it.",
    )
    p.add_argument(
        "--dataset",
        nargs="*",
        default=None,
        help="Dataset id(s) (e.g., ds004504). If provided with multiple --script, count must match or provide a single id applied to all scripts.",
    )
    p.add_argument(
        "--source-tree",
        type=Path,
        default=None,
        help="Local dataset tree used as manifest when --generate-from-tree is set (should look like a dataset root).",
    )
    p.add_argument(
        "--clone-source-tree",
        action="store_true",
        help="When --generate-from-tree is set and --source-tree is not provided, clone https://github.com/OpenNeuroDatasets/<ds>.git into --source-tree-base-dir and use it as the manifest.",
    )
    p.add_argument(
        "--source-tree-base-dir",
        type=Path,
        default=None,
        help="Base directory for cloning SourceTrees (default: <data-dir>/_source_tree).",
    )
    p.add_argument(
        "--github-repo-base",
        type=str,
        default="https://github.com/OpenNeuroDatasets",
        help="Base URL for dataset git repos (default: https://github.com/OpenNeuroDatasets).",
    )
    p.add_argument(
        "--update-source-tree",
        action="store_true",
        help="If the SourceTree repo already exists, attempt a best-effort update before generating the file list.",
    )
    p.add_argument(
        "--s3-base",
        type=str,
        default="https://s3.amazonaws.com/openneuro.org",
        help="Base URL for OpenNeuro public bucket (default: https://s3.amazonaws.com/openneuro.org).",
    )
    p.add_argument(
        "--include-regex",
        type=str,
        default=None,
        help="Optional include filter (regex on POSIX relative path) for --generate-from-tree.",
    )
    p.add_argument(
        "--exclude-regex",
        type=str,
        default=None,
        help="Optional exclude filter (regex on POSIX relative path) for --generate-from-tree.",
    )
    p.add_argument(
        "--write-generated-script",
        type=Path,
        default=None,
        help="If set with --generate-from-tree, write the generated curl list to this path (UTF-8) for reproducibility.",
    )
    p.add_argument("--config", type=Path, default=_default_config_path(), help="Path to ingest YAML config (used for defaults).")
    p.add_argument("--data-dir", type=Path, default=None, help="Override received/raw base directory.")
    p.add_argument("--processed-dir", type=Path, default=None, help="Override processed/output base directory (used when --build-index).")
    p.add_argument("--jobs", type=int, default=8, help="Parallel download workers (default: 8).")
    p.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds (default: 60).")
    p.add_argument("--retries", type=int, default=2, help="Retries per file (default: 2).")
    p.add_argument("--backoff", type=float, default=1.0, help="Base backoff seconds (exponential) between retries (default: 1.0).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files (default: skip existing).")
    p.add_argument(
        "--strip-output-prefix",
        type=str,
        default=None,
        help="If set, strip this leading path component/prefix from script -o paths before writing.",
    )
    p.add_argument(
        "--strip-first-component",
        action="store_true",
        help="Strip the first path component from all script -o output paths (useful when script writes to dsXXXXXX-<ver>/...).",
    )
    p.add_argument(
        "--keep-script-layout",
        action="store_true",
        help="Write exactly to the script output paths under --data-dir (no dataset root rewrite).",
    )
    p.add_argument(
        "--include-derivatives",
        action="store_true",
        help="Include files under derivatives/ (default: skip derivatives to reduce size).",
    )
    p.add_argument("--build-index", action="store_true", help="After download, build BIDS file index (file_index.csv).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    p.add_argument("--dry-run", action="store_true", help="Parse and print what would be downloaded, without downloading.")
    return p.parse_args(argv)


def _normalize_dataset_args(scripts: Sequence[Path], dataset_args: Sequence[str] | None) -> list[str | None]:
    if not dataset_args:
        return [None for _ in scripts]
    ds_list = [str(x) for x in dataset_args if str(x).strip()]
    if not ds_list:
        return [None for _ in scripts]
    if len(ds_list) == 1 and len(scripts) > 1:
        return [ds_list[0] for _ in scripts]
    if len(ds_list) != len(scripts):
        raise ValueError("When providing multiple --script, you must provide either 1 dataset id or one per script.")
    return ds_list


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    config = _load_config(args.config)
    base_dir = _resolve_data_dir(config, args.data_dir)
    processed_dir = _resolve_processed_dir(config, args.processed_dir, base_dir)

    logger.info("Base data directory: %s", base_dir)
    if args.build_index:
        logger.info("Processed directory: %s", processed_dir)

    exit_code = 0
    try:
        # ------------------------------------------------------------------
        # Mode A: existing curl list script(s)
        # ------------------------------------------------------------------
        if args.script:
            scripts: list[Path] = [Path(p) for p in (args.script or [])]
            for sp in scripts:
                if not sp.exists():
                    logger.error("Script not found: %s", sp)
                    return 1

            ds_for_script = _normalize_dataset_args(scripts, args.dataset)

            for script_path, ds_id in zip(scripts, ds_for_script):
                try:
                    logger.info("Parsing script: %s", script_path)
                    specs = _parse_script(script_path)
                    if not specs:
                        raise RuntimeError(f"No curl downloads found in script: {script_path}")

                    rel_paths = [_safe_rel_path(s.out_rel_posix) for s in specs]
                    common_first = _common_first_component(rel_paths)

                    # Auto-infer dataset id from common first component when not provided.
                    inferred_ds = _infer_dataset_id_from_component(common_first) if common_first else None
                    if ds_id is None:
                        ds_id = inferred_ds

                    if args.keep_script_layout or ds_id is None:
                        dataset_root = base_dir
                    else:
                        dataset_root = base_dir / ds_id

                    # Decide stripping behaviour.
                    strip_prefix = args.strip_output_prefix
                    strip_first = bool(args.strip_first_component)
                    if strip_prefix is None and not strip_first:
                        # If script outputs all under dsXXXXXX-<ver>/..., strip that by default when
                        # we are rewriting into dataset_root/<dataset_id>/...
                        if ds_id is not None and common_first and common_first != ds_id:
                            if inferred_ds == ds_id:
                                strip_first = True

                    planned: list[tuple[str, Path]] = []
                    skipped_derivatives = 0
                    for spec, rel in zip(specs, rel_paths):
                        out_rel = rel
                        if strip_prefix:
                            pref = Path(*PurePosixPath(strip_prefix).parts)
                            try:
                                out_rel = out_rel.relative_to(pref)
                            except Exception:
                                pass
                        if strip_first and out_rel.parts:
                            out_rel = Path(*out_rel.parts[1:]) if len(out_rel.parts) > 1 else Path(out_rel.name)
                        if not args.include_derivatives and _is_derivatives_path(out_rel):
                            skipped_derivatives += 1
                            continue
                        dest = dataset_root / out_rel
                        planned.append((spec.url, dest))

                    if skipped_derivatives and not args.include_derivatives:
                        logger.info("Skipping derivatives/: %d file(s)", skipped_derivatives)
                    logger.info("Planned downloads: %d file(s)", len(planned))
                    if not planned:
                        raise RuntimeError(
                            "No files left to download after filtering. "
                            "If your script only contains derivatives/, pass --include-derivatives."
                        )
                    if args.dry_run:
                        for url, dest in planned[:50]:
                            logger.info("DRY RUN: %s -> %s", _redact_url(url), dest)
                        if len(planned) > 50:
                            logger.info("DRY RUN: ... (%d more)", len(planned) - 50)
                        continue

                    # Parallel download
                    try:
                        from tqdm import tqdm  # type: ignore
                    except Exception:
                        tqdm = None  # type: ignore

                    bar = tqdm(total=len(planned), desc=f"{ds_id or script_path.name}", unit="file") if tqdm else None
                    with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
                        futs = [
                            ex.submit(
                                _download_to_path,
                                url,
                                dest,
                                timeout_s=float(args.timeout),
                                retries=int(args.retries),
                                backoff_s=float(args.backoff),
                                overwrite=bool(args.overwrite),
                            )
                            for (url, dest) in planned
                        ]
                        for fut in as_completed(futs):
                            try:
                                fut.result()
                            except Exception as exc:
                                logger.error("Download error: %s", exc)
                                exit_code = 1
                            finally:
                                if bar is not None:
                                    bar.update(1)
                    if bar is not None:
                        bar.close()

                    # Optional file index build
                    if args.build_index:
                        if ds_id is None:
                            raise RuntimeError("--build-index requires --dataset (or an inferable dataset id).")
                        try:
                            from openneuro import bids_index
                        except Exception as exc:  # pragma: no cover - import failure
                            raise RuntimeError("Unable to import bids_index module") from exc
                        ds_root = base_dir / ds_id
                        logger.info("Building file_index.csv for %s", ds_root)
                        index_df = bids_index.build_file_index(ds_root)
                        out_dir = processed_dir / ds_id
                        out_dir.mkdir(parents=True, exist_ok=True)
                        index_path = out_dir / "file_index.csv"
                        index_df.to_csv(index_path, index=False)
                        logger.info("Saved index to %s (%d rows)", index_path, len(index_df))

                except Exception as exc:
                    logger.error("Failed processing script %s: %s", script_path, exc)
                    exit_code = 1

        # ------------------------------------------------------------------
        # Mode B: generate from SourceTree (optional clone) and download
        # ------------------------------------------------------------------
        if args.generate_from_tree:
            ds_list = [str(x) for x in (args.dataset or []) if str(x).strip()]
            if len(ds_list) != 1:
                raise RuntimeError("--generate-from-tree requires exactly one --dataset id (e.g. --dataset ds003059)")
            ds_id = ds_list[0]

            source_tree = args.source_tree
            if source_tree is None:
                if not args.clone_source_tree:
                    raise RuntimeError("--generate-from-tree requires --source-tree or --clone-source-tree")
                base = args.source_tree_base_dir
                if base is None:
                    base = base_dir / "_source_tree"
                source_tree = _clone_source_tree_if_needed(
                    dataset_id=ds_id,
                    dest_dir=Path(base),
                    github_repo_base=str(args.github_repo_base),
                    update=bool(args.update_source_tree),
                )

            logger.info("Generating curl list from SourceTree: %s", source_tree)
            specs = _generate_specs_from_tree(
                dataset_id=ds_id,
                source_root=Path(source_tree),
                s3_base=str(args.s3_base),
                include_derivatives=bool(args.include_derivatives),
                include_regex=args.include_regex,
                exclude_regex=args.exclude_regex,
            )

            if args.write_generated_script:
                out_path = Path(args.write_generated_script)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                lines = [f"curl --create-dirs {s.url} -o {s.out_rel_posix}" for s in specs]
                out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                logger.info("Wrote generated curl list: %s (%d lines)", out_path, len(lines))

            # In tree mode we always write to <data-dir>/<dataset_id>/<rel>
            dataset_root = base_dir / ds_id
            planned = [(s.url, dataset_root / _safe_rel_path(s.out_rel_posix)) for s in specs]
            logger.info("Planned downloads: %d file(s)", len(planned))
            if args.dry_run:
                for url, dest in planned[:50]:
                    logger.info("DRY RUN: %s -> %s", _redact_url(url), dest)
                if len(planned) > 50:
                    logger.info("DRY RUN: ... (%d more)", len(planned) - 50)
                return 0 if exit_code == 0 else 1

            try:
                from tqdm import tqdm  # type: ignore
            except Exception:
                tqdm = None  # type: ignore

            bar = tqdm(total=len(planned), desc=f"{ds_id}", unit="file") if tqdm else None
            with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
                futs = [
                    ex.submit(
                        _download_to_path,
                        url,
                        dest,
                        timeout_s=float(args.timeout),
                        retries=int(args.retries),
                        backoff_s=float(args.backoff),
                        overwrite=bool(args.overwrite),
                    )
                    for (url, dest) in planned
                ]
                for fut in as_completed(futs):
                    try:
                        fut.result()
                    except Exception as exc:
                        logger.error("Download error: %s", exc)
                        exit_code = 1
                    finally:
                        if bar is not None:
                            bar.update(1)
            if bar is not None:
                bar.close()

            if args.build_index:
                try:
                    from openneuro import bids_index
                except Exception as exc:  # pragma: no cover - import failure
                    raise RuntimeError("Unable to import bids_index module") from exc
                ds_root = base_dir / ds_id
                logger.info("Building file_index.csv for %s", ds_root)
                index_df = bids_index.build_file_index(ds_root)
                out_dir = processed_dir / ds_id
                out_dir.mkdir(parents=True, exist_ok=True)
                index_path = out_dir / "file_index.csv"
                index_df.to_csv(index_path, index=False)
                logger.info("Saved index to %s (%d rows)", index_path, len(index_df))

    except Exception as exc:
        logger.error("%s", exc)
        return 1

    if exit_code == 0:
        logger.info("All scripts processed successfully.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())


