"""BIDS index builder for EEG, MEG, and fMRI datasets.

Walks a downloaded dataset root and builds a tabular file index (subject,
session, task, run, modality, paths to sidecars such as ``*_eeg.json``,
``channels.tsv``, ``events.tsv``, optional hashes/sizes).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from .bdf_adapter import parse_bdf_entities

logger = logging.getLogger(__name__)


def resolve_dataset_root(
    config: Mapping[str, Any],
    received_dir: Path,
    dataset_id: str,
) -> Path:
    """Resolve the on-disk root for one dataset.

    Default layout is ``<received_dir>/<dataset_id>``. For non-BIDS sources
    (for example local sleep datasets), allow explicit per-dataset roots via:

    - ``paths.dataset_received_dirs.<dataset_id>`` (preferred)
    - ``paths.dataset_roots.<dataset_id>.received`` (legacy-compatible)
    """
    try:
        paths_cfg = config.get("paths", {}) if isinstance(config, Mapping) else {}
        if isinstance(paths_cfg, Mapping):
            ds_dirs = paths_cfg.get("dataset_received_dirs", {})
            if isinstance(ds_dirs, Mapping):
                raw = ds_dirs.get(dataset_id)
                if raw:
                    return Path(str(raw))

            ds_roots = paths_cfg.get("dataset_roots", {})
            if isinstance(ds_roots, Mapping):
                ds_cfg = ds_roots.get(dataset_id, {})
                if isinstance(ds_cfg, Mapping):
                    raw2 = ds_cfg.get("received")
                    if raw2:
                        return Path(str(raw2))
    except Exception:
        logger.exception("Failed to resolve dataset root override for %s", dataset_id)

    return Path(received_dir) / dataset_id


def _should_skip_relpath(rel_path: Path) -> bool:
    """Return True for paths that should never be processed/indexed.

    This primarily filters out macOS metadata files such as AppleDouble (prefix
    ``._``) and other dotfiles (e.g. ``.DS_Store``) that can accidentally match
    our glob patterns (e.g. ``*_bold.nii.gz``) but are not real data files.
    """
    parts = rel_path.parts
    if any(part.startswith(".") for part in parts):
        return True
    # Safety net (in case Path.parts behaviour differs or future refactors pass
    # absolute paths here)
    if rel_path.name.startswith("._"):
        return True
    return False


def _compute_md5_stream(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute MD5 hash in chunks to avoid loading entire files into memory."""
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _maybe_compute_md5(path: Path, size_bytes: int, max_bytes: int) -> Optional[str]:
    """Compute MD5 only when file size is below a safety threshold.

    Indexing large fMRI NIfTI files can be extremely slow if we hash every file.
    For ingest purposes we primarily need stable paths + sidecars; file size
    already provides a coarse integrity signal. MD5 is therefore best-effort.
    """
    try:
        if size_bytes <= 0:
            return None
        if size_bytes > max_bytes:
            return None
        return _compute_md5_stream(path)
    except Exception as e:
        logger.warning(f"Could not compute hash for {path}: {e}")
        return None


def _extract_subject_candidate(token: str) -> Optional[str]:
    """Internal helper: extract subject candidate."""
    m = re.search(r"\b(?:sub-)?([A-Za-z]+[0-9]{2,})\b", str(token))
    if not m:
        token_str = str(token).strip()
        if token_str.isdigit() and len(token_str) >= 3:
            return token_str
        return None
    candidate = m.group(1)
    low = candidate.lower()
    if low.startswith(("run", "task", "ses", "acq")):
        return None
    return candidate


def _infer_non_bids_subject(parts: tuple[str, ...], stem: str) -> Optional[str]:
    """Best-effort subject fallback for non-BIDS datasets."""
    parts_list = list(parts)
    lowered = [str(p).lower() for p in parts_list]
    for anchor in ("subjects", "subject"):
        if anchor in lowered:
            idx = lowered.index(anchor)
            for token in parts_list[idx + 1 : idx + 3]:
                cand = _extract_subject_candidate(str(token))
                if cand:
                    return cand
    for token in parts_list + [stem]:
        cand = _extract_subject_candidate(str(token))
        if cand:
            return cand
    return None


def _pick_best_fmri_events_tsv(
    parent: Path,
    stem: str,
    task: Optional[str],
    run: Optional[str],
    acq: Optional[str],
) -> Optional[Path]:
    """Internal helper: pick best fmri events tsv."""
    direct = parent / f"{stem}_events.tsv"
    if direct.exists():
        return direct

    legacy = parent / "events.tsv"
    if legacy.exists():
        return legacy

    candidates = [p for p in sorted(parent.glob("*events.tsv")) if p.is_file() and not p.name.startswith("._")]
    if not candidates:
        return None

    def _score(p: Path) -> int:
        """Internal helper: score."""
        name = p.name.lower()
        s = 0
        if stem.lower() in name:
            s += 4
        if task and f"task-{str(task).lower()}" in name:
            s += 2
        if run and f"run-{str(run).lower()}" in name:
            s += 1
        if acq and f"acq-{str(acq).lower()}" in name:
            s += 1
        return s

    return max(candidates, key=_score)


def _parse_bids_entities_from_stem(stem: str) -> Dict[str, Optional[str]]:
    """Extract common BIDS entities from a file stem."""
    entities: Dict[str, Optional[str]] = {
        "subject": None,
        "session": None,
        "task": None,
        "run": None,
        "acq": None,
    }
    for part in stem.split("_"):
        if part.startswith("sub-"):
            entities["subject"] = part[4:]
        elif part.startswith("ses-"):
            entities["session"] = part[4:]
        elif part.startswith("task-"):
            entities["task"] = part[5:]
        elif part.startswith("run-"):
            entities["run"] = part[4:]
        elif part.startswith("acq-"):
            entities["acq"] = part[4:]
    return entities


def _infer_bids_datatype(rel_path: Path) -> Optional[str]:
    """Infer the BIDS datatype folder from a relative path."""
    lowered = [str(part).strip().lower() for part in rel_path.parts]
    for name in ("eeg", "meg", "ecg", "pupil", "beh", "func"):
        if name in lowered:
            return "fmri" if name == "func" else name
    return None


def _build_bundle_key(
    subject: Optional[str],
    session: Optional[str],
    task: Optional[str],
    run: Optional[str],
    acq: Optional[str],
) -> str:
    """Build a stable grouping key for multimodal recordings."""
    return "|".join(
        [
            str(subject or ""),
            str(session or ""),
            str(task or ""),
            str(run or ""),
            str(acq or ""),
        ]
    )


def _probe_nwb_index_metadata(nwb_file: Path) -> Dict[str, str]:
    """Return lightweight NWB metadata for file-index provenance."""
    meta = {
        "nwb_acquisitions": "",
        "nwb_processing_modules": "",
        "nwb_intervals": "",
        "nwb_devices": "",
        "nwb_modality_hints": "",
        "nwb_probe_error": "",
    }
    try:
        import h5py

        with h5py.File(nwb_file, "r") as handle:
            acquisitions = sorted(str(k) for k in handle.get("acquisition", {}).keys())
            processing = sorted(str(k) for k in handle.get("processing", {}).keys())
            intervals = sorted(str(k) for k in handle.get("intervals", {}).keys())
            devices = sorted(str(k) for k in handle.get("general/devices", {}).keys())
        search_space = " ".join(acquisitions + processing + intervals + devices).lower()
        hints: list[str] = []
        if "ecephys" in search_space or "electrical" in search_space or "lfp" in search_space:
            hints.append("ecephys")
        if "eeg" in search_space:
            hints.append("eeg")
        if "emg" in search_space:
            hints.append("emg")
        if "ophys" in search_space or "image" in search_space:
            hints.append("ophys")
        if "behavior" in search_space:
            hints.append("behavior")
        meta.update(
            {
                "nwb_acquisitions": "|".join(acquisitions),
                "nwb_processing_modules": "|".join(processing),
                "nwb_intervals": "|".join(intervals),
                "nwb_devices": "|".join(devices),
                "nwb_modality_hints": "|".join(dict.fromkeys(hints)),
            }
        )
    except Exception as exc:
        meta["nwb_probe_error"] = str(exc)
    return meta


def _resolve_fmri_indexing_options(
    config: Mapping[str, Any] | None,
    dataset_id: str | None,
) -> tuple[list[str], bool]:
    """Resolve fMRI index patterns and layout constraints from config."""
    bold_patterns = ["*_bold.nii.gz", "*_bold.nii"]
    require_func_dir = True
    if not isinstance(config, Mapping):
        return bold_patterns, require_func_dir

    indexing_cfg = config.get("indexing", {}) if isinstance(config, Mapping) else {}
    fmri_cfg = indexing_cfg.get("fmri", {}) if isinstance(indexing_cfg, Mapping) else {}
    if isinstance(fmri_cfg, Mapping):
        raw = fmri_cfg.get("bold_patterns")
        if isinstance(raw, list) and raw:
            bold_patterns = [str(x) for x in raw if str(x).strip()]
        if "require_func_dir" in fmri_cfg:
            require_func_dir = bool(fmri_cfg.get("require_func_dir"))

        ds_map = fmri_cfg.get("datasets", {})
        if dataset_id and isinstance(ds_map, Mapping):
            ds_cfg = ds_map.get(dataset_id, {})
            if isinstance(ds_cfg, Mapping):
                raw_ds = ds_cfg.get("bold_patterns")
                if isinstance(raw_ds, list) and raw_ds:
                    bold_patterns = [str(x) for x in raw_ds if str(x).strip()]
                if "require_func_dir" in ds_cfg:
                    require_func_dir = bool(ds_cfg.get("require_func_dir"))
    return bold_patterns, require_func_dir


def build_file_index(
    dataset_root: Path,
    config: Mapping[str, Any] | None = None,
    dataset_id: str | None = None,
) -> pd.DataFrame:
    """Build a file index for EEG/MEG and fMRI files under a BIDS dataset root.

    Args:
        dataset_root: Path to the dataset root directory.
        config: Optional ingest config (pattern overrides per modality/dataset).
        dataset_id: Optional dataset id for per-dataset config branches.

    Returns:
        DataFrame with ``path``, ``subject``, ``session``, ``task``, ``run``,
        ``acq``, ``modality``, ``md5``, ``size``, and sidecar path columns.
    """
    records: List[Dict[str, Any]] = []
    root = Path(dataset_root)
    # Avoid hashing huge files during indexing (especially fMRI NIfTIs).
    # 64 MiB default is enough to hash EEG headers/sidecars quickly but skips
    # multi-hundred-MB BOLD volumes.
    md5_max_bytes = int(64 * 1024 * 1024)

    # ------------------------------------------------------------------
    # Electrophysiology files. The directory datatype determines whether a file
    # is treated as EEG, MEG, or ECG in the index.
    # ------------------------------------------------------------------
    signal_extensions = [".edf", ".vhdr", ".set", ".bdf", ".hea", ".fif"]
    for ext in signal_extensions:
        for eeg_file in root.rglob(f"*{ext}"):
            try:
                rel_path = eeg_file.relative_to(root)
            except ValueError:
                continue

            if _should_skip_relpath(rel_path):
                continue
            parts = rel_path.parts
            if "derivatives" in parts:
                continue

            # WFDB headers (*.hea) require a paired signal file (*.mat or *.dat).
            # During active downloads we may temporarily have header-only records;
            # skip those to avoid guaranteed preprocessing failures.
            if eeg_file.suffix.lower() == ".hea":
                has_paired_signal = False
                for signal_ext in (".mat", ".dat"):
                    paired = eeg_file.with_suffix(signal_ext)
                    if paired.exists():
                        has_paired_signal = True
                        break
                if not has_paired_signal:
                    continue

            # Parse BIDS filename
            stem_parts = eeg_file.stem.split("_")
            subject: Optional[str] = None
            session: Optional[str] = None
            task: Optional[str] = None
            run: Optional[str] = None
            acq: Optional[str] = None

            for part in stem_parts:
                if part.startswith("sub-"):
                    subject = part[4:]
                elif part.startswith("ses-"):
                    session = part[4:]
                elif part.startswith("task-"):
                    task = part[5:]
                elif part.startswith("run-"):
                    run = part[4:]
                elif part.startswith("acq-"):
                    acq = part[4:]

            # Non-BIDS BDF collections may declare a reusable adapter policy
            # with a mapping table and/or named-group filename regex.  Apply it
            # before the generic fallback so source-provided session labels win.
            if eeg_file.suffix.lower() == ".bdf":
                adapter_entities = parse_bdf_entities(
                    eeg_file,
                    config or {},
                    dataset_id,
                    dataset_root=root,
                )
                subject = adapter_entities.get("subject") or subject
                session = adapter_entities.get("session") or session
                task = adapter_entities.get("task") or task
                run = adapter_entities.get("run") or run
                acq = adapter_entities.get("acq") or acq

            # Non-BIDS fallback: infer subject token from path segments/file stem
            # (e.g. ANPHY `Subjects/EPCTL01/...`).
            if subject is None:
                subject = _infer_non_bids_subject(parts, eeg_file.stem)
            datatype = _infer_bids_datatype(rel_path) or "eeg"
            modality = datatype if datatype in {"eeg", "meg", "ecg"} else "eeg"

            # Find sidecars
            signal_json = eeg_file.with_suffix(".json")
            if not signal_json.exists():
                signal_json = None
            # BIDS typically uses per-recording sidecars:
            #   sub-*_task-*_channels.tsv, sub-*_task-*_events.tsv
            # Some datasets may also have legacy "channels.tsv"/"events.tsv" in the folder.
            base_stem = eeg_file.stem
            base_core = base_stem
            if base_stem.endswith("_eeg"):
                base_core = base_stem[:-4]
            elif base_stem.endswith("_ieeg"):
                base_core = base_stem[:-5]
            elif base_stem.endswith("_meg"):
                base_core = base_stem[:-4]

            channels_tsv = eeg_file.parent / f"{base_core}_channels.tsv"
            if not channels_tsv.exists():
                channels_tsv = eeg_file.parent / "channels.tsv"
                if not channels_tsv.exists():
                    channels_tsv = None

            events_tsv = eeg_file.parent / f"{base_core}_events.tsv"
            if not events_tsv.exists():
                events_tsv = eeg_file.parent / "events.tsv"
                if not events_tsv.exists():
                    events_tsv = None

            # Compute file size and md5
            try:
                size = eeg_file.stat().st_size
                md5_hash = _maybe_compute_md5(eeg_file, size_bytes=size, max_bytes=md5_max_bytes)
            except Exception as e:
                logger.warning(f"Could not stat {eeg_file}: {e}")
                size = 0
                md5_hash = None

            records.append(
                {
                    "path": str(eeg_file.relative_to(root)),
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "run": run,
                    "acq": acq,
                    "modality": modality,
                    "datatype": datatype,
                    "bundle_key": _build_bundle_key(subject, session, task, run, acq),
                    "md5": md5_hash,
                    "size": size,
                    "signal_json": str(signal_json.relative_to(root)) if signal_json else None,
                    "eeg_json": str(signal_json.relative_to(root)) if (signal_json and modality in {"eeg", "ecg"}) else None,
                    "meg_json": str(signal_json.relative_to(root)) if (signal_json and modality == "meg") else None,
                    "channels_tsv": str(channels_tsv.relative_to(root)) if channels_tsv else None,
                    "events_tsv": str(events_tsv.relative_to(root)) if events_tsv else None,
                    # fMRI-specific sidecars (not applicable for EEG)
                    "fmri_json": None,
                    "fmri_events_tsv": None,
                }
            )

    # ------------------------------------------------------------------
    # CTF MEG recordings. CTF stores each recording as a directory ending
    # in ``*_meg.ds`` (containing ``.meg4``/``.res4``/``.acq``/...), not a
    # single signal file, so the extension loop above misses them. Treat
    # each ``.ds`` directory as one MEG signal entry.
    # ------------------------------------------------------------------
    for ds_dir in root.rglob("*.ds"):
        if not ds_dir.is_dir():
            continue
        # CTF recordings contain internal sub-directories that also end in
        # ``.ds`` (e.g. ``hz.ds``); skip those so each recording is indexed
        # once, at the outer ``*_meg.ds`` level.
        if any(parent.name.endswith(".ds") for parent in ds_dir.parents):
            continue
        try:
            rel_path = ds_dir.relative_to(root)
        except ValueError:
            continue
        if _should_skip_relpath(rel_path):
            continue
        parts = rel_path.parts
        if "derivatives" in parts:
            continue

        stem = ds_dir.stem  # "..._meg" (.ds stripped)
        stem_parts = stem.split("_")
        subject = session = task = run = acq = None
        for part in stem_parts:
            if part.startswith("sub-"):
                subject = part[4:]
            elif part.startswith("ses-"):
                session = part[4:]
            elif part.startswith("task-"):
                task = part[5:]
            elif part.startswith("run-"):
                run = part[4:]
            elif part.startswith("acq-"):
                acq = part[4:]
        if subject is None:
            subject = _infer_non_bids_subject(parts, stem)
        datatype = _infer_bids_datatype(rel_path) or "meg"
        modality = "meg"

        base_core = stem[:-4] if stem.endswith("_meg") else stem
        signal_json = ds_dir.parent / f"{base_core}_meg.json"
        if not signal_json.exists():
            signal_json = ds_dir.with_suffix(".json")
            if not signal_json.exists():
                signal_json = None
        channels_tsv = ds_dir.parent / f"{base_core}_channels.tsv"
        if not channels_tsv.exists():
            channels_tsv = None
        events_tsv = ds_dir.parent / f"{base_core}_events.tsv"
        if not events_tsv.exists():
            events_tsv = None

        # Size = recursive sum of the .ds directory contents; md5 skipped
        # (multi-GB binary blobs; size is the integrity signal here).
        try:
            size = sum(f.stat().st_size for f in ds_dir.rglob("*") if f.is_file())
        except Exception as e:
            logger.warning(f"Could not size {ds_dir}: {e}")
            size = 0

        records.append(
            {
                "path": str(ds_dir.relative_to(root)),
                "subject": subject,
                "session": session,
                "task": task,
                "run": run,
                "acq": acq,
                "modality": modality,
                "datatype": datatype,
                "bundle_key": _build_bundle_key(subject, session, task, run, acq),
                "md5": None,
                "size": size,
                "signal_json": str(signal_json.relative_to(root)) if signal_json else None,
                "eeg_json": None,
                "meg_json": str(signal_json.relative_to(root)) if signal_json else None,
                "channels_tsv": str(channels_tsv.relative_to(root)) if channels_tsv else None,
                "events_tsv": str(events_tsv.relative_to(root)) if events_tsv else None,
                "fmri_json": None,
                "fmri_events_tsv": None,
            }
        )

    # ------------------------------------------------------------------
    # NWB files (DANDI/local NWB source assets)
    # ------------------------------------------------------------------
    for nwb_file in root.rglob("*.nwb"):
        try:
            rel_path = nwb_file.relative_to(root)
        except ValueError:
            continue

        if _should_skip_relpath(rel_path):
            continue
        parts = rel_path.parts
        if "derivatives" in parts:
            continue

        entities = _parse_bids_entities_from_stem(nwb_file.stem)
        subject = entities["subject"] or _infer_non_bids_subject(parts, nwb_file.stem)

        try:
            size = nwb_file.stat().st_size
            md5_hash = _maybe_compute_md5(nwb_file, size_bytes=size, max_bytes=md5_max_bytes)
        except Exception as e:
            logger.warning(f"Could not stat {nwb_file}: {e}")
            size = 0
            md5_hash = None

        nwb_meta = _probe_nwb_index_metadata(nwb_file)
        records.append(
            {
                "path": str(nwb_file.relative_to(root)),
                "subject": subject,
                "session": entities["session"],
                "task": entities["task"],
                "run": entities["run"],
                "acq": entities["acq"],
                "modality": "nwb",
                    "datatype": "nwb",
                    "bundle_key": _build_bundle_key(
                        subject,
                        entities["session"],
                        entities["task"],
                        entities["run"],
                        entities["acq"],
                    ),
                "md5": md5_hash,
                "size": size,
                "signal_json": None,
                "eeg_json": None,
                "meg_json": None,
                "channels_tsv": None,
                "events_tsv": None,
                "fmri_json": None,
                "fmri_events_tsv": None,
                **nwb_meta,
            }
        )

    # ------------------------------------------------------------------
    # fMRI BOLD files
    # ------------------------------------------------------------------
    bold_patterns, require_func_dir = _resolve_fmri_indexing_options(config, dataset_id)
    for pattern in bold_patterns:
        for bold_file in root.rglob(pattern):
            try:
                rel_path = bold_file.relative_to(root)
            except ValueError:
                continue

            if _should_skip_relpath(rel_path):
                continue
            parts = rel_path.parts
            if "derivatives" in parts:
                continue

            is_func_layout = any(str(p).lower() == "func" for p in parts)
            if require_func_dir and not is_func_layout:
                continue

            # Derive stem without NIfTI suffix (.nii or .nii.gz)
            name = bold_file.name
            if name.endswith(".nii.gz"):
                stem = name[:-7]
            elif name.endswith(".nii"):
                stem = name[:-4]
            else:
                stem = bold_file.stem

            stem_parts = stem.split("_")
            subject: Optional[str] = None
            session: Optional[str] = None
            task: Optional[str] = None
            run: Optional[str] = None
            acq: Optional[str] = None

            for part in stem_parts:
                if part.startswith("sub-"):
                    subject = part[4:]
                elif part.startswith("ses-"):
                    session = part[4:]
                elif part.startswith("task-"):
                    task = part[5:]
                elif part.startswith("run-"):
                    run = part[4:]
                elif part.startswith("acq-"):
                    acq = part[4:]

            # fMRI sidecars: BOLD JSON and events TSV
            bold_json = bold_file.with_name(f"{stem}.json")
            if not bold_json.exists():
                bold_json = None

            fmri_events = _pick_best_fmri_events_tsv(
                parent=bold_file.parent,
                stem=stem,
                task=task,
                run=run,
                acq=acq,
            )

            # Compute file size and md5 (can be large but consistent with EEG handling)
            try:
                size = bold_file.stat().st_size
                md5_hash = _maybe_compute_md5(bold_file, size_bytes=size, max_bytes=md5_max_bytes)
            except Exception as e:
                logger.warning(f"Could not stat {bold_file}: {e}")
                size = 0
                md5_hash = None

            records.append(
                {
                    "path": str(bold_file.relative_to(root)),
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "run": run,
                    "acq": acq,
                    "modality": "fmri",
                    "datatype": "func",
                    "bundle_key": _build_bundle_key(subject, session, task, run, acq),
                    "md5": md5_hash,
                    "size": size,
                    # EEG-specific sidecars (not applicable for fMRI)
                    "signal_json": None,
                    "eeg_json": None,
                    "meg_json": None,
                    "channels_tsv": None,
                    "events_tsv": None,
                    # fMRI sidecars
                    "fmri_json": str(bold_json.relative_to(root)) if bold_json else None,
                    "fmri_events_tsv": str(fmri_events.relative_to(root)) if fmri_events else None,
                }
            )

    # ------------------------------------------------------------------
    # Tabular BIDS modalities (pupillometry / behavioral tables)
    # ------------------------------------------------------------------
    for ext in (".tsv", ".csv"):
        for table_file in root.rglob(f"*{ext}"):
            try:
                rel_path = table_file.relative_to(root)
            except ValueError:
                continue
            if _should_skip_relpath(rel_path):
                continue
            parts = rel_path.parts
            if "derivatives" in parts:
                continue
            datatype = _infer_bids_datatype(rel_path)
            if datatype not in {"pupil", "beh"}:
                continue
            if table_file.stem.endswith("_events") or table_file.stem.endswith("_channels"):
                # Skip BIDS sidecar tables; they are provenance, not primary signal/behavior data.
                continue
            entities = _parse_bids_entities_from_stem(table_file.stem)
            subject = entities["subject"] or _infer_non_bids_subject(parts, table_file.stem)
            if subject is None:
                continue
            if entities["task"] is None and entities["run"] is None and entities["acq"] is None:
                # Avoid indexing generic sidecars like participants.tsv/codebooks.
                continue
            sidecar_json = table_file.with_suffix(".json")
            if not sidecar_json.exists():
                sidecar_json = None
            events_tsv = table_file.parent / f"{table_file.stem}_events.tsv"
            if not events_tsv.exists():
                events_tsv = None
            try:
                size = table_file.stat().st_size
                md5_hash = _maybe_compute_md5(table_file, size_bytes=size, max_bytes=md5_max_bytes)
            except Exception as e:
                logger.warning(f"Could not stat {table_file}: {e}")
                size = 0
                md5_hash = None
            records.append(
                {
                    "path": str(table_file.relative_to(root)),
                    "subject": subject,
                    "session": entities["session"],
                    "task": entities["task"],
                    "run": entities["run"],
                    "acq": entities["acq"],
                    "modality": datatype,
                    "datatype": datatype,
                    "bundle_key": _build_bundle_key(
                        subject,
                        entities["session"],
                        entities["task"],
                        entities["run"],
                        entities["acq"],
                    ),
                    "md5": md5_hash,
                    "size": size,
                    "signal_json": str(sidecar_json.relative_to(root)) if sidecar_json else None,
                    "eeg_json": str(sidecar_json.relative_to(root)) if sidecar_json else None,
                    "meg_json": None,
                    "channels_tsv": None,
                    "events_tsv": str(events_tsv.relative_to(root)) if events_tsv else None,
                    "fmri_json": None,
                    "fmri_events_tsv": None,
                }
            )

    columns = [
        "path",
        "subject",
        "session",
        "task",
        "run",
        "acq",
        "modality",
        "datatype",
        "bundle_key",
        "md5",
        "size",
        "signal_json",
        "eeg_json",
        "meg_json",
        "channels_tsv",
        "events_tsv",
        "fmri_json",
        "fmri_events_tsv",
        "nwb_acquisitions",
        "nwb_processing_modules",
        "nwb_intervals",
        "nwb_devices",
        "nwb_modality_hints",
        "nwb_probe_error",
    ]
    df = pd.DataFrame(records, columns=columns)
    if not df.empty and "modality" in df.columns:
        counts = df["modality"].value_counts().to_dict()
        logger.info(f"Indexed {len(df)} files (by modality: {counts})")
    else:
        logger.info(f"Indexed {len(df)} files")
    return df


