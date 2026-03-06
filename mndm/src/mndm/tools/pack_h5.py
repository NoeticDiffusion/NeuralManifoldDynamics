from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np


def _find_latest_mnps_run(ds_path: Path) -> Optional[Path]:
    runs = [
        p
        for pattern in ("neuralmanifolddynamics_*", "mnps_*")
        for p in ds_path.glob(pattern)
        if p.is_dir()
    ]
    if not runs:
        return None
    # Prefer mtime for "latest" (more robust than name sorting).
    return max(runs, key=lambda p: p.stat().st_mtime)


def _find_h5_files(run_dir: Path) -> List[Path]:
    # Typical layout: <run>/<recording_dir>/*.h5
    return sorted(run_dir.glob("**/*.h5"))


def _sanitize_group_name(name: str) -> str:
    # HDF5 group names can't contain null bytes; keep it simple and filesystem-safe.
    return (
        str(name)
        .replace("\\", "/")
        .replace(":", "__")
        .replace(" ", "_")
        .strip("/")
    )


def _attr_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    return str(value)


def pack_run(
    run_dir: Path,
    out_h5: Path,
    overwrite: bool = False,
) -> Path:
    run_dir = Path(run_dir)
    out_h5 = Path(out_h5)
    if not run_dir.exists():
        raise FileNotFoundError(f"MNPS run directory not found: {run_dir}")

    h5_paths = _find_h5_files(run_dir)
    out_resolved = out_h5.resolve()
    tmp_resolved = out_h5.with_suffix(out_h5.suffix + ".tmp").resolve()
    h5_paths = [
        p for p in h5_paths if p.resolve() not in {out_resolved, tmp_resolved}
    ]
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found under: {run_dir}")

    if out_h5.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists (use --overwrite): {out_h5}")
        out_h5.unlink()

    out_h5.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file then rename (best-effort atomic-ish on local FS).
    tmp_path = out_h5.with_suffix(out_h5.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    index_rows: List[Dict[str, Any]] = []
    used_names: Dict[str, int] = {}

    with h5py.File(tmp_path, "w") as out:
        out.attrs["packed"] = True
        out.attrs["packed_created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        out.attrs["source_run_dir"] = str(run_dir)
        out.attrs["n_files"] = int(len(h5_paths))

        for i, src_path in enumerate(h5_paths, start=1):
            src_relpath = str(src_path.relative_to(run_dir)).replace("\\", "/")
            rel_no_ext = str(Path(src_relpath).with_suffix(""))
            base_name = _sanitize_group_name(rel_no_ext)
            if not base_name or base_name == ".":
                base_name = _sanitize_group_name(src_path.stem)
            rel_hash = hashlib.md5(src_relpath.encode("utf-8")).hexdigest()[:8]
            candidate = f"{base_name}__{rel_hash}"

            # Ensure uniqueness even in pathological hash-collision cases.
            n = used_names.get(candidate, 0)
            used_names[candidate] = n + 1
            group_name = candidate if n == 0 else f"{candidate}__dup{n}"
            print(f"[{i}/{len(h5_paths)}] Packing {src_relpath} -> /{group_name}")

            with h5py.File(src_path, "r") as src:
                g = out.require_group(group_name)
                # Copy root attrs into subgroup attrs (keeps per-recording metadata).
                n_attrs_skipped = 0
                for k, v in src.attrs.items():
                    try:
                        g.attrs[k] = v
                    except Exception:
                        # Best-effort; skip un-serializable attrs
                        n_attrs_skipped += 1
                        continue
                if n_attrs_skipped:
                    g.attrs["n_attrs_skipped"] = int(n_attrs_skipped)

                # Copy datasets/groups recursively into subgroup
                for key in src.keys():
                    try:
                        src.copy(key, g)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed copying key '{key}' from '{src_path}' into group '{group_name}'"
                        ) from exc

                index_rows.append(
                    {
                        "group": group_name,
                        "src_relpath": src_relpath,
                        "dataset_id": _attr_str(src.attrs.get("dataset_id", "")),
                        "subject_id": _attr_str(src.attrs.get("subject_id", "")),
                        "session": _attr_str(src.attrs.get("session", "")),
                        "condition": _attr_str(src.attrs.get("condition", "")),
                        "task": _attr_str(src.attrs.get("task", "")),
                        "run": _attr_str(src.attrs.get("run", "")),
                        "acq": _attr_str(src.attrs.get("acq", "")),
                        "group_label": _attr_str(src.attrs.get("group", "")),
                        "n_attrs_skipped": str(int(n_attrs_skipped)),
                    }
                )

        # Add a simple index group for fast discovery without scanning all subgroups.
        idx_grp = out.require_group("__index__")
        str_dt = h5py.string_dtype(encoding="utf-8")
        cols = [
            "group",
            "src_relpath",
            "dataset_id",
            "subject_id",
            "session",
            "condition",
            "task",
            "run",
            "acq",
            "group_label",
            "n_attrs_skipped",
        ]
        for c in cols:
            arr = [str(r.get(c, "")) for r in index_rows]
            idx_grp.create_dataset(c, data=arr, dtype=str_dt)

    tmp_path.replace(out_h5)
    return out_h5


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Pack a MNPS run directory (many small H5) into a single H5 container")
    p.add_argument("--run-dir", type=Path, required=True, help="MNPS run directory (e.g. .../mnps_ds005555_YYYYMMDD_HHMMSS)")
    p.add_argument("--out", type=Path, default=None, help="Output H5 path (default: <run-dir>/packed.h5)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    args = p.parse_args(argv)

    out = args.out or (Path(args.run_dir) / "packed.h5")
    pack_run(Path(args.run_dir), out, overwrite=bool(args.overwrite))
    print(f"Wrote packed H5: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


