"""Tests for pack_h5 utility."""

from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def _write_small_h5(path: Path, dataset_id: str, subject_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["dataset_id"] = dataset_id
        f.attrs["subject_id"] = subject_id
        f.create_dataset("mnps_3d", data=np.arange(3, dtype=np.float32))


def test_pack_run_excludes_output_and_uses_reserved_index_group(tmp_path: Path):
    from mndm.tools.pack_h5 import pack_run

    run_dir = tmp_path / "mnps_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_small_h5(run_dir / "a.h5", "ds1", "sub-01")
    _write_small_h5(run_dir / "b.h5", "ds1", "sub-02")

    out = run_dir / "packed.h5"
    first = pack_run(run_dir, out, overwrite=True)
    assert first.exists()
    # Second run should not try to ingest previous packed.h5.
    second = pack_run(run_dir, out, overwrite=True)
    assert second.exists()

    with h5py.File(out, "r") as f:
        assert "__index__" in f
        idx = f["__index__"]
        assert "group" in idx
        groups = [g.decode("utf-8") if isinstance(g, bytes) else str(g) for g in idx["group"][()]]
        assert len(groups) == 2
        assert all(g != "." for g in groups)

