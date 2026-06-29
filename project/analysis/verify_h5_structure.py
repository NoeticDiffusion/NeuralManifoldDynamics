"""Quick structural check: verify row_source and features_projection_z in the latest H5."""
import sys, h5py, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

rdir = sorted(Path(r"E:/Science_Datasets/openneuro/processed/ds003645").glob("neuralmanifolddynamics_*"))[-1]
h5p  = rdir / "sub-002_meeg_FacePerception_run-1" / "sub-002_meeg_FacePerception_run-1.h5"
print(f"Checking: {h5p}")
if not h5p.exists():
    print("H5 not found -- run may not be complete yet")
    sys.exit(1)

with h5py.File(h5p, "r") as f:
    tops = list(f.keys())
    print("Top-level groups:", tops)

    if "row_source" in tops:
        hm = f["row_source/has_meg"][:]
        rs = f["row_source/row_source"][:]
        print(f"[PASS] row_source: set={int((hm==0).sum())} fif={int((hm==1).sum())}")
        print(f"       schema: {f['row_source'].attrs.get('_schema_version','?')}")
    else:
        print("[FAIL] row_source MISSING -- H5 is from an old run")

    if "features_projection_z" in tops:
        pz = f["features_projection_z/values"][:]
        names = [x.decode() if isinstance(x, bytes) else x for x in f["features_projection_z/names"][:]]
        meg_d = next((i for i,n in enumerate(names) if n == "meg_delta"), None)
        if meg_d is not None:
            std = float(np.nanstd(pz[:, meg_d]))
            ok = "PASS" if std > 0.5 else "FAIL"
            print(f"[{ok}] features_projection_z: meg_delta std={std:.4f} (expected >0.5)")
        else:
            print("[INFO] features_projection_z: meg_delta not in names")
        print(f"       export_transform: {f['features_projection_z'].attrs.get('export_transform','?')}")
    else:
        print("[FAIL] features_projection_z MISSING -- H5 is from an old run")
