from pathlib import Path
import pandas as pd

root = Path(r"K:/processed/ds004100_dynamical_families/ds004100")
idx = root / "file_index.csv"
feat = root / "features.parquet"
run = root / "neuralmanifolddynamics_ds004100_20260904_195701"

print("=== index ===")
if idx.exists():
    dfi = pd.read_csv(idx)
    print(f"file_index rows={len(dfi)} cols={list(dfi.columns)[:12]}")
    for c in dfi.columns:
        if c.lower() in {"subject", "participant_id", "file", "path", "filename", "raw_file"}:
            print(f"  unique {c}={dfi[c].nunique()}")

print("=== features ===")
df = pd.read_parquet(feat)
print(f"features rows={len(df)} cols={len(df.columns)}")
for c in ["subject", "participant_id", "raw_file", "file", "source_file", "run", "session"]:
    if c in df.columns:
        print(f"  unique {c}={df[c].nunique()}")
# common ingest columns
for c in df.columns:
    if any(k in c.lower() for k in ("file", "subject", "participant", "run_id", "session")):
        if df[c].nunique() < 400:
            print(f"  {c}: nunique={df[c].nunique()}")

h5s = list(run.rglob("*.h5"))
print(f"\nnewest_run_h5={len(h5s)}")
print(f"all_processed_h5={len(list(root.rglob('*.h5')))}")
