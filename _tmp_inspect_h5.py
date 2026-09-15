from collections import Counter
from pathlib import Path
import h5py
import pandas as pd

root = Path(r"K:/processed/ds004100_dynamical_families/ds004100")
runs = sorted(
    [p for p in root.iterdir() if p.is_dir() and p.name.startswith("neuralmanifolddynamics_")],
    key=lambda p: p.name,
)
print("run_dirs:", [p.name for p in runs])
run = runs[-1]
print(f"newest={run}")

h5s = sorted(run.rglob("*.h5"))
print(f"n_h5={len(h5s)}")

idx = root / "file_index.csv"
feat = root / "features.parquet"
if idx.exists():
    dfi = pd.read_csv(idx)
    print(f"file_index rows={len(dfi)} unique_path={dfi['path'].nunique()} unique_subject={dfi['subject'].nunique()}")
if feat.exists():
    df = pd.read_parquet(feat)
    print(f"features rows={len(df)} unique_file={df['file'].nunique() if 'file' in df.columns else 'NA'}")

REQUIRED = ("mnps_3d", "coords_9d", "jacobian/J_hat", "dynamical_families")
present = Counter()
missing = {k: [] for k in REQUIRED}
regional = []
family_status = {fam: Counter() for fam in ("diffusion", "destination", "resilience")}
family_validity = {fam: Counter() for fam in ("diffusion", "destination", "resilience")}
family_reason = {fam: Counter() for fam in ("destination", "resilience")}


def _scalar(f, key):
    if key not in f:
        return "MISSING"
    obj = f[key]
    if not isinstance(obj, h5py.Dataset):
        return "GROUP"
    val = obj[()]
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


for p in h5s:
    with h5py.File(p, "r") as f:
        for k in REQUIRED:
            if k in f:
                present[k] += 1
            else:
                missing[k].append(p.name)
        if "regional_mnps" in f:
            regional.append(p.name)
        for fam in ("diffusion", "destination", "resilience"):
            family_status[fam][_scalar(f, f"dynamical_families/{fam}/v1/computation_status")] += 1
            family_validity[fam][_scalar(f, f"dynamical_families/{fam}/v1/measurement_validity")] += 1
        family_reason["destination"][_scalar(f, "dynamical_families/destination/v1/failure_reason")] += 1
        family_reason["resilience"][_scalar(f, "dynamical_families/resilience/v1/failure_reason")] += 1

print("\n=== capability counts ===")
for k in REQUIRED:
    print(f"has /{k}: {present[k]}/{len(h5s)}")
    if missing[k]:
        print(f"  missing examples: {missing[k][:8]}")
print(f"has /regional_mnps: {len(regional)}/{len(h5s)}")
if regional:
    print(f"  unexpected: {regional[:8]}")

print("\n=== family computation_status ===")
for fam in ("diffusion", "destination", "resilience"):
    print(f"{fam}: {dict(family_status[fam])}")
    print(f"  measurement_validity: {dict(family_validity[fam])}")
    if fam in family_reason:
        print(f"  failure_reason: {dict(family_reason[fam])}")
