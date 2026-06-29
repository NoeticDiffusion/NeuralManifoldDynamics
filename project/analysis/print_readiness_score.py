"""Print the current MEG readiness score."""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

rs_path = Path(r"E:/Science_Datasets/openneuro/processed/ds003645/meg_eeg_comparison/meg_readiness_score.json")
if rs_path.exists():
    rs = json.load(open(rs_path))
    print(f"Weighted score: {rs.get('weighted_score','?')}")
    print(f"Interpretation: {rs.get('interpretation','?')}")
    for k, v in rs.get("scores", {}).items():
        print(f"  {k}: {v}")
else:
    print("meg_readiness_score.json not found")
