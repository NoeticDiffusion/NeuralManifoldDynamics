"""Compare F3 vs C3 baseline-corrected sign agreement across all subjects."""
import pandas as pd
import numpy as np
from pathlib import Path

PROC = Path(r"M:\datasets\processed\openneuro\ds005555")
c3 = pd.read_csv(PROC / "baseline_corrected_all_psg_c3.csv")
f3 = pd.read_csv(PROC / "baseline_corrected_all_psg_f3.csv")

ev_c3 = (c3[c3["bin"] == "event"][["subject", "d_bc", "m_bc", "e_bc"]]
         .rename(columns=lambda c: f"c3_{c}" if c != "subject" else c))
ev_f3 = (f3[f3["bin"] == "event"][["subject", "d_bc", "m_bc", "e_bc"]]
         .rename(columns=lambda c: f"f3_{c}" if c != "subject" else c))

mg = ev_c3.merge(ev_f3, on="subject", how="inner")
print(f"Subjects in both channels: {len(mg)}")
print()
for dim in ["d", "m", "e"]:
    agree = (np.sign(mg[f"c3_{dim}_bc"]) == np.sign(mg[f"f3_{dim}_bc"])).sum()
    pct = 100 * agree / len(mg)
    c3_med = mg[f"c3_{dim}_bc"].median()
    f3_med = mg[f"f3_{dim}_bc"].median()
    print(f"  MNPS-{dim}: C3/F3 sign-agree = {agree}/{len(mg)} ({pct:.0f}%)  "
          f"C3_median={c3_med:+.3f}  F3_median={f3_med:+.3f}")
