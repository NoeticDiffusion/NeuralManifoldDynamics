import json, numpy as np
from pathlib import Path

qc = json.load(open(r"M:\datasets\processed\openneuro\ds005555\batch_event_locked_qc_psg_c3.json"))
warns = [q for q in qc if not q.get("qc_pass") and q.get("n_fails", 0) >= 0]
print(f"WARN subjects ({len(warns)}):")
print(f"  {'Sub':<10} {'Rate/min':>9}  Reason")
for q in warns:
    rate_ok = q.get("rate_in_range", True)
    excl_ok = q.get("excl_ok", True)
    reason = []
    if not rate_ok:
        reason.append(f"rate={q.get('rate_per_min',0):.2f}/min")
    if not excl_ok:
        reason.append(f"excl={q.get('excl_frac',0):.2f}")
    print(f"  {q['subject']:<10} {q.get('rate_per_min',0):>9.2f}  {', '.join(reason)}")

rates = [q.get("rate_per_min", 0) for q in qc if q.get("n_fails", 0) >= 0]
print(f"\nRate distribution (all {len(rates)} subjects):")
for p in [5, 25, 50, 75, 95]:
    print(f"  p{p}: {np.percentile(rates, p):.2f}/min")
print(f"  mean: {np.mean(rates):.2f}  std: {np.std(rates):.2f}")
n_low = sum(1 for r in rates if r < 0.3)
n_high = sum(1 for r in rates if r > 5.0)
n_ok = sum(1 for r in rates if 0.3 <= r <= 5.0)
print(f"\n  In range [0.3-5.0]: {n_ok}  Too low (<0.3): {n_low}  Too high (>5.0): {n_high}")
