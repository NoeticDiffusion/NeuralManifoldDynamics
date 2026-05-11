"""Quick smoke test: 6 s windows driven by overlay config, no staging-epoch dependence."""
import sys, csv, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "mndm" / "src"))
sys.path.insert(0, str(REPO / "core" / "src"))

from core.config_loader import load_config
from mndm.pipeline.event_locked_config import (
    event_locked_profile_from_config, alignment_config_from_profile,
    matching_config_from_profile, export_config_from_yaml,
)
from mndm.pipeline.event_alignment import align_events_to_windows
from mndm.pipeline.control_matching import build_matched_controls
from mndm.pipeline.event_locked_export import build_event_locked_table
from mndm.pipeline.event_annotations import EventTable
from mndm.schema import MNPSPayload

WIN = 6.0
STEP = 2.0

cfg = load_config(REPO / "mndm/config/config_ingest_ds005555_sleep_spindles.yaml")
profile = event_locked_profile_from_config(cfg, "ds005555")
al_cfg  = alignment_config_from_profile(profile, cfg, "ds005555")
mc_cfg  = matching_config_from_profile(profile)
ex_cfg  = export_config_from_yaml(cfg, "ds005555")

# --- Load staging from real events.tsv ---
EVENTS = Path(r"M:\datasets\received\openneuro\ds005555\sub-1\eeg\sub-1_task-Sleep_acq-psg_events.tsv")
SM_STR = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "R": 4}
SM_INT = {str(i): i for i in range(6)}

def _parse_stage(raw: str) -> int:
    r = raw.strip()
    return SM_STR.get(r, SM_INT.get(r, -1))

onsets, durs, stages = [], [], []
with EVENTS.open(newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        try:
            onsets.append(float(row["onset"]))
            durs.append(float(row.get("duration", 30)))
            raw = row.get("stage_hum", row.get("stage_ai", "")).strip()
            stages.append(_parse_stage(raw))
        except ValueError:
            pass
onsets  = np.array(onsets);  durs = np.array(durs);  stages = np.array(stages, dtype=np.int16)
total_sec = float(onsets[-1] + durs[-1])

# --- Build 6 s MNPS time axis ---
n_win = int((total_sec - WIN) / STEP) + 1
w_start   = np.arange(n_win) * STEP
w_end     = w_start + WIN
t_center  = (w_start + w_end) / 2.0

stage_per_win = np.full(n_win, -1, dtype=np.int16)
for o, d, s in zip(onsets, durs, stages):
    stage_per_win[(t_center >= o) & (t_center < o + d)] = s

rng = np.random.default_rng(0)
x   = rng.random((n_win, 3)).astype(np.float32)
xd  = rng.standard_normal((n_win, 3)).astype(np.float32) * 0.02
c9  = rng.random((n_win, 9)).astype(np.float32)

payload = MNPSPayload(
    time=t_center, x=x, x_dot=xd, stage=stage_per_win,
    window_start=w_start, window_end=w_end,
    coords_9d=c9, coords_9d_names=["m_a","m_e","m_o","d_n","d_l","d_s","e_e","e_s","e_m"],
    attrs={"window_sec": WIN, "overlap": (WIN - STEP) / WIN},
)

# --- Synthetic spindles (every 3rd N2 epoch) ---
n2_epochs = [(o, d) for o, d, s in zip(onsets, durs, stages) if s == 2]
selected  = n2_epochs[::3]
sp_onsets = np.array([o + d / 2 - 0.5 for o, d in selected])
table = EventTable(
    onset_sec=sp_onsets,
    duration_sec=np.ones(len(selected)),
    event_type=np.array(["sleep_spindle"] * len(selected), dtype=object),
    source=np.array(["synthetic:every_3rd_N2"] * len(selected), dtype=object),
    n_events_loaded=len(selected),
)

al  = align_events_to_windows(table, window_start=w_start, window_end=w_end,
                              time=t_center, stage=stage_per_win, config=al_cfg)
ctl = build_matched_controls(table, time=t_center, window_start=w_start,
                             window_end=w_end, stage=stage_per_win, config=mc_cfg)
rows = build_event_locked_table(payload=payload, alignment=al, controls=ctl,
                                event_table=table, subject_id="sub-1",
                                dataset_id="ds005555", config=ex_cfg)

s_rows = [r for r in rows if r["condition"] == "spindle_event"]
c_rows = [r for r in rows if r["condition"] == "matched_control"]
bins = {}
for r in s_rows:
    bins[r["bin_label"]] = bins.get(r["bin_label"], 0) + 1

print("=== ds005555 6s-window overlay smoke test ===")
print(f"  Profile         : {profile.profile_name}")
print(f"  Window/step     : {WIN}s / {STEP}s")
print(f"  MNPS windows    : {n_win}")
print(f"  N2 windows      : {int((stage_per_win==2).sum())}")
print(f"  Spindles        : {table.n} injected")
print(f"  Aligned         : {al.qc['n_events_aligned']} "
      f"({al.qc['n_events_excluded_stage_transition']} excluded at transitions)")
print(f"  Spindle rows    : {len(s_rows)}")
print(f"  Control rows    : {len(c_rows)}")
print(f"  Match rate      : {ctl.qc['match_success_rate']}")
print(f"  Bin counts      : {bins}")
print(f"  MNPS finite frac: {sum(1 for r in s_rows if r.get('mnps_finite')) / max(1, len(s_rows)):.3f}")
print(f"  coords_9d in row: {'m_a' in (s_rows[0] if s_rows else {})}")
if s_rows:
    print(f"  alignment_ref   : {s_rows[0].get('alignment_reference')}")
    print(f"  control_seed    : {s_rows[0].get('control_seed')}")

# Profile provenance check
pd = profile.to_dict()
assert pd["profile_name"] == "sleep_spindle_event_locked_v1"
assert pd["reference"] == "peak"
assert pd["window_length_s"] == 6.0
assert len(pd["bins"]) == 5
assert all(b["label"] in ("pre_far","pre_near","event","post_near","post_far") for b in pd["bins"])
print()
print("  [PASS] Profile provenance assertions OK")
print("  [PASS] All 5 bins defined correctly")
if "pre_near" in bins or "post_near" in bins or "post_far" in bins:
    print("  [PASS] Short-window bins populated (pre_near/post_near/post_far present)")
else:
    print("  [NOTE] Only pre_far+event populated — spindles are centered in 30s epochs")
    print("         (expected: peak reference on synthetic spindles with 2s step window)")
