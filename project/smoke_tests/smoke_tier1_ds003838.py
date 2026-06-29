"""Smoke test: Tier-1 items (pupil columns + trajectory curvature) on ds003838.

Run from repo root:
  python project/smoke_tests/smoke_tier1_ds003838.py
"""
from __future__ import annotations
import sys
import types
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mndm" / "src"))

from mndm.pipeline.block_native_export import build_block_native_table
from mndm.pipeline.block_native_config import BlockWindowProfileConfig
from mndm.pipeline.stage_blocking import StageBlockInterval


H5_ROOT = Path(
    r"H:\SourceRepo2\NeuralManifoldDynamics"
    r"\.full_ds003838_hrv_blocknative\ds003838"
    r"\neuralmanifolddynamics_ds003838_20260607_111755"
)

PROFILE = BlockWindowProfileConfig(
    kind="sliding",
    window_length_sec=8.0,
    step_sec=4.0,
    min_block_sec=8.0,
    min_windows_per_block=1,
)


def _load_payload(h5_path: Path):
    import h5py

    with h5py.File(h5_path, "r") as h:
        t = np.array(h["time"], dtype=np.float64)
        x = np.array(h["coords_3d_subject_anchored/values"], dtype=np.float32)
        x_dot = np.zeros_like(x)

        fr_values = np.array(h["features_raw/values"], dtype=np.float32)
        raw_names = h["features_raw/names"][()]
        fr_names = [v.decode() if isinstance(v, bytes) else str(v) for v in raw_names]

        anchor_state: dict = {}
        if "anchor_state" in h:
            a_vals = np.array(h["anchor_state/values"], dtype=np.float32)
            a_names_raw = h["anchor_state/names"][()]
            a_names = [v.decode() if isinstance(v, bytes) else str(v) for v in a_names_raw]
            anchor_state = {"values": a_vals, "names": a_names}

        labels: dict = {}
        if "labels" in h:
            for k in h["labels"].keys():
                raw = h[f"labels/{k}"][()]
                if raw.dtype.kind in ("S", "U", "O"):
                    labels[k] = np.array(
                        [v.decode() if isinstance(v, bytes) else str(v) for v in raw]
                    )
                else:
                    labels[k] = raw

        # Build time bounds for the one-block approximation.
        t_finite = t[np.isfinite(t)]
        t_start = float(t_finite[0]) - 4.0 if t_finite.size else 0.0
        t_end = float(t_finite[-1]) + 4.0 if t_finite.size else 100.0

    payload = types.SimpleNamespace(
        time=t,
        x=x,
        x_dot=x_dot,
        features_raw_values=fr_values,
        features_raw_names=fr_names,
        anchor_state=anchor_state,
        anchor_state_dot={},
        anchor_quality={},
        labels=labels,
    )
    return payload, t_start, t_end


def _one_big_block(t_start: float, t_end: float) -> list:
    """Return a single StageBlockInterval covering the whole run."""
    return [
        StageBlockInterval(
            start_sec=t_start,
            end_sec=t_end,
            stage_code=1,
            source_event_idx=0,
            block_id=0,
            block_parameter=float("nan"),
            support_event_indices=(),
            derived_from="smoke_test",
            end_reason="test",
            membership_mode="",
            bridge_tail_sec=0.0,
            bridge_tail_cap_sec=0.0,
            is_inferred=False,
        )
    ]


def run_smoke(sub: str, task: str) -> bool:
    run_label = f"{sub}_{task}"
    h5_path = H5_ROOT / run_label / f"{run_label}.h5"
    if not h5_path.exists():
        print(f"  SKIP  {run_label}: H5 not found")
        return True

    payload, t_start, t_end = _load_payload(h5_path)
    blocks = _one_big_block(t_start, t_end)

    rows = build_block_native_table(
        blocks=blocks,
        profile=PROFILE,
        payload=payload,
        subject_id=sub,
        run_id=task,
        dataset_id="ds003838",
    )

    ok = True
    print(f"\n{'='*60}")
    print(f"  {run_label}: {len(rows)} windows")

    if not rows:
        print("  ERROR: no windows generated")
        return False

    r0 = rows[0]

    # -------------------------------------------------------
    # Check 1: pupil columns present and non-trivial
    # -------------------------------------------------------
    pupil_cols = [k for k in r0 if k.startswith("pupil_")]
    print(f"  pupil cols: {pupil_cols}")
    if not pupil_cols:
        print("  FAIL: no pupil_ columns in output")
        ok = False
    else:
        for col in ("pupil_diameter_mean", "pupil_dilation_velocity"):
            if col not in r0:
                print(f"  FAIL: expected column {col!r} missing")
                ok = False
                continue
            vals = np.array([float(r[col]) for r in rows])
            n_finite = int(np.sum(np.isfinite(vals)))
            val_range = (
                f"{np.nanmin(vals):.4f} .. {np.nanmax(vals):.4f}"
                if n_finite > 0 else "all NaN"
            )
            print(f"  {col}: n_finite={n_finite}/{len(rows)}, {val_range}")
            if n_finite == 0:
                print(f"  WARN: {col} is all NaN (pupil data may be missing for this subject/run)")

    # -------------------------------------------------------
    # Check 2: trajectory columns present and meaningful
    # -------------------------------------------------------
    traj_cols = [k for k in r0 if k.startswith("traj_")]
    print(f"  traj cols: {traj_cols}")
    expected_traj = [
        "traj_path_length", "traj_endpoint_dist", "traj_efficiency",
        "traj_mean_curvature", "traj_max_curvature", "traj_n_finite_windows",
    ]
    for col in expected_traj:
        if col not in r0:
            print(f"  FAIL: expected column {col!r} missing")
            ok = False

    if "traj_path_length" in r0:
        pl = float(rows[0]["traj_path_length"])
        ed = float(rows[0]["traj_endpoint_dist"])
        eff = float(rows[0]["traj_efficiency"])
        mc = float(rows[0]["traj_mean_curvature"])
        xc = float(rows[0]["traj_max_curvature"])
        nfw = int(rows[0]["traj_n_finite_windows"])
        print(f"  traj_n_finite_windows : {nfw}")
        print(f"  traj_path_length      : {pl:.4f}")
        print(f"  traj_endpoint_dist    : {ed:.4f}")
        print(f"  traj_efficiency       : {eff:.4f}  (0=circular, 1=straight)")
        print(f"  traj_mean_curvature   : {mc:.4f} rad")
        print(f"  traj_max_curvature    : {xc:.4f} rad")

        # Validate: all windows in the block should share the same traj values.
        n_unique_pl = len({round(float(r["traj_path_length"]), 6) for r in rows if np.isfinite(float(r["traj_path_length"]))})
        if n_unique_pl > 1:
            print(f"  FAIL: traj_path_length has {n_unique_pl} distinct values (should be 1 per block)")
            ok = False
        else:
            print("  OK: traj values are block-constant")

        # Sanity: efficiency in [0,1], curvature >= 0
        if np.isfinite(eff) and not (0.0 <= eff <= 1.0 + 1e-6):
            print(f"  FAIL: traj_efficiency={eff:.4f} outside [0,1]")
            ok = False
        if np.isfinite(mc) and mc < 0:
            print(f"  FAIL: traj_mean_curvature={mc:.4f} < 0")
            ok = False

    # -------------------------------------------------------
    # Check 3: SWI fix — source_window_index should be varied (>1 unique)
    # -------------------------------------------------------
    swi = np.array([int(r["source_window_index"]) for r in rows])
    n_unique_swi = len(set(swi.tolist()))
    print(f"  source_window_index: {n_unique_swi} unique / {len(rows)} windows")
    if n_unique_swi == 1 and len(rows) > 1:
        print(f"  FAIL (SWI bug): all windows map to single index {swi[0]}")
        ok = False
    else:
        print("  OK: SWI is varied")

    # -------------------------------------------------------
    # Check 4: m/d/e vary (basic sanity after SWI fix)
    # -------------------------------------------------------
    m_vals = np.array([float(r["m"]) for r in rows])
    n_fin_m = int(np.sum(np.isfinite(m_vals)))
    if n_fin_m > 1:
        m_std = float(np.nanstd(m_vals))
        print(f"  m: n_finite={n_fin_m}, std={m_std:.4f}")
        if m_std < 1e-6:
            print("  FAIL: m is constant (SWI fix may not have taken effect)")
            ok = False
        else:
            print("  OK: m varies")

    print(f"  {'PASS' if ok else 'FAIL'}  {run_label}")
    return ok


def main():
    all_ok = True
    for sub, task in [
        ("sub-032", "rest"),
        ("sub-032", "digit_span"),
        ("sub-094", "digit_span"),   # the one that was already correct before fix
    ]:
        all_ok &= run_smoke(sub, task)

    print("\n" + "="*60)
    print(f"  OVERALL: {'ALL PASS' if all_ok else 'SOME FAILURES'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
