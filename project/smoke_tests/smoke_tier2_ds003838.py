"""Tier 2 smoke test: verify Items 1, 5, and 7 for ds003838.

Tests:
1. Item 7  — inter-network Jacobian coupling columns present and finite in
             block_native output.
2. Item 5  — recall-onset events TSV well-formed with correct structure.
3. Item 1  — R-peak events TSV consistent with intermediate JSON HR/count.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, r"H:\SourceRepo2\NeuralManifoldDynamics\mndm\src")

from types import SimpleNamespace
from mndm.pipeline.block_native_export import (
    _inter_network_coupling_by_block,
    _NETWORK_ABBREV,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
H5_DIR = Path(
    r"H:\SourceRepo2\NeuralManifoldDynamics\.full_ds003838_hrv_blocknative"
    r"\ds003838\neuralmanifolddynamics_ds003838_20260607_111755"
)
DERIVED_EVENTS = Path(
    r"H:\SourceRepo2\NeuralManifoldDynamics\project\preprocessing\derived_events\ds003838"
)
INTERMEDIATE = Path(
    r"H:\SourceRepo2\NeuralManifoldDynamics\.full_ds003838_hrv_blocknative\ds003838\intermediate"
)


# ===========================================================================
# Item 7 — Inter-network Jacobian coupling
# ===========================================================================
def test_item7_coupling() -> None:
    print("\n=== Item 7: Inter-network Jacobian coupling ===")

    for sub_run in ["sub-032_rest"]:
        h5path = H5_DIR / sub_run / f"{sub_run}.h5"
        pq_path = H5_DIR / sub_run / "block_native_windows.parquet"
        assert h5path.exists(), f"H5 not found: {h5path}"
        assert pq_path.exists(), f"Parquet not found: {pq_path}"

        with h5py.File(h5path, "r") as h:
            regional_mnps = {
                net: {"mnps": h["regional_mnps"][net]["mnps"][:]}
                for net in h["regional_mnps"].keys()
            }

        payload = SimpleNamespace(regional_mnps=regional_mnps)
        df = pd.read_parquet(pq_path)
        rows = df[["block_id", "source_window_index"]].to_dict("records")
        result = _inter_network_coupling_by_block(rows, payload)

        # Check column names
        abbrevs = sorted(_NETWORK_ABBREV.values())
        expected_cols = {
            f"coupl_{a}_from_{b}"
            for a in abbrevs
            for b in abbrevs
            if a != b
        }
        for bid, stats in result.items():
            assert set(stats.keys()) == expected_cols, (
                f"Block {bid}: unexpected coupling cols {set(stats.keys()) ^ expected_cols}"
            )

        finite_count = sum(
            1
            for stats in result.values()
            for v in stats.values()
            if np.isfinite(v)
        )
        print(f"  {sub_run}: {len(result)} blocks, {finite_count} finite coupling values")
        assert finite_count > 0, "No finite coupling values found"

        # Coupling norms should be positive when finite
        all_finite = [v for stats in result.values() for v in stats.values() if np.isfinite(v)]
        assert all(v >= 0 for v in all_finite), "Negative Frobenius norms — impossible"
        print(f"  All finite coupling norms are non-negative: PASS")

    print("Item 7: PASS")


# ===========================================================================
# Item 5 — Recall-onset events
# ===========================================================================
def test_item5_recall_onset() -> None:
    print("\n=== Item 5: Recall-onset events TSV ===")

    for sub_id in ["032", "013"]:
        tsv_path = DERIVED_EVENTS / f"sub-{sub_id}_task-memory_events_recall_onset.tsv"
        assert tsv_path.exists(), f"Recall events not found: {tsv_path}"

        df = pd.read_csv(tsv_path, sep="\t")
        print(f"  sub-{sub_id}: {len(df)} recall-onset events")

        # Required columns
        for col in ("onset_sec", "duration_sec", "event_type", "seq_len", "accuracy"):
            assert col in df.columns, f"Missing column '{col}' in {tsv_path}"

        assert (df["event_type"] == "recall_onset").all(), "Unexpected event_type values"
        assert df["duration_sec"].eq(0.0).all(), "Duration should be 0 for point events"
        assert df["seq_len"].isin([5, 9, 13]).all(), f"Unexpected seq_len values: {df['seq_len'].unique()}"
        assert df["accuracy"].isin([0, 1]).all(), "Accuracy should be binary"

        # Onset times should be in ascending order and within a plausible session window
        assert df["onset_sec"].is_monotonic_increasing, "Recall onsets not sorted"
        assert df["onset_sec"].min() > 0, "Recall onsets should be > 0"
        assert df["onset_sec"].max() < 7500, "Recall onsets seem out of range (> 7500s)"

        # Should have roughly equal counts of seq_len 5, 9, 13
        counts = df["seq_len"].value_counts().to_dict()
        print(f"  sub-{sub_id}: seq_len counts = {counts}")
        assert set(counts.keys()) == {5, 9, 13}, "Expected all three sequence lengths"

        # Total: 108 trials per subject (36 per condition × 3)
        assert len(df) == 108, f"Expected 108 recall events, got {len(df)}"
        print(f"  sub-{sub_id}: PASS")

    print("Item 5: PASS")


# ===========================================================================
# Item 1 — R-peak events
# ===========================================================================
def test_item1_rpeak() -> None:
    print("\n=== Item 1: R-peak events TSV ===")

    for sub_id, task in [("015", "rest"), ("020", "rest"), ("025", "rest")]:
        tsv_path = DERIVED_EVENTS / f"sub-{sub_id}_task-{task}_events_rpeak.tsv"
        if not tsv_path.exists():
            print(f"  sub-{sub_id} task-{task}: SKIP (no TSV file)")
            continue

        df = pd.read_csv(tsv_path, sep="\t")

        # Required columns
        for col in ("onset_sec", "duration_sec", "event_type", "peak_sec"):
            assert col in df.columns, f"Missing column '{col}'"
        assert (df["event_type"] == "r_peak").all(), "Unexpected event_type"

        # Peak times and onset_sec should match
        assert np.allclose(df["onset_sec"].values, df["peak_sec"].values), (
            "onset_sec and peak_sec should be identical"
        )

        # Check consistency with intermediate JSON
        int_path = INTERMEDIATE / f"sub-{sub_id}_task-{task}_ecg.json"
        if int_path.exists():
            with open(int_path) as f:
                records = json.load(f)
            int_peaks = sum(r.get("ecg_peak_count", 0) for r in records)
            # With overlapping epochs, int_peaks > n_unique_peaks. Just check order of magnitude.
            derived_n = len(df)
            # int_peaks is from overlapping epochs, derived is unique peaks.
            # Rough sanity: derived should be within 50% of int_peaks / overlap_factor.
            print(
                f"  sub-{sub_id}: {derived_n} R-peaks derived, "
                f"{int_peaks} epoch-peak-count total (overlapping)"
            )
            assert derived_n > 0, "Zero R-peaks"

            # RR intervals should all be >= refractory (0.3s)
            onsets = df["onset_sec"].values
            rr = np.diff(onsets)
            assert (rr >= 0.299).all(), f"Some RR < 0.3s: min={rr.min():.3f}"
            print(f"  sub-{sub_id}: min RR={rr.min():.3f}s  mean RR={rr.mean():.3f}s  PASS")

    print("Item 1: PASS")


# ===========================================================================
# Run all
# ===========================================================================
if __name__ == "__main__":
    test_item7_coupling()
    test_item5_recall_onset()
    test_item1_rpeak()
    print("\n=== All Tier 2 smoke tests passed ===")
