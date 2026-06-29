from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.block_native_config import BlockWindowProfileConfig
from mndm.pipeline.block_native_export import (
    _find_closest_window_idx,
    _inter_network_coupling_by_block,
    block_windows_to_columns,
    blocks_to_columns,
    build_block_native_table,
    inject_block_native_into_payload,
)
from mndm.pipeline.block_windows import BlockWindowSpec, generate_block_windows
from mndm.pipeline.stage_blocking import StageBlockInterval
from mndm.schema import MNPSPayload


def _make_block(
    *,
    block_id: int = 0,
    start_sec: float = 0.0,
    end_sec: float = 20.0,
    stage_code: int = 53,
    block_parameter: float = 20.0,
) -> StageBlockInterval:
    return StageBlockInterval(
        start_sec=start_sec,
        end_sec=end_sec,
        stage_code=stage_code,
        source_event_idx=3,
        block_id=block_id,
        block_parameter=block_parameter,
        support_event_indices=(5, 6),
        derived_from="stage_blocking",
        end_reason="bridge_tail",
        membership_mode="overlap_frac_ge",
        bridge_tail_sec=0.5,
        bridge_tail_cap_sec=1.0,
        is_inferred=True,
    )


def test_blocks_to_columns_includes_frequency_and_provenance():
    block = _make_block()
    cols = blocks_to_columns([block])

    assert set(cols.keys()) >= {
        "block_id",
        "stage_code",
        "start_sec",
        "end_sec",
        "duration_sec",
        "frequency_hz",
        "source_event_idx",
        "support_event_count",
        "derived_from",
        "end_reason",
        "membership_mode",
        "bridge_tail_sec",
        "bridge_tail_cap_sec",
        "is_inferred",
    }
    assert cols["start_sec"][0] == pytest.approx(0.0)
    assert cols["end_sec"][0] == pytest.approx(20.0)
    assert cols["duration_sec"][0] == pytest.approx(20.0)
    assert cols["frequency_hz"][0] == pytest.approx(20.0)
    assert int(cols["source_event_idx"][0]) == 3
    assert int(cols["support_event_count"][0]) == 2


def test_block_windows_to_columns_includes_source_window_index():
    block = _make_block()
    rows = generate_block_windows(
        [block],
        BlockWindowSpec(kind="sliding", window_length_sec=4.0, step_sec=4.0),
    )
    source_idx = [7] * len(rows)
    cols = block_windows_to_columns(rows, source_window_index=source_idx)

    assert "source_window_index" in cols
    assert cols["source_window_index"].shape[0] == len(rows)
    assert int(cols["source_window_index"][0]) == 7


def test_build_block_native_table_adds_source_window_index_and_frequency():
    block = _make_block(start_sec=10.0, end_sec=30.0, block_parameter=15.0)
    payload = SimpleNamespace(
        time=np.linspace(0.0, 40.0, 81, dtype=np.float32),
        x=np.zeros((81, 3), dtype=np.float32),
        x_dot=np.zeros((81, 3), dtype=np.float32),
    )
    rows = build_block_native_table(
        blocks=[block],
        profile=BlockWindowProfileConfig(kind="sliding", window_length_sec=4.0, step_sec=4.0),
        payload=payload,
        subject_id="sub-001",
        run_id="run-01",
        dataset_id="dsX",
    )

    assert len(rows) > 0
    assert "source_window_index" in rows[0]
    assert "frequency_hz" in rows[0]
    assert rows[0]["frequency_hz"] == pytest.approx(15.0)
    assert isinstance(rows[0]["source_window_index"], int)


def test_build_block_native_table_includes_anchor_and_task_labels():
    block = _make_block(start_sec=10.0, end_sec=30.0, block_parameter=15.0)
    n = 81
    payload = SimpleNamespace(
        time=np.linspace(0.0, 40.0, n, dtype=np.float32),
        x=np.zeros((n, 3), dtype=np.float32),
        x_dot=np.zeros((n, 3), dtype=np.float32),
        features_raw_values=np.stack(
            [
                np.linspace(60.0, 62.0, n, dtype=np.float32),
                np.linspace(20.0, 24.0, n, dtype=np.float32),
                np.ones((n,), dtype=np.float32),
            ],
            axis=1,
        ),
        features_raw_names=["ecg_hrv_hr_mean_bpm", "ecg_hrv_rmssd_ms", "qc_ok_ecg_hrv"],
        labels={
            "task_state_label": np.array(["listen"] * n, dtype=object),
            "task_load_n": np.full((n,), 5.0, dtype=np.float32),
        },
        anchor_state={
            "values": np.stack(
                [
                    np.linspace(0.0, 1.0, n, dtype=np.float32),
                    np.linspace(1.0, 2.0, n, dtype=np.float32),
                ],
                axis=1,
            ),
            "names": ["anchor_index", "vagal_index"],
        },
        anchor_state_dot={
            "values": np.ones((n, 2), dtype=np.float32) * 0.1,
            "names": ["anchor_index", "vagal_index"],
        },
        anchor_quality={
            "values": np.ones((n, 2), dtype=np.float32) * 0.95,
            "names": ["anchor_index", "vagal_index"],
        },
    )
    rows = build_block_native_table(
        blocks=[block],
        profile=BlockWindowProfileConfig(kind="sliding", window_length_sec=4.0, step_sec=4.0),
        payload=payload,
        subject_id="sub-001",
        run_id="run-01",
        dataset_id="dsX",
    )

    assert rows
    row = rows[0]
    assert row["task_state_label"] == "listen"
    assert row["task_load_n"] == pytest.approx(5.0)
    assert "ecg_hrv_hr_mean_bpm" in row
    assert "ecg_hrv_rmssd_ms" in row
    assert "qc_ok_ecg_hrv" in row
    assert "anchor_index" in row
    assert "anchor_index_dot" in row
    assert "anchor_index_quality" in row


def test_inject_block_native_into_payload_writes_csv_and_qc(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    events_df = pd.DataFrame(
        [
            {"onset": 0.0, "duration": 8.0, "trial_type": "Eyes Closed: Every 1000 ms"},
        ]
    )
    payload = MNPSPayload(
        time=np.linspace(0.0, 20.0, 21, dtype=np.float64),
        x=np.zeros((21, 3), dtype=np.float32),
        x_dot=np.zeros((21, 3), dtype=np.float32),
        attrs={"fs_out": 1.0, "window_sec": 4.0, "overlap": 0.5},
    )
    config = {
        "block_native": {
            "datasets": {
                "dsX": {
                    "enabled": True,
                    "source": {
                        "kind": "duration_events",
                        "label_column": "trial_type",
                        "onset_column": "onset",
                        "duration_column": "duration",
                        "block_event_labels": ["Eyes Closed: Every 1000 ms"],
                        "block_event_stage_codes": {"Eyes Closed: Every 1000 ms": 10},
                        "min_block_sec": 2.0,
                    },
                    "window_profile": {
                        "kind": "sliding",
                        "window_length_sec": 4.0,
                        "step_sec": 2.0,
                    },
                    "export": {
                        "write_parquet": False,
                        "write_csv": True,
                    },
                }
            }
        }
    }

    manifest = inject_block_native_into_payload(
        payload,
        events_df=events_df,
        config=config,
        dataset_id="dsX",
        subject_id="sub-001",
        run_id="run-01",
        output_dir=tmp_path,
    )

    assert manifest["status"] == "ok"
    assert manifest["n_blocks"] == 1
    assert manifest["n_windows"] >= 1
    assert payload.block_table_columns
    assert payload.block_window_table_columns
    assert "source_window_index" in payload.block_window_table_columns
    assert "frequency_hz" in payload.block_table_columns
    assert manifest["export_paths"]
    assert (tmp_path / "block_native_windows.csv").exists()
    assert isinstance(manifest.get("qc_entry", {}), dict)


def test_find_closest_window_idx_nan_padded_time():
    """Regression: numpy 2.x argmin returns first-NaN index for NaN-padded arrays.

    Simulates the ds003838 digit_span scenario where payload.time is NaN-padded
    beyond the valid recording epochs. Before the fix, np.argmin would return the
    first NaN position (e.g. 1847) for every window regardless of win_center.
    After the fix, np.nanargmin correctly returns the nearest finite index.
    """
    payload = SimpleNamespace(
        time=np.concatenate(
            [np.arange(1, 1848, dtype=np.float64) * 4.0,   # 1847 finite values: 4..7388
             np.full(2614, np.nan, dtype=np.float64)]        # 2614 NaN-padded values
        )
    )

    # Block window centered at 724s should match index 180 (t[180] = 4*181 = 724)
    idx = _find_closest_window_idx(payload, win_start=720.0, win_end=728.0)
    assert idx == 180, f"Expected 180, got {idx}"

    # Block window centered at 7320s should match index 1829 (t[1829] = 4*1830 = 7320)
    idx2 = _find_closest_window_idx(payload, win_start=7316.0, win_end=7324.0)
    assert idx2 == 1829, f"Expected 1829, got {idx2}"

    # A window far outside the finite range should return -1 (no match within win_len)
    idx3 = _find_closest_window_idx(payload, win_start=100000.0, win_end=100008.0)
    assert idx3 == -1, f"Expected -1, got {idx3}"

    # All-NaN time axis should return -1
    payload_all_nan = SimpleNamespace(time=np.full(100, np.nan, dtype=np.float64))
    idx4 = _find_closest_window_idx(payload_all_nan, win_start=0.0, win_end=8.0)
    assert idx4 == -1, f"Expected -1 for all-NaN time, got {idx4}"


def test_inter_network_coupling_basic():
    """Inter-network Jacobian coupling returns 12 finite columns for a well-sampled block."""
    rng = np.random.default_rng(0)
    T = 30
    nets = {
        "central": rng.normal(size=(T, 3)).astype(np.float32),
        "frontal": rng.normal(size=(T, 3)).astype(np.float32),
        "parietal_occipital": rng.normal(size=(T, 3)).astype(np.float32),
        "temporal": rng.normal(size=(T, 3)).astype(np.float32),
    }
    payload = SimpleNamespace(regional_mnps={net: {"mnps": arr} for net, arr in nets.items()})
    rows = [{"block_id": 0, "source_window_index": i} for i in range(T)]
    result = _inter_network_coupling_by_block(rows, payload)

    assert 0 in result, "Block 0 missing from result"
    stats = result[0]
    # 4 networks × 3 off-diagonal directions = 12 coupling columns
    assert len(stats) == 12, f"Expected 12 coupling columns, got {len(stats)}"
    finite_vals = [v for v in stats.values() if np.isfinite(v)]
    assert len(finite_vals) == 12, f"Expected all 12 columns finite, got {len(finite_vals)}"
    assert all(v >= 0 for v in finite_vals), "Frobenius norms must be non-negative"


def test_inter_network_coupling_too_few_samples():
    """Blocks with fewer than k+1 valid windows produce NaN coupling."""
    rng = np.random.default_rng(1)
    T = 30
    nets = {net: rng.normal(size=(T, 3)).astype(np.float32)
            for net in ("central", "frontal", "parietal_occipital", "temporal")}
    payload = SimpleNamespace(regional_mnps={net: {"mnps": arr} for net, arr in nets.items()})
    # Only 10 windows — less than k+1 = 13 for 4 networks
    rows = [{"block_id": 0, "source_window_index": i} for i in range(10)]
    result = _inter_network_coupling_by_block(rows, payload)
    assert 0 in result
    assert all(not np.isfinite(v) for v in result[0].values()), (
        "Should be all NaN when too few samples"
    )


def test_inter_network_coupling_no_regional_mnps():
    """Returns empty dict when payload has no regional_mnps."""
    payload = SimpleNamespace()  # no regional_mnps
    rows = [{"block_id": 0, "source_window_index": i} for i in range(20)]
    result = _inter_network_coupling_by_block(rows, payload)
    assert result == {}, "Should return empty dict when regional_mnps absent"


def test_inter_network_coupling_column_names():
    """Column names follow the coupl_X_from_Y convention."""
    rng = np.random.default_rng(2)
    T = 20
    nets = {
        "central": rng.normal(size=(T, 3)).astype(np.float32),
        "frontal": rng.normal(size=(T, 3)).astype(np.float32),
    }
    payload = SimpleNamespace(regional_mnps={net: {"mnps": arr} for net, arr in nets.items()})
    rows = [{"block_id": 0, "source_window_index": i} for i in range(T)]
    result = _inter_network_coupling_by_block(rows, payload)
    cols = set(result[0].keys())
    assert "coupl_cntr_from_frnt" in cols, "Expected coupl_cntr_from_frnt"
    assert "coupl_frnt_from_cntr" in cols, "Expected coupl_frnt_from_cntr"
    assert len(cols) == 2, f"Expected 2 coupling columns for 2 networks, got {len(cols)}"
