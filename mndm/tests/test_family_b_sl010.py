"""SL-010 Type-C history-gain replication: CPC-blind 0286, not alpha."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "src"))
sys.path.insert(0, str(TESTS))

from family_b_sl002 import _json_ready  # noqa: E402
from family_b_sl005 import PANELS, PRIMARY_PANEL  # noqa: E402
from family_b_sl002 import OOS_THRESHOLD, RIDGE_ALPHA  # noqa: E402
from family_b_sl006 import M1_N_PARAMS  # noqa: E402
from family_b_sl010 import (  # noqa: E402
    DISCOVERY_SUBJECT,
    FROZEN_SCORER,
    REPLICATION_SUBJECT,
    TYPE_C_SL010_FEATURES_DIR,
    TYPE_C_SL010_OVERLAY,
    adjacent_hops_canonical,
    replication_decision,
    require_frozen_subject,
    score_type_c_history,
    select_replication_subject,
    sl010_report,
)


def _assert_no_none(value, path: str = "root") -> None:
    if value is None:
        if path.endswith("selected_subject") or path.endswith("selection_error"):
            return
        raise AssertionError(f"unexpected None at {path}")
    if isinstance(value, dict):
        for key, inner in value.items():
            _assert_no_none(inner, f"{path}.{key}")
    elif isinstance(value, list):
        for i, inner in enumerate(value):
            _assert_no_none(inner, f"{path}[{i}]")


def test_frozen_subject_rule_and_not_alpha():
    assert DISCOVERY_SUBJECT == "0284"
    assert REPLICATION_SUBJECT == "0286"
    assert PANELS[PRIMARY_PANEL][0] == "eeg_theta"
    source = Path(__file__).resolve().parent.joinpath("family_b_sl010.py").read_text(
        encoding="utf-8"
    )
    assert "from scipy.linalg import logm" not in source
    assert "spectral_abscissa" not in source
    assert "cpc_inspected" in source


def test_select_replication_subject_is_filename_hours_only(tmp_path: Path):
    (tmp_path / "0284").mkdir()
    (tmp_path / "0286").mkdir()
    (tmp_path / "0296").mkdir()
    (tmp_path / "0285").mkdir()
    for hour in range(1, 13):
        (tmp_path / "0284" / f"0284_{hour:03d}_001_EEG.hea").write_text("x", encoding="utf-8")
        (tmp_path / "0286" / f"0286_{hour:03d}_001_EEG.hea").write_text("x", encoding="utf-8")
    for hour in range(1, 7):
        (tmp_path / "0296" / f"0296_{hour:03d}_001_EEG.hea").write_text("x", encoding="utf-8")
    (tmp_path / "0285" / "0285_001_001_EEG.hea").write_text("x", encoding="utf-8")
    assert select_replication_subject(tmp_path) == "0286"


def test_require_frozen_subject_rejects_mixed_files():
    mixed = pd.DataFrame({"file": ["0286_001_020_EEG.hea", "0999_001_001_EEG.hea"]})
    reason = require_frozen_subject(mixed, "0286")
    assert reason is not None
    assert "0999" in reason
    pure = pd.DataFrame({"file": ["0286_001_020_EEG.hea", "0286_002_021_EEG.hea"]})
    assert require_frozen_subject(pure, "0286") is None


def test_adjacent_hops_reject_a_single_gap():
    starts = [0.0, 2.0, 4.0, 8.0, 10.0]
    audit = adjacent_hops_canonical(starts)
    assert audit["canonical"] is False
    assert audit["n_noncanonical_hops"] == 1
    even = [0.0, 2.0, 4.0, 6.0]
    assert adjacent_hops_canonical(even)["canonical"] is True


def test_score_rejects_timestamp_gap_before_triples():
    cols = list(PANELS[PRIMARY_PANEL])
    n = 20
    t = np.arange(n, dtype=float) * 2.0
    t[10:] += 2.0
    data = {c: np.ones(n, dtype=np.float32) for c in cols}
    data["file"] = ["0286_001_020_EEG.hea"] * n
    data["t_start"] = t
    scored = score_type_c_history(
        pd.DataFrame(data),
        rng_seed=1,
        features_dir=TYPE_C_SL010_FEATURES_DIR,
        overlay=TYPE_C_SL010_OVERLAY,
    )
    assert scored["recordings"][0]["status"] == "GRID_INVALID"
    assert scored["recordings"][0]["hop_audit"]["n_noncanonical_hops"] == 1


def test_frozen_scorer_tokens_are_auditable():
    assert FROZEN_SCORER["threshold"] == OOS_THRESHOLD == 0.9
    assert FROZEN_SCORER["ridge_alpha"] == RIDGE_ALPHA == 1.0
    assert FROZEN_SCORER["min_train_m1"] == M1_N_PARAMS == 78
    assert FROZEN_SCORER["hop_rule"] == "all_adjacent_hops"


def test_replication_decision_requires_positive_delta_p_and_controls():
    def cov(p, rel, closure=False):
        return {
            "median_p_r": p,
            "recording_rel_mse_median": rel,
            "closure_pass": closure,
            "p_recordings_p_r_gt_half": 1.0 if p > 0.5 else 0.0,
        }

    miss = replication_decision(
        {
            "m0": cov(0.21, 0.94),
            "m1": cov(0.20, 0.95),
            "duplicate": cov(0.22, 0.93),
            "shuffle": cov(0.19, 0.96),
        }
    )
    assert miss["replicate_pass"] is False
    assert miss["replication_branch"] == "NOT_REPLICATED"

    modest = replication_decision(
        {
            "m0": cov(0.21, 0.94),
            "m1": cov(0.40, 0.88),
            "duplicate": cov(0.22, 0.93),
            "shuffle": cov(0.18, 0.97),
        }
    )
    assert modest["beats_controls"] is True
    assert modest["replicate_pass"] is True
    assert modest["replication_branch"] == "REPLICATED_MODEST"
    assert modest["branch"] == "MODEST_HISTORY"

    strong = replication_decision(
        {
            "m0": cov(0.21, 0.94),
            "m1": cov(0.70, 0.55, closure=True),
            "duplicate": cov(0.25, 0.90),
            "shuffle": cov(0.22, 0.91),
        }
    )
    assert strong["replication_branch"] == "REPLICATED_STRONG"
    assert strong["cpc_inspected"] is False
    assert strong["alpha_computed"] is False


def test_sl010_report_is_cpc_blind_and_finite():
    report = sl010_report(rng_seed=906)
    assert report["family_b_8s"] == "CLOSED_METHOD_LIMITED"
    assert report["not_alpha_rescue"] is True
    assert report["not_operator"] is True
    assert report["cpc_blind"] is True
    assert report["replication_subject"] == "0286"
    ready = _json_ready(report)
    json.dumps(ready, allow_nan=False)
    _assert_no_none(ready)
    assert report["frozen_scorer"]["threshold"] == 0.9
    assert report["frozen_scorer"]["min_train_m1"] == 78
    lock = report["discovery_lock"]
    assert lock["alpha_computed"] is False
    assert lock["cpc_inspected"] is False
    if lock.get("status") == "SCORED":
        assert lock["decision"]["branch"] == "STRONG_HISTORY"
    rep = report["replication"]
    assert rep["alpha_computed"] is False
    assert rep["cpc_inspected"] is False
    if report["selected_subject"] is not None:
        assert report["selection_matches_freeze"] is True
    if rep.get("status") == "SCORED":
        dec = rep["replication"]
        assert dec["replication_branch"] in {
            "REPLICATED_STRONG",
            "REPLICATED_MODEST",
            "NOT_REPLICATED",
        }
        assert dec["cpc_inspected"] is False
        assert "alpha" not in dec["branch"].lower()
