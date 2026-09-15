import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.signal_support_export import build_signal_support_export
from mndm.parallel import _compute_features_from_preprocessed, write_qc_json
from mndm.pipeline.summary import SubjectSummaryRunner
import json


def test_extent_keeps_nominal_bounds_and_unknown_raw_mapping():
    frame = pd.DataFrame({"epoch_id": [4], "file": ["raw.edf"], "t_start": [20940.0], "t_end": [20970.0]})
    result = build_signal_support_export(frame, [{"raw_file": "raw.edf", "temporal_support": {"status": "unknown", "reason": "nonlocal_filter"}}])
    extent = result["per_epoch_input_extent"][0]
    assert extent["nominal_time_start_sec"] == 20940.0
    assert extent["original_sample_start"] is None
    assert extent["support_status"] == "unknown"


def test_reference_population_is_separate_and_unknown_without_hash():
    frame = pd.DataFrame({"epoch_id": [1], "file": ["raw.edf"], "t_start": [0.0], "t_end": [30.0]})
    result = build_signal_support_export(frame, [], feature_baselines={"m": {"standardization_center": 1.0, "standardization_scale": 2.0}})
    assert result["reference_fit_population"]["status"] == "unknown"
    assert result["reference_fit_population"]["features"]["m"]["center"] == 1.0


def test_missing_source_record_does_not_invent_quality_or_filter_support():
    frame = pd.DataFrame({"epoch_id": [1], "file": ["missing.edf"], "t_start": [0.0], "t_end": [1.0]})
    result = build_signal_support_export(frame, [])
    assert result["source_quality_status"] == "not_recorded"
    assert result["source_quality_intervals"] == []
    assert result["temporal_support_status"] == "unknown"


def test_signal_support_export_round_trips_through_h5_writer(require_real_h5py, tmp_path):
    """The H5 group is present with temporal_support_status even with zero QC records.

    Regression guard for ingest_jacobian_fidelity_handover_2.md item 6: an
    analysis-repo H5 reader that fails closed when
    /provenance/signal_support_provenance is missing must always find at
    least {status: unknown, temporal_support_status: unknown}, not an
    absent group.
    """
    import h5py

    from core.io.h5_writer import write_h5
    from mndm.schema import MNPSPayload

    frame = pd.DataFrame({"epoch_id": [1], "file": ["missing.edf"], "t_start": [0.0], "t_end": [1.0]})
    export = build_signal_support_export(frame, [])
    assert export["temporal_support_status"] == "unknown"

    payload = MNPSPayload(
        time=np.array([0.0, 1.0]),
        x=np.zeros((2, 3), dtype=np.float32),
        x_dot=np.zeros((2, 3), dtype=np.float32),
        provenance={"signal_support_provenance": export},
    )
    output = write_h5(tmp_path / "signal_support.h5", "test", payload)
    with h5py.File(output, "r") as handle:
        assert "/provenance/signal_support_provenance" in handle
        grp = handle["/provenance/signal_support_provenance"]
        assert grp.attrs.get("temporal_support_status") == "unknown" or grp["temporal_support_status"][()] == b"unknown"


def test_feature_concat_retains_modality_support_attrs(monkeypatch):
    import mndm.parallel as parallel

    def handler(payload, config):
        frame = pd.DataFrame({"epoch_id": [0], "t_start": [0.0], "t_end": [1.0]})
        frame.attrs["signal_support_provenance"] = {
            "schema": "mndm.signal_support_provenance.v1",
            "temporal_support": {"status": "unknown", "reason": "test"},
        }
        return frame

    monkeypatch.setattr(parallel, "_get_modality_handlers", lambda: {"ppg": handler})
    class Pre:
        signals = {"ppg": [[1.0, 2.0]]}
        sfreq = 1.0
        channels = ["ppg"]
        meta = {"dataset_id": "ds-test"}
    merged, _ = _compute_features_from_preprocessed(Pre(), {})
    assert merged.attrs["signal_support_provenance"]["temporal_support"]["status"] == "unknown"


def test_worker_metadata_qc_roundtrip_keeps_operations_and_schema(tmp_path):
    record = {
        "schema": "mndm.signal_support_provenance.v1",
        "source": {"path": "raw.edf", "exists": True, "sha256": "b" * 64, "sha256_after": "b" * 64, "hash_stable": True},
        "operations": [{"name": "ppg_bandpass", "status": "applied"}],
        "temporal_support": {"status": "unknown", "reason": "not_certified"},
    }
    write_qc_json({"signal_support_provenance": record}, tmp_path, "raw.edf")
    frame = pd.DataFrame({"epoch_id": [0], "file": ["raw.edf"], "t_start": [0.0], "t_end": [1.0]})
    result = build_signal_support_export(frame, [{"raw_file": "raw.edf", **record}])
    assert result["execution_records"][0]["validation_status"] == "valid"
    assert result["execution_records"][0]["record"]["operations"][0]["name"] == "ppg_bandpass"


def test_summary_reads_actual_qc_writer_filename_and_requires_source_hash(tmp_path):
    qc = tmp_path / "qc_artifacts"
    qc.mkdir()
    (qc / "raw_qc_artifacts.json").write_text(json.dumps({
            "signal_support_provenance": {
                "schema": "mndm.signal_support_provenance.v1", "status": "unknown", "source": {"path": "raw.edf", "exists": True, "sha256": "a" * 64, "sha256_after": "a" * 64, "hash_stable": True}
        }
    }), encoding="utf-8")
    runner = SubjectSummaryRunner.__new__(SubjectSummaryRunner)
    runner.ds_path = tmp_path
    result = runner._signal_support_provenance(pd.DataFrame({"file": ["raw.edf"]}))
    assert result["records"][0]["source"]["sha256"] == "a" * 64

    (qc / "bad_qc_artifacts.json").write_text(json.dumps({
            "signal_support_provenance": {"schema": "mndm.signal_support_provenance.v1", "status": "known", "source": {"path": "bad.edf", "exists": True, "sha256": "a" * 64, "sha256_after": "b" * 64, "hash_stable": False}}
    }), encoding="utf-8")
    bad = runner._signal_support_provenance(pd.DataFrame({"file": ["bad.edf"]}))
    assert bad["records"][0]["reason"] == "support_source_hash_unstable"


def test_artifact_qc_applied_true_only_when_sidecar_confirms_it(tmp_path):
    """_artifact_qc_applied must not treat a configured-but-unconfirmed method as applied.

    Regression guard for ingest_jacobian_fidelity_handover_2.md item 9.
    """
    qc = tmp_path / "qc_artifacts"
    qc.mkdir()
    runner = SubjectSummaryRunner.__new__(SubjectSummaryRunner)
    runner.ds_path = tmp_path

    # No sidecar evidence at all -> None (nothing to confirm or deny).
    assert runner._artifact_qc_applied(pd.DataFrame({"file": ["missing.edf"]})) is None

    # Sidecar present, method configured but not applied (e.g. ICA skipped:
    # no EEG channels) -> False, not silently True.
    (qc / "skipped_qc_artifacts.json").write_text(
        json.dumps({"artifact": {"method": "ica", "applied": False}}), encoding="utf-8"
    )
    assert runner._artifact_qc_applied(pd.DataFrame({"file": ["skipped.edf"]})) is False

    # Sidecar present, method not configured at all -> False.
    (qc / "none_qc_artifacts.json").write_text(
        json.dumps({"artifact": {"method": "none", "applied": False}}), encoding="utf-8"
    )
    assert runner._artifact_qc_applied(pd.DataFrame({"file": ["none.edf"]})) is False

    # Sidecar present and confirms the method actually ran -> True.
    (qc / "applied_qc_artifacts.json").write_text(
        json.dumps({"artifact": {"method": "ica", "applied": True}}), encoding="utf-8"
    )
    assert runner._artifact_qc_applied(pd.DataFrame({"file": ["applied.edf"]})) is True

    # Multiple files for one grouping: True only if ALL confirm applied.
    mixed = pd.DataFrame({"file": ["applied.edf", "none.edf"]})
    assert runner._artifact_qc_applied(mixed) is False

    # Sidecar exists but is malformed/unreadable JSON, and is the ONLY
    # evidence for this grouping -> False (evidence exists, just doesn't
    # confirm), not None. Regression for an independent-review finding: the
    # pre-fix code required `found_any` to be set from a successfully
    # parsed "artifact" mapping, so a corrupt sidecar with no other file in
    # the grouping fell through to `found_any=False` -> None, contradicting
    # the documented contract.
    (qc / "corrupt_qc_artifacts.json").write_text("{not valid json", encoding="utf-8")
    assert runner._artifact_qc_applied(pd.DataFrame({"file": ["corrupt.edf"]})) is False

    # Sidecar exists and is valid JSON but missing the "artifact" key
    # entirely -> also False (evidence exists, just doesn't confirm).
    (qc / "noartifact_qc_artifacts.json").write_text(json.dumps({}), encoding="utf-8")
    assert runner._artifact_qc_applied(pd.DataFrame({"file": ["noartifact.edf"]})) is False


def test_build_qc_windows_export_does_not_report_qc_ok_eeg_1_without_confirmed_artifact_method():
    """qc_ok_eeg must become -1 (not_assessed), not a silent pass-through 1.

    Regression guard for ingest_jacobian_fidelity_handover_2.md item 9: the
    feature-level qc_ok_eeg (finite core bands only) must not be exported
    unchanged when no artifact-reduction method is confirmed to have run.
    """
    from mndm.pipeline.summary import _build_qc_windows_export

    sub_frame = pd.DataFrame({"qc_ok_eeg": [1, 1, 0]})
    x = np.zeros((3, 3), dtype=np.float32)
    x_coverage = np.ones((3, 3), dtype=np.float32)

    not_confirmed = _build_qc_windows_export(
        sub_frame=sub_frame, stage=None, x=x, coords_9d=None,
        x_coverage=x_coverage, min_axis_coverage=0.3, artifact_qc_applied=False,
    )
    assert np.all(not_confirmed["qc_ok_eeg"] == -1)

    unknown_evidence = _build_qc_windows_export(
        sub_frame=sub_frame, stage=None, x=x, coords_9d=None,
        x_coverage=x_coverage, min_axis_coverage=0.3, artifact_qc_applied=None,
    )
    assert np.all(unknown_evidence["qc_ok_eeg"] == -1)

    confirmed = _build_qc_windows_export(
        sub_frame=sub_frame, stage=None, x=x, coords_9d=None,
        x_coverage=x_coverage, min_axis_coverage=0.3, artifact_qc_applied=True,
    )
    np.testing.assert_array_equal(confirmed["qc_ok_eeg"], [1, 1, 0])
