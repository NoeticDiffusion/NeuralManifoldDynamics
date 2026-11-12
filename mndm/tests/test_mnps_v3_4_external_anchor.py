"""Focused contract tests for selectable external MNPS 3.4 anchors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mndm.mnps.v3_4 import coordinates
from mndm.mnps.v3_4.candidates import runtime_implementation_hashes
from mndm.mnps.v3_4.external_anchor import (
    PRESET_UNITS,
    PRESETS,
    anchor_run_fingerprint,
    build_measurement_export_with_external_anchor,
    load_selected_anchor,
    resolve_anchor_selection,
    guard_run_anchor_selection,
)
from mndm.cli import _mnps_v3_4_anchor_options_from_args, build_parser


def _anchor_manifest(*, anchor_id="unit-ec", surface_hash="surface-a", shift=0.0):
    transforms = {
        name: [
            {"feature": feature, "transform": transform, "coefficient": coefficient}
            for feature, transform, coefficient in inputs
        ]
        for name, inputs in coordinates.COORDINATE_INPUTS.items()
    }
    overlay = {
        "dataset_id": "ds-unit",
        "dataset_version": "unit-v1",
        "sampling_rate_hz": 1000,
        "realized_sampling_rate_hz": 250,
        "resampling_method": "MNE to 250 Hz",
        "filter_specification": "1-45 Hz",
        "line_frequency_hz": 50,
        "reference": "average",
        "channel_selection": "all 2 EEG channels",
        "bad_channel_policy": "no dropping or interpolation",
        "channel_label_set_compatibility": "casefold_for_anchor_matching; exact_labels_preserved",
        "continuous_segment_boundaries": "source segments retained",
        "edge_support_policy": "10 s edge guard",
        "estimator_parameter_manifest": "frozen unit estimators",
    }
    manifest = {
        "schema": "mndm.mnps_v3_4.anchor_manifest.v1",
        "anchor_id": anchor_id,
        "definition_version": "3.4",
        "status": "experimental_frozen_reference",
        "qualified": False,
        "surface": {
            "mnps_v3_4_source_surface_hash": surface_hash,
            "mnps_v3_4_channel_compatibility_set_hash": "channels-a",
            "mnps_v3_4_reference": "average",
            "mnps_v3_4_reference_set_hash": "ref-a",
        },
        "reference": {
            "scope": "synthetic reference cohort",
            "condition": "EyesClosed",
            "participants": ["sub-001"],
            "participant_count": 1,
            "dataset_overlay": overlay,
            "fit_and_apply_overlap": True,
            "evaluation_status": "transductive_pilot_calibration_not_held_out_validation",
        },
        "feature_units": dict(PRESET_UNITS),
        "feature_order": list(coordinates.FEATURE_ORDER),
        "coordinate_order": list(coordinates.COORDINATE_ORDER),
        "transforms": transforms,
        "coordinate_weights": {name: 1.0 for name in coordinates.COORDINATE_ORDER},
        "feature_anchors": {
            name: {"center": float(shift), "scale": 1.0, "status": "computed"}
            for name in coordinates.FEATURE_ORDER
        },
        "coordinate_anchors": {
            name: {"center": 0.0, "scale": 0.1 if name == "alpha_level" else 1.0, "status": "computed"}
            for name in coordinates.COORDINATE_ORDER
        },
        "anchor_fit_source": {"cohort_manifest_sha256": "cohort-hash", "features_table_sha256": "features-hash"},
    }
    manifest["anchor_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return manifest


def _config(tmp_path: Path, anchor_path: Path | None, *, compatibility="strict", reason=None):
    overlay = _anchor_manifest()["reference"]["dataset_overlay"]
    return {
        "config_dir": str(tmp_path),
        "schema": "mnps_flat_coordinates",
        "definition_version": "3.4",
        "enabled": True,
        "measurement_surface": {
            "modality": "eeg",
            "dataset_overlay": overlay,
            "feature_units": dict(PRESET_UNITS),
        },
        "scaling": {
            "anchor_manifest": None,
            "anchor_manifests_by_condition": {},
            "external_anchor": {
                "preset": None,
                "path": str(anchor_path) if anchor_path is not None else None,
                "compatibility": compatibility,
                "transfer_reason": reason,
            },
        },
    }


def _frame(*, surface_hash="surface-a", task="EyesOpen", values=None, n=1):
    values = values or {
        "eeg_delta": 10.0,
        "eeg_theta": 100.0,
        "eeg_alpha": 100.0,
        "eeg_beta_alpha": 0.5,
        "eeg_gamma": 10.0,
        "eeg_hjorth_mobility": 0.2,
        "eeg_alpha_theta": 0.8,
        "eeg_permutation_entropy": 0.6,
        "mnps_v3_4_alpha_centroid_hz": 10.0,
        "mnps_v3_4_alpha_envelope_cv": 0.25,
    }
    data = {name: [value] * n for name, value in values.items()}
    data.update({
        "file": [f"sub-999_task-{task}_eeg.edf"] * n,
        "subject": ["sub-999"] * n,
        "task": [task] * n,
        "t_start": list(range(n)),
        "t_end": [value + 8 for value in range(n)],
        "mnps_v3_4_source_surface_hash": [surface_hash] * n,
        "mnps_v3_4_channel_compatibility_set_hash": ["channels-a"] * n,
        "mnps_v3_4_reference": ["average"] * n,
        "mnps_v3_4_reference_set_hash": ["ref-a"] * n,
        "mnps_v3_4_channel_labels_json": ['["Fp1","Fp2"]'] * n,
        "mnps_v3_4_channel_order_hash": ["order-a"] * n,
        "mnps_v3_4_implementation_hashes_json": [
            json.dumps(runtime_implementation_hashes(), sort_keys=True, separators=(",", ":"))
        ] * n,
        "eeg_entropy_degraded_mode": [False] * n,
        "qc_ok_eeg": [True] * n,
    })
    return pd.DataFrame(data)


def _write_anchor(path: Path, manifest):
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("preset", "anchor_id", "sha256"),
    [
        ("ds005385_ec_v1", "ds005385_ses1_pretask_EyesClosed_20261111_v1", "79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab"),
        ("ds005385_eo_v1", "ds005385_ses1_pretask_EyesOpen_20261111_v1", "f3bb94310778e91b046fbf77f0540ce860af6a65d4607e1e8593c01f800705b1"),
    ],
)
def test_registered_presets_resolve_exact_bytes_and_ids(preset, anchor_id, sha256):
    loaded = load_selected_anchor({
        "schema": "mnps_flat_coordinates",
        "definition_version": "3.4",
        "enabled": True,
        "scaling": {"external_anchor": {"preset": preset}},
    })
    assert loaded is not None
    manifest, selection = loaded
    assert manifest["anchor_id"] == anchor_id
    assert manifest["_runtime_file_sha256"] == sha256
    assert manifest["schema"] == "mndm.mnps_v3_4.anchor_manifest.v1"
    assert selection["preset"] == preset


def test_selection_rejects_preset_and_custom_path_together(tmp_path):
    with pytest.raises(ValueError, match="preset or custom JSON"):
        resolve_anchor_selection({
            "schema": "mnps_flat_coordinates",
            "definition_version": "3.4",
            "scaling": {"external_anchor": {"preset": "ds005385_ec_v1", "path": "x.json"}},
        })


def test_custom_path_hash_and_missing_file_checks(tmp_path):
    path = _write_anchor(tmp_path / "anchor.json", _anchor_manifest())
    loaded = load_selected_anchor(_config(tmp_path, path))
    assert loaded is not None
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded[0]["_runtime_file_sha256"] == raw_sha
    relative = _config(tmp_path, None)
    relative["scaling"]["external_anchor"]["path"] = "anchor.json"
    assert load_selected_anchor(relative)[0]["anchor_id"] == "unit-ec"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_selected_anchor(_config(tmp_path, tmp_path / "missing.json"))
    wrong_expected = _config(tmp_path, path)
    wrong_expected["scaling"]["external_anchor"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_selected_anchor(wrong_expected)
    bad = json.loads(path.read_text(encoding="utf-8"))
    bad["anchor_id"] = "tampered"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="anchor_hash"):
        load_selected_anchor(_config(tmp_path, path))


@pytest.mark.parametrize("failure", ["missing_file", "corrupt_embedded_hash"])
def test_unavailable_anchor_keeps_raw_measurements_and_is_pinned_as_unsupported(tmp_path, failure):
    if failure == "missing_file":
        path = tmp_path / "missing.json"
    else:
        path = _write_anchor(tmp_path / "corrupt.json", _anchor_manifest())
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["anchor_hash"] = "0" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")
    config = _config(tmp_path, path)
    payload = build_measurement_export_with_external_anchor(_frame(), config)
    assert payload["provenance"]["anchor_application_status"] == "unsupported_anchor_application"
    assert "anchor_manifest_unavailable_or_invalid" in payload["provenance"]["anchor_application_failure_reason"]
    assert np.isfinite(payload["raw_features"]["values"]).any()
    assert not np.isfinite(payload["coordinates"]["values"]).any()

    run_dir = tmp_path / "run-invalid-anchor"
    run_dir.mkdir()
    first = guard_run_anchor_selection(run_dir, config, resume=False)
    assert first["selection_status"] == "unsupported_manifest"
    assert guard_run_anchor_selection(run_dir, config, resume=True) == first


@pytest.mark.parametrize("field", ["sha256", "anchor_id"])
def test_registered_preset_expectations_cannot_override_registry(tmp_path, field):
    config = {
        "schema": "mnps_flat_coordinates",
        "definition_version": "3.4",
        "enabled": True,
        "scaling": {"external_anchor": {"preset": "ds005385_ec_v1", field: "override"}},
    }
    with pytest.raises(ValueError, match="cannot be overridden"):
        load_selected_anchor(config)


def test_frozen_application_uses_declared_transform_and_composite_delta_theta(tmp_path):
    path = _write_anchor(tmp_path / "anchor.json", _anchor_manifest())
    payload = build_measurement_export_with_external_anchor(_frame(), _config(tmp_path, path))
    assert payload is not None
    names = payload["coordinates"]["names"]
    delta_theta = payload["coordinates"]["values"][0, names.index("delta_theta_level")]
    assert delta_theta == pytest.approx(1.5)
    alpha = payload["coordinates"]["values"][0, names.index("alpha_level")]
    assert alpha == pytest.approx(20.0)
    assert payload["provenance"]["anchor_reference_condition"] == "EyesClosed"
    assert payload["provenance"]["target_condition"] == "EyesOpen"
    assert payload["provenance"]["anchor_application_status"] == "computed"


def test_fixed_anchor_does_not_refit_when_unrelated_target_rows_are_added(tmp_path):
    path = _write_anchor(tmp_path / "anchor.json", _anchor_manifest())
    config = _config(tmp_path, path)
    first = _frame()
    unrelated = _frame(values={**_frame().iloc[0].to_dict(), "eeg_alpha": 1e12})
    extra = pd.concat([first, unrelated], ignore_index=True)
    one = build_measurement_export_with_external_anchor(first, config)
    both = build_measurement_export_with_external_anchor(extra, config)
    assert np.allclose(one["coordinates"]["values"][0], both["coordinates"]["values"][0], equal_nan=True)


def test_support_masks_are_preserved_and_values_are_not_clipped(tmp_path):
    path = _write_anchor(tmp_path / "anchor.json", _anchor_manifest())
    values = _frame().iloc[0].drop(labels=["file", "subject", "task", "t_start", "t_end"]).to_dict()
    values.update({"eeg_alpha": 0.0, "eeg_delta": 1e80})
    frame = _frame(values=values)
    payload = build_measurement_export_with_external_anchor(
        frame, _config(tmp_path, path), artifact_qc_applied=True
    )
    names = payload["coordinates"]["names"]
    assert np.isnan(payload["coordinates"]["values"][0, names.index("alpha_level")])
    assert payload["coordinate_validity_mask"][0, names.index("alpha_level")] == 0
    assert payload["coordinates"]["values"][0, names.index("delta_theta_level")] > 4.0
    assert payload["support"]["qc"]["qc_ok_eeg"][0] == 1


def test_strict_rejects_surface_mismatch_but_transfer_records_it_and_stays_unqualified(tmp_path):
    path = _write_anchor(tmp_path / "anchor.json", _anchor_manifest())
    target = _frame(surface_hash="surface-b")
    strict = build_measurement_export_with_external_anchor(target, _config(tmp_path, path))
    assert strict["provenance"]["anchor_application_status"] == "unsupported_anchor_application"
    assert "strict_anchor_compatibility_failed" in strict["provenance"]["anchor_application_failure_reason"]
    assert strict["qualified"] is False
    assert np.isfinite(strict["raw_features"]["values"]).any()
    assert not np.isfinite(strict["coordinates"]["values"]).any()
    payload = build_measurement_export_with_external_anchor(
        target,
        _config(tmp_path, path, compatibility="exploratory_transfer", reason="test transfer to alternate montage"),
    )
    provenance = payload["provenance"]
    assert payload["qualified"] is False
    assert provenance["anchor_application_status"] == "computed_exploratory_transfer"
    assert provenance["measurement_compatibility"]["compatible"] is False
    assert provenance["measurement_compatibility"]["known_mismatches"]
    assert provenance["measurement_compatibility"]["transfer_reason"] == "test transfer to alternate montage"
    assert provenance["target_measurement_identity"]["mnps_v3_4_source_surface_hash"] == "surface-b"


@pytest.mark.parametrize("damage", ["coordinate_mapping", "invalid_scale", "missing_feature"])
def test_transfer_does_not_bypass_structural_or_numerical_failures(tmp_path, damage):
    manifest = _anchor_manifest()
    if damage == "coordinate_mapping":
        manifest["coordinate_order"] = list(reversed(manifest["coordinate_order"]))
        manifest.pop("anchor_hash")
        manifest["anchor_hash"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    elif damage == "invalid_scale":
        manifest["feature_anchors"]["eeg_alpha"]["scale"] = 0.0
        manifest.pop("anchor_hash")
        manifest["anchor_hash"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    path = _write_anchor(tmp_path / "anchor.json", manifest)
    frame = _frame(surface_hash="surface-b")
    if damage == "missing_feature":
        frame = frame.drop(columns=["eeg_gamma"])
        payload = build_measurement_export_with_external_anchor(
            frame,
            _config(tmp_path, path, compatibility="exploratory_transfer", reason="structural failure test"),
        )
        names = payload["coordinates"]["names"]
        assert payload["provenance"]["anchor_application_status"] == "computed_exploratory_transfer"
        assert np.isfinite(payload["coordinates"]["values"][0, names.index("alpha_level")])
        assert np.isnan(payload["coordinates"]["values"][0, names.index("gamma_level")])
    else:
        payload = build_measurement_export_with_external_anchor(
            frame,
            _config(tmp_path, path, compatibility="exploratory_transfer", reason="structural failure test"),
        )
        assert payload["provenance"]["anchor_application_status"] == "unsupported_anchor_application"
        assert "anchor_contract_invalid" in payload["provenance"]["anchor_application_failure_reason"]
        assert np.isfinite(payload["raw_features"]["values"]).any()


@pytest.mark.parametrize("compatibility", ["strict", "exploratory_transfer"])
@pytest.mark.parametrize("hash_state", ["missing", "different", "partial_missing"])
def test_external_anchor_rejects_missing_or_different_feature_implementation_hashes(
    tmp_path, compatibility, hash_state
):
    path = _write_anchor(tmp_path / "anchor.json", _anchor_manifest())
    frame = _frame(surface_hash="surface-a" if compatibility == "strict" else "surface-b", n=2 if hash_state == "partial_missing" else 1)
    if hash_state == "missing":
        frame = frame.drop(columns=["mnps_v3_4_implementation_hashes_json"])
    elif hash_state == "partial_missing":
        frame.loc[1, "mnps_v3_4_implementation_hashes_json"] = None
    else:
        frame["mnps_v3_4_implementation_hashes_json"] = '{"coordinates.py":"stale"}'
    config = _config(
        tmp_path,
        path,
        compatibility=compatibility,
        reason="hash mismatch boundary test" if compatibility == "exploratory_transfer" else None,
    )
    payload = build_measurement_export_with_external_anchor(frame, config)
    assert payload["provenance"]["anchor_application_status"] == "unsupported_anchor_application"
    assert "implementation" in payload["provenance"]["anchor_application_failure_reason"]
    assert np.isfinite(payload["raw_features"]["values"]).any()
    assert not np.isfinite(payload["coordinates"]["values"]).any()


def test_distinct_anchor_ids_produce_distinct_provenance_and_apply_across_target_conditions(tmp_path):
    ec_path = _write_anchor(tmp_path / "ec.json", _anchor_manifest(anchor_id="ref-ec", shift=0.0))
    eo_path = _write_anchor(tmp_path / "eo.json", _anchor_manifest(anchor_id="ref-eo", shift=0.5))
    frame = _frame(task="EyesOpen")
    ec = build_measurement_export_with_external_anchor(frame, _config(tmp_path, ec_path))
    eo = build_measurement_export_with_external_anchor(frame, _config(tmp_path, eo_path))
    assert ec["provenance"]["anchor_id"] == "ref-ec"
    assert eo["provenance"]["anchor_id"] == "ref-eo"
    assert ec["provenance"]["target_condition"] == eo["provenance"]["target_condition"] == "EyesOpen"
    assert ec["provenance"]["anchor_reference_condition"] == eo["provenance"]["anchor_reference_condition"] == "EyesClosed"
    assert not np.allclose(ec["coordinates"]["values"], eo["coordinates"]["values"], equal_nan=True)


def test_participant_overlap_normalizes_numeric_bids_ids_and_reports_reference_evaluation(tmp_path):
    path = _write_anchor(tmp_path / "anchor.json", _anchor_manifest())
    target = _frame()
    target["subject"] = "1"  # Same participant as the reference manifest's ``sub-001``.
    payload = build_measurement_export_with_external_anchor(target, _config(tmp_path, path))
    provenance = payload["provenance"]
    assert provenance["target_reference_participant_overlap_status"] == "overlap_detected"
    assert provenance["target_reference_overlap_participant_count"] == 1
    assert provenance["reference_fit_and_apply_overlap"] is True
    assert provenance["reference_evaluation_status"] == "transductive_pilot_calibration_not_held_out_validation"


def test_participant_overlap_is_unknown_when_dataset_namespaces_differ(tmp_path):
    path = _write_anchor(tmp_path / "anchor.json", _anchor_manifest())
    config = _config(tmp_path, path, compatibility="exploratory_transfer", reason="cross-dataset probe")
    config["measurement_surface"]["dataset_overlay"]["dataset_id"] = "ds-other"
    payload = build_measurement_export_with_external_anchor(_frame(surface_hash="surface-b"), config)
    provenance = payload["provenance"]
    assert provenance["target_reference_participant_overlap_status"] == "unknown"
    assert provenance["participant_overlap_identity_scope"] == "different_dataset_ids"
    assert provenance["target_reference_overlap_participant_count"] == 0


def test_blank_dataset_ids_do_not_define_a_participant_namespace(tmp_path):
    manifest = _anchor_manifest()
    manifest["reference"]["dataset_overlay"]["dataset_id"] = ""
    manifest.pop("anchor_hash")
    manifest["anchor_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    path = _write_anchor(tmp_path / "anchor.json", manifest)
    config = _config(tmp_path, path, compatibility="exploratory_transfer", reason="blank namespace test")
    config["measurement_surface"]["dataset_overlay"]["dataset_id"] = ""
    payload = build_measurement_export_with_external_anchor(_frame(surface_hash="surface-b"), config)
    provenance = payload["provenance"]
    assert provenance["target_reference_participant_overlap_status"] == "unknown"
    assert provenance["participant_overlap_identity_scope"] == "dataset_identity_missing"


def test_default_export_path_is_delegated_unchanged_when_no_external_anchor_selected(monkeypatch, tmp_path):
    sentinel = {"schema": "default-path"}
    calls = {}

    def fake_export(frame, config, *, artifact_qc_applied=None):
        calls["frame"] = frame
        calls["config"] = config
        calls["artifact_qc_applied"] = artifact_qc_applied
        return sentinel

    monkeypatch.setattr(coordinates, "build_measurement_export", fake_export)
    config = {"schema": "mnps_flat_coordinates", "enabled": True, "scaling": {}}
    frame = pd.DataFrame({"x": [1]})
    assert build_measurement_export_with_external_anchor(frame, config, artifact_qc_applied=True) is sentinel
    assert calls["frame"] is frame
    assert calls["config"]["scaling"] == {}
    assert calls["artifact_qc_applied"] is True


def test_cli_exposes_named_presets_custom_paths_and_rejects_ambiguous_sources():
    parser = build_parser()
    preset_args = parser.parse_args([
        "summarize", "--mnps-v3-4-anchor-preset", "ds005385_ec_v1"
    ])
    assert _mnps_v3_4_anchor_options_from_args(preset_args)["preset"] == "ds005385_ec_v1"
    custom_args = parser.parse_args([
        "resummarize", "--mnps-v3-4-anchor-json", "frozen.json",
        "--mnps-v3-4-anchor-compatibility", "exploratory_transfer",
        "--mnps-v3-4-transfer-reason", "explicit test reason",
    ])
    options = _mnps_v3_4_anchor_options_from_args(custom_args)
    assert options["path"].endswith("frozen.json")
    assert options["compatibility"] == "exploratory_transfer"
    with pytest.raises(SystemExit):
        parser.parse_args([
            "summarize", "--mnps-v3-4-anchor-preset", "ds005385_ec_v1",
            "--mnps-v3-4-anchor-json", "frozen.json",
        ])


def test_resume_guard_prevents_anchor_realization_mixing(tmp_path):
    ec = _write_anchor(tmp_path / "ec.json", _anchor_manifest(anchor_id="ref-ec"))
    eo = _write_anchor(tmp_path / "eo.json", _anchor_manifest(anchor_id="ref-eo"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert guard_run_anchor_selection(run_dir, _config(tmp_path, ec), resume=False)["anchor_id"] == "ref-ec"
    with pytest.raises(ValueError, match="Refusing to mix or overwrite"):
        guard_run_anchor_selection(run_dir, _config(tmp_path, eo), resume=True)


def test_resume_fingerprint_covers_target_surface_and_reference_units(tmp_path):
    path = _write_anchor(tmp_path / "anchor.json", _anchor_manifest())
    config = _config(tmp_path, path)
    base = anchor_run_fingerprint(config)
    changed_surface = _config(tmp_path, path)
    changed_surface["measurement_surface"]["dataset_overlay"]["filter_specification"] = "different filter"
    changed_units = _config(tmp_path, path)
    changed_units["scaling"]["external_anchor"]["reference_feature_units"] = {
        **PRESET_UNITS,
        "eeg_delta": "different unit",
    }
    assert anchor_run_fingerprint(changed_surface) != base
    assert anchor_run_fingerprint(changed_units) != base

