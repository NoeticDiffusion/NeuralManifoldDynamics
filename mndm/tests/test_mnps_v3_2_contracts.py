"""Tests for the shared MNPS v3.2 measurement result envelope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_2.contracts import (
    MeasurementRecord,
    RealizationRecord,
    ReproducibilityRecord,
    TimeSupportRecord,
)
from mndm.mnps.v3_2.ed import compute_ed_raw_sensor_covariance_participation_ratio
from mndm.mnps.v3_2.nie import compute_nie_raw_weighted_global_efficiency


def _ordered_hash(labels: tuple[str, ...]) -> str:
    canonical = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_ed_uses_common_record_without_inventing_time_or_run_provenance() -> None:
    time = np.arange(16, dtype=np.float64)
    signals = np.vstack((np.sin(time), np.cos(time), np.sin(time / 3.0)))
    result = compute_ed_raw_sensor_covariance_participation_ratio(
        signals, ("F3", "C3", "P3")
    )

    record = result.to_measurement_record()
    payload = result.to_dict()["measurement_record"]

    assert record.family_id == "ED"
    assert record.value == result.value
    assert record.computed and record.valid and not record.qualified
    assert record.support.support_status == "unknown"
    assert record.support.sample_count == signals.shape[1]
    assert record.support.sample_rate_hz is None
    assert record.realization.audit_status == "complete"
    assert record.realization.ordered_labels == ("F3", "C3", "P3")
    assert record.realization.reference_id is None
    assert record.reproducibility.status == "not_supplied"
    assert payload["support"]["source_id"] is None
    assert payload["reproducibility"]["protocol_sha256"] is None


def test_nie_uses_common_record_and_keeps_nodes_distinct_from_sensor_channels() -> None:
    labels = ("A", "B", "C")
    weights = np.array(
        [[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]],
        dtype=np.float64,
    )
    result = compute_nie_raw_weighted_global_efficiency(
        weights,
        labels,
        _ordered_hash(labels),
        "test.edge.v1",
        "test.no_threshold.v1",
    )

    record = result.to_measurement_record()

    assert record.family_id == "NIE"
    assert record.realization.kind == "network_nodes"
    assert record.realization.ordered_labels == labels
    assert record.realization.order_sha256 == _ordered_hash(labels)
    assert record.support.support_status == "unknown"
    assert not record.qualified


def test_common_record_serializes_stable_vectors_and_requires_component_names() -> None:
    record = MeasurementRecord(
        family_id="AR",
        estimator_id="AR.test.v1",
        schema_id="nmdm.test.v1",
        value=(0.25, 0.75),
        units="1",
        computed=True,
        valid=True,
        qualified=False,
        status="computed_exploratory",
        reason=None,
        support=TimeSupportRecord(support_status="unknown"),
        realization=RealizationRecord(
            audit_status="unknown",
            kind=None,
            requested_id=None,
            actual_id=None,
            ordered_labels=None,
            order_sha256=None,
            set_sha256=None,
            count=None,
        ),
        qc=(),
        reproducibility=ReproducibilityRecord(status="not_supplied"),
        value_components=("pair_01", "pair_02"),
    )

    assert record.to_dict()["value"] == [0.25, 0.75]
    assert record.to_dict()["value_components"] == ["pair_01", "pair_02"]

    with pytest.raises(ValueError, match="ordered component name"):
        MeasurementRecord(
            family_id="AR",
            estimator_id="AR.test.v1",
            schema_id="nmdm.test.v1",
            value=(0.25, 0.75),
            units="1",
            computed=True,
            valid=True,
            qualified=False,
            status="computed_exploratory",
            reason=None,
            support=record.support,
            realization=record.realization,
            qc=(),
            reproducibility=record.reproducibility,
        )
