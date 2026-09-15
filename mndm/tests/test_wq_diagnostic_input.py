import h5py
import numpy as np
import pytest

from mndm.tools.wq_diagnostic_input import load_gate_e_arrays


def _write_fixture(path, *, ids=(0, 1), targets=(1, 2), dt=(30.0, 30.0), qmeta=None, reach_status="invalid"):
    qmeta = qmeta or {}
    with h5py.File(path, "w") as h:
        base = "transition_residuals/v1/primary"
        series = f"{base}/series"
        h.create_dataset(f"{series}/phi_one_step", data=np.repeat(np.eye(2)[None], len(ids), axis=0))
        h.create_dataset(f"{series}/source_window_id", data=ids)
        h.create_dataset(f"{series}/target_window_id", data=targets)
        h.create_dataset(f"{series}/dt_sec", data=dt)
        for k, v in {"schema_version":"mndm.transition_residuals.v1", "computation_status":"computed", "measurement_validity":"not_assessed", "claim_status":"no_biological_claim"}.items():
            h.create_dataset(f"{base}/{k}", data=np.bytes_(v))
        qbase = "transition_residual_covariance_proxy/v1/primary"
        h.create_dataset(f"{qbase}/covariance", data=np.eye(2))
        values = {
            "schema_version":"mndm.transition_residual_covariance_proxy.v1", "computation_status":"computed",
            "q_time_semantics":"one_step_transition_covariance", "q_units":"state_squared",
            "conversion_model":"not_applicable", "q_scope":"recording", "q_semantics":"transition_residual_covariance_proxy",
            "measurement_validity":"not_assessed", "claim_status":"no_biological_claim", "q_dt_sec":30.0,
            "q_max_dt_deviation_sec":0.0,
        }
        values.update(qmeta)
        for k, v in values.items():
            h.create_dataset(f"{qbase}/{k}", data=np.bytes_(v) if isinstance(v, str) else v)
        rbase = "stochastic_reachability/v1/primary"
        for k, v in {"schema_version":"mndm.stochastic_reachability.v1", "computation_status":reach_status, "failure_reason":"reachability_numerical_overflow", "measurement_validity":"not_applicable", "claim_status":"no_biological_claim"}.items():
            h.create_dataset(f"{rbase}/{k}", data=np.bytes_(v))


def test_loader_requires_admissible_metadata_and_preserves_wq_status(tmp_path):
    p = tmp_path / "valid.h5"
    _write_fixture(p)
    phi, q, source = load_gate_e_arrays(p)
    assert phi.shape == (2, 2, 2)
    assert q.shape == (2, 2)
    assert source["q_metadata"]["q_scope"] == "recording"
    assert source["reachability_metadata"]["failure_reason"] == "reachability_numerical_overflow"
    assert len(source["source_sha256"]) == 64
    assert len(source["phi_sha256"]) == 64


@pytest.mark.parametrize("kwargs,pattern", [
    ({"ids": (0, 2), "targets": (1, 3)}, "adjacent"),
    ({"ids": (0, 2), "targets": (1, 3)}, "adjacent"),
    ({"dt": (30.0, 60.0)}, "fixed expected Q dt"),
    ({"qmeta": {"q_units": "state_squared_per_second_squared"}}, "semantics or units"),
    ({"qmeta": {"conversion_model": "required_before_stochastic_reachability"}}, "scope or conversion"),
    ({"qmeta": {"computation_status": "unavailable"}}, "computed"),
])
def test_loader_refuses_gap_dt_or_wrong_q_contract(tmp_path, kwargs, pattern):
    p = tmp_path / "bad.h5"
    _write_fixture(p, **kwargs)
    with pytest.raises(ValueError, match=pattern):
        load_gate_e_arrays(p)


def test_loader_requires_all_transition_ids_and_dt(tmp_path):
    p = tmp_path / "missing.h5"
    _write_fixture(p)
    with h5py.File(p, "a") as h:
        del h["transition_residuals/v1/primary/series/dt_sec"]
    with pytest.raises(ValueError, match="missing diagnostic input"):
        load_gate_e_arrays(p)
