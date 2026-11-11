"""Support/provenance checks for the SL-039 technical spectral pilot."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = ROOT / "project/new_mnps_v.3.2/science_lead/sl039_v33b_boas_spectral_pilot.py"
SPEC = importlib.util.spec_from_file_location("sl039_v33b_spectral_pilot", PILOT_PATH)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def test_nyquist_uses_original_source_rate():
    assert pilot.validate_source_nyquist(256.0) == 128.0
    with pytest.raises(ValueError, match="source_nyquist_does_not_support_required_band"):
        pilot.validate_source_nyquist(90.0)


def test_processed_signal_hash_requires_byte_exact_match():
    signal = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    digest = pilot.array_sha256(signal)
    assert pilot.validate_processed_hash(signal, digest) == digest
    changed = signal.copy()
    changed[0, 0] = 1.0
    with pytest.raises(ValueError, match="processed_sha256_mismatch"):
        pilot.validate_processed_hash(changed, digest)


def test_component_validity_is_independent_and_null_is_preserved():
    no_alpha = {
        "qualified": False,
        "aperiodic_exponent": 1.8,
        "individual_alpha_peak_hz": None,
        "exponent_valid": True,
        "alpha_peak_valid": False,
        "reason": "no_supported_periodic_alpha_peak",
    }
    pilot.validate_component_result(no_alpha)

    invalid_contract = dict(no_alpha, alpha_peak_valid=True)
    with pytest.raises(ValueError, match="component_null_validity_mismatch"):
        pilot.validate_component_result(invalid_contract)
