"""Known spectral shapes and independent support failure paths; no EEG outcomes."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from importlib.metadata import version, PackageNotFoundError

PATH = Path(__file__).resolve().parents[2] / "project/new_mnps_v.3.2/science_lead/sl025b_spectral.py"
SPEC = importlib.util.spec_from_file_location("sl025b_spectral", PATH)
spectral = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(spectral)


@pytest.fixture
def spectral_runtime():
    for package, expected in spectral.VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            pytest.skip(f"optional spectral prerequisite: install {package}=={expected}")
        if actual != expected:
            pytest.skip(f"optional spectral prerequisite requires {package}=={expected}; got {actual}")


def test_known_exponent_and_periodic_peak(spectral_runtime):
    f = np.arange(1, 45.25, .25)
    p = 10 ** (1 - 2 * np.log10(f) + .5 * np.exp(-.5 * ((f - 10) / 1) ** 2))
    out = spectral.fit_psd(f, p)
    assert out["exponent_valid"] and out["alpha_peak_valid"]
    assert abs(out["aperiodic_exponent"] - 2) < .01
    assert abs(out["individual_alpha_peak_hz"] - 10) < .05


def test_no_alpha_peak_preserves_exponent(spectral_runtime):
    f = np.arange(1, 45.25, .25)
    for center in (None, 20):
        periodic = 0 if center is None else .5 * np.exp(-.5 * ((f - center) / 1) ** 2)
        out = spectral.fit_psd(f, 10 ** (1 - 2 * np.log10(f) + periodic))
        assert out["exponent_valid"]
        assert abs(out["aperiodic_exponent"] - 2) < .01
        assert not out["alpha_peak_valid"]
        assert out["individual_alpha_peak_hz"] is None


def test_missing_inputs_gaps_and_zero_are_not_measurements(spectral_runtime):
    for samples, kwargs in [(np.zeros(10000), {}), (np.ones(9000), {}),
                            (np.full(10000, np.nan), {}),
                            (np.arange(10000), {"has_gap": True})]:
        out = spectral.estimate_channel(samples, **kwargs)
        assert not out["exponent_valid"] and not out["alpha_peak_valid"]
        assert out["aperiodic_exponent"] is None
    f = np.arange(1, 45.25, .25)
    assert spectral.fit_psd(f, np.zeros_like(f))["reason"] == "nonpositive_or_nonfinite_psd"


def test_welch_producer_preserves_support(spectral_runtime):
    rng = np.random.default_rng(9025)
    time = np.arange(10000) / 500
    out = spectral.estimate_channel(np.sin(2 * np.pi * 10 * time) + .2 * rng.normal(size=10000))
    assert out["exponent_valid"] and out["alpha_peak_valid"]
    assert abs(out["individual_alpha_peak_hz"] - 10) < .3
    assert out["welch_segments"] == 9 and out["n_samples"] == 10000


def test_dependency_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(spectral, "version", lambda package: "unexpected")
    out = spectral.fit_psd(np.arange(1, 45.25, .25), np.ones(177))
    assert out["reason"].startswith("dependency_version_mismatch:")
    assert out["aperiodic_exponent"] is None
