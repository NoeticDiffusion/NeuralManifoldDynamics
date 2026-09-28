"""Synthetic contracts for the gated fNIRS preprocessing slice."""

from __future__ import annotations

import builtins
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm import preprocess_fnirs as fnirs  # noqa: E402


class FakeRaw:
    """Small MNE Raw stand-in with the clock fields used by the contract."""

    def __init__(self, data: np.ndarray, channel_types: list[str], *, names: list[str] | None = None) -> None:
        self._data = np.asarray(data, dtype=float).copy()
        self.ch_names = list(names or [f"S{index // 2 + 1}_D{index // 2 + 1} {760 if index % 2 == 0 else 850}" for index in range(len(channel_types))])
        self._channel_types = list(channel_types)
        self.info = {"sfreq": 8.0, "nchan": len(channel_types)}
        self.info["chs"] = [{"unit": 0, "loc": np.zeros(12, dtype=float)} for _ in channel_types]
        pair_geometry = (
            ((0.0, 0.0, 0.0), (0.008, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.030, 0.0, 0.0)),
            ((0.01171875, 0.0, 0.0), (0.01953125, 0.0, 0.0)),
        )
        for index, channel in enumerate(self.info["chs"]):
            pair_index, wavelength_index = divmod(index, 2)
            source, detector = pair_geometry[pair_index]
            channel["loc"][:3] = (np.asarray(source) + np.asarray(detector)) / 2
            channel["loc"][3:6] = source
            channel["loc"][6:9] = detector
            channel["loc"][9] = (760.0, 850.0)[wavelength_index]
        self.filenames = ()
        self.n_times = self._data.shape[1]
        self.first_samp = 20
        self.first_time = 2.5
        self.times = np.arange(self.n_times, dtype=float) / self.info["sfreq"]

    def get_data(self) -> np.ndarray:
        return self._data.copy()

    def get_channel_types(self) -> list[str]:
        return list(self._channel_types)

    def copy(self) -> "FakeRaw":
        return deepcopy(self)

    def pick(self, picks: list[int] | np.ndarray) -> "FakeRaw":
        indices = np.asarray(picks, dtype=int)
        self._data = self._data[indices].copy()
        self.ch_names = [self.ch_names[int(index)] for index in indices]
        self._channel_types = [self._channel_types[int(index)] for index in indices]
        self.info["chs"] = [self.info["chs"][int(index)] for index in indices]
        self.info["nchan"] = len(self.ch_names)
        return self

    def load_data(self) -> "FakeRaw":
        return self


def _validation(classification: str = "cw_amplitude", **changes: object) -> dict:
    summary = {
        "channel_count": 4,
        "sample_count": 64,
        "sampling_frequency_hz": 8.0,
        "time_gap_sample_indices": [],
        "time_start_s": 2.5,
        "time_end_s": 10.375,
        "median_sample_interval_s": 0.125,
        "measurement_pair_count": 2,
        "short_channel_pair_count": 1,
        "wavelengths_nm": [760.0, 850.0],
        "channel_semantic_keys": [
            {"source": "S1", "detector": "D1", "wavelength_or_chromophore": 760.0},
            {"source": "S1", "detector": "D1", "wavelength_or_chromophore": 850.0},
            {"source": "S2", "detector": "D2", "wavelength_or_chromophore": 760.0},
            {"source": "S2", "detector": "D2", "wavelength_or_chromophore": 850.0},
        ],
        "measurement_pair_keys": [
            {"source": "S1", "detector": "D1"},
            {"source": "S2", "detector": "D2"},
        ],
        "snirf_path": "sub-01_task-test_nirs.snirf",
        "snirf_sha256": hashlib.sha256(b"synthetic source content").hexdigest(),
    }
    summary.update(changes)
    return {"valid": True, "admissible": True, "classification": classification, "summary": summary}


def _fake_api(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, dict]]) -> None:
    def optical_density(raw: FakeRaw) -> FakeRaw:
        calls.append(("optical_density", {}))
        out = raw.copy()
        out._data = -np.log(out._data / out._data.mean(axis=1, keepdims=True))
        out._channel_types = ["fnirs_od"] * len(out.ch_names)
        return out

    def scalp_coupling_index(raw: FakeRaw, *, l_freq: float, h_freq: float) -> np.ndarray:
        calls.append(("scalp_coupling_index", {"l_freq": l_freq, "h_freq": h_freq}))
        return np.array([0.01, 0.2, 0.8, 0.99])

    def tddr(raw: FakeRaw) -> FakeRaw:
        calls.append(("tddr", {}))
        return raw.copy()

    def short_channel_regression(raw: FakeRaw, *, max_dist: float) -> FakeRaw:
        calls.append(("short_channel_regression", {"max_dist": max_dist}))
        return raw.copy()

    def beer_lambert_law(raw: FakeRaw, *, ppf: float) -> FakeRaw:
        calls.append(("beer_lambert_law", {"ppf": ppf}))
        out = raw.copy()
        out._channel_types = ["hbo", "hbr", "hbo", "hbr"]
        out.ch_names = [name.rsplit(" ", 1)[0] + " hbo" if index % 2 == 0 else name.rsplit(" ", 1)[0] + " hbr" for index, name in enumerate(out.ch_names)]
        out.info["nchan"] = len(out.ch_names)
        out.info["chs"] = [{"unit": 6} for _ in out.ch_names]
        return out

    monkeypatch.setattr(
        fnirs,
        "_load_mne_functions",
        lambda: fnirs._MneFns(
            mne_version=fnirs.MNE_VERSION,
            mne_nirs_version=fnirs.MNE_NIRS_VERSION,
            optical_density=optical_density,
            scalp_coupling_index=scalp_coupling_index,
            tddr=tddr,
            short_channel_regression=short_channel_regression,
            beer_lambert_law=beer_lambert_law,
            molar_unit_code=6,
        ),
    )


def _raw(classification: str = "cw_amplitude", *, data: np.ndarray | None = None) -> FakeRaw:
    if data is None:
        time = np.arange(64, dtype=float) / 8.0
        data = np.stack([1.0 + (index + 1) * 0.01 * np.sin(time + index) for index in range(4)])
    types = ["fnirs_cw_amplitude"] * 4 if classification == "cw_amplitude" else ["fnirs_od"] * 4
    return FakeRaw(data, types)


def test_cw_path_applies_frozen_operations_and_records_unqualified_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, dict]] = []
    _fake_api(monkeypatch, calls)
    source = tmp_path / "raw.snirf"
    source.write_bytes(b"synthetic source content")

    raw = _raw()
    raw.filenames = (str(source),)
    result = fnirs.preprocess_fnirs(
        raw,
        validation=_validation(),
        input_classification="cw_amplitude",
        source_path=source,
    )

    assert [name for name, _ in calls] == [
        "optical_density",
        "scalp_coupling_index",
        "tddr",
        "short_channel_regression",
        "short_channel_regression",
        "beer_lambert_law",
    ]
    assert calls[1][1] == {"l_freq": 0.7, "h_freq": 1.5}
    assert calls[3][1] == {"max_dist": 0.010}
    assert calls[5][1] == {"ppf": 6.0}
    assert result.computation_status == "computed"
    assert result.measurement_validity == "not_assessed"
    assert result.claim_status == "no_biological_claim"
    assert result.diagnostics["scalp_coupling_index_cutoff"] is None
    assert result.scalp_coupling_index["S1_D1 760"] == pytest.approx(0.01)
    assert result.provenance["source"]["sha256"] == __import__("hashlib").sha256(b"synthetic source content").hexdigest()
    assert result.provenance["sample_clock"]["sampling_frequency_hz"] == 8.0
    assert result.provenance["sample_clock"]["first_sample_index"] == 20
    assert result.provenance["sample_clock"]["first_time_s"] == 2.5
    assert result.provenance["filter_policy"].startswith("no canonical time-series filter")
    regression = next(
        operation for operation in result.provenance["operations"]
        if operation["name"] == "wavelength_stratified_nearest_short_channel_regression"
    )
    channel_regressors = regression["parameters"]["channel_regressors"]
    assert [(row["target_channel"], row["regressor_channel"]) for row in channel_regressors] == [
        ("S2_D2 760", "S1_D1 760"),
        ("S2_D2 850", "S1_D1 850"),
    ]
    assert result.raw_od.get_channel_types() == ["fnirs_od"] * 4
    assert result.raw_haemoglobin.get_channel_types() == ["hbo", "hbr", "hbo", "hbr"]
    assert result.provenance["haemoglobin_units"] == "M (molar; MNE FIFF_UNIT_MOL)"
    assert all(channel["unit"] == 6 for channel in result.raw_haemoglobin.info["chs"])
    np.testing.assert_array_equal(result.raw_od.times, _raw().times)
    np.testing.assert_array_equal(result.raw_haemoglobin.times, _raw().times)


def test_invalid_nonpositive_intensity_fails_before_any_transform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []
    _fake_api(monkeypatch, calls)
    data = _raw().get_data()
    data[0, 0] = 0.0
    source = tmp_path / "raw.snirf"
    source.write_bytes(b"synthetic source content")
    raw = _raw(data=data)
    raw.filenames = (str(source),)

    with pytest.raises(fnirs.FnirsPreprocessingError, match="strictly positive") as exc:
        fnirs.preprocess_fnirs(
            raw,
            validation=_validation(),
            input_classification="cw_amplitude",
            source_path=source,
        )

    assert exc.value.code == "intensity_nonpositive"
    assert calls == []


def test_raw_channel_identity_order_must_match_validated_snirf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []
    _fake_api(monkeypatch, calls)
    source = tmp_path / "raw.snirf"
    source.write_bytes(b"synthetic source content")
    raw = _raw()
    raw.filenames = (str(source),)
    raw.ch_names[0], raw.ch_names[1] = raw.ch_names[1], raw.ch_names[0]

    with pytest.raises(fnirs.FnirsPreprocessingError) as exc:
        fnirs.preprocess_fnirs(raw, validation=_validation(), input_classification="cw_amplitude", source_path=source)

    assert exc.value.code == "metadata_raw_channel_identity_mismatch"
    assert calls == []


def test_raw_wavelengths_must_match_validated_channel_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []
    _fake_api(monkeypatch, calls)
    source = tmp_path / "raw.snirf"
    source.write_bytes(b"synthetic source content")
    raw = _raw()
    raw.filenames = (str(source),)
    raw.info["chs"][0]["loc"][9] = 850.0

    with pytest.raises(fnirs.FnirsPreprocessingError) as exc:
        fnirs.preprocess_fnirs(raw, validation=_validation(), input_classification="cw_amplitude", source_path=source)

    assert exc.value.code == "metadata_raw_wavelength_mismatch"
    assert calls == []


def test_raw_time_support_must_match_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []
    _fake_api(monkeypatch, calls)
    source = tmp_path / "raw.snirf"
    source.write_bytes(b"synthetic source content")
    raw = _raw()
    raw.filenames = (str(source),)

    with pytest.raises(fnirs.FnirsPreprocessingError) as exc:
        fnirs.preprocess_fnirs(
            raw,
            validation=_validation(time_end_s=11.0),
            input_classification="cw_amplitude",
            source_path=source,
        )

    assert exc.value.code == "metadata_raw_time_mismatch"
    assert calls == []


def test_validation_report_must_match_hashed_source_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []
    _fake_api(monkeypatch, calls)
    source = tmp_path / "raw.snirf"
    source.write_bytes(b"synthetic source content")
    raw = _raw()
    raw.filenames = (str(source),)
    report = _validation(snirf_sha256=hashlib.sha256(b"different source content").hexdigest())

    with pytest.raises(fnirs.FnirsPreprocessingError) as exc:
        fnirs.preprocess_fnirs(
            raw,
            validation=report,
            input_classification="cw_amplitude",
            source_path=source,
        )

    assert exc.value.code == "metadata_source_identity_mismatch"
    assert calls == []


def test_optical_density_input_skips_intensity_conversion_and_accepts_negative_od(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict]] = []
    _fake_api(monkeypatch, calls)
    data = _raw().get_data() - 1.1
    raw_od = _raw("optical_density", data=data)
    source = tmp_path / "raw.snirf"
    source.write_bytes(b"synthetic source content")
    raw_od.filenames = (str(source),)

    result = fnirs.preprocess_fnirs(
        raw_od,
        validation=_validation("optical_density"),
        input_classification="optical_density",
        source_path=source,
    )

    assert [name for name, _ in calls] == [
        "scalp_coupling_index",
        "tddr",
        "short_channel_regression",
        "short_channel_regression",
        "beer_lambert_law",
    ]
    assert result.provenance["operations"][0]["status"] == "not_applicable"
    assert result.raw_od.get_channel_types() == ["fnirs_od"] * 4


def test_short_channel_regression_failure_does_not_fall_back_to_uncorrected_od(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[str] = []

    def optical_density(raw: FakeRaw) -> FakeRaw:
        calls.append("od")
        out = raw.copy()
        out._channel_types = ["fnirs_od"] * 4
        return out

    def sci(raw: FakeRaw, *, l_freq: float, h_freq: float) -> np.ndarray:
        calls.append("sci")
        return np.ones(4)

    def tddr(raw: FakeRaw) -> FakeRaw:
        calls.append("tddr")
        return raw.copy()

    def regression(raw: FakeRaw, *, max_dist: float) -> FakeRaw:
        calls.append("regression")
        raise RuntimeError("no short channel in geometry")

    monkeypatch.setattr(
        fnirs,
        "_load_mne_functions",
        lambda: fnirs._MneFns(
            fnirs.MNE_VERSION,
            fnirs.MNE_NIRS_VERSION,
            optical_density,
            sci,
            tddr,
            regression,
            lambda *a, **k: a[0],
            6,
        ),
    )
    source = tmp_path / "raw.snirf"
    source.write_bytes(b"synthetic source content")
    raw = _raw()
    raw.filenames = (str(source),)

    with pytest.raises(fnirs.FnirsPreprocessingError) as exc:
        fnirs.preprocess_fnirs(raw, validation=_validation(), input_classification="cw_amplitude", source_path=source)

    assert exc.value.code == "short_channel_regression_failed"
    assert calls == ["od", "sci", "tddr", "regression"]


def test_short_channel_regression_must_preserve_subset_channel_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict]] = []
    _fake_api(monkeypatch, calls)
    fns = fnirs._load_mne_functions()

    def reorder(raw: FakeRaw, *, max_dist: float) -> FakeRaw:
        return raw.copy().pick(list(reversed(range(len(raw.ch_names)))))

    monkeypatch.setattr(
        fnirs,
        "_load_mne_functions",
        lambda: fnirs._MneFns(
            fns.mne_version,
            fns.mne_nirs_version,
            fns.optical_density,
            fns.scalp_coupling_index,
            fns.tddr,
            reorder,
            fns.beer_lambert_law,
            fns.molar_unit_code,
        ),
    )
    source = tmp_path / "raw.snirf"
    source.write_bytes(b"synthetic source content")
    raw = _raw()
    raw.filenames = (str(source),)

    with pytest.raises(fnirs.FnirsPreprocessingError) as exc:
        fnirs.preprocess_fnirs(raw, validation=_validation(), input_classification="cw_amplitude", source_path=source)

    assert exc.value.code == "short_channel_regression_channel_identity_mismatch"


def test_exact_short_channel_cutoff_fails_closed_before_regression(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict]] = []
    _fake_api(monkeypatch, calls)
    source = tmp_path / "raw.snirf"
    source.write_bytes(b"synthetic source content")
    raw = _raw()
    raw.filenames = (str(source),)
    for channel in raw.info["chs"][2:]:
        channel["loc"][6] = 0.010
        channel["loc"][0] = 0.005

    with pytest.raises(fnirs.FnirsPreprocessingError) as exc:
        fnirs.preprocess_fnirs(
            raw,
            validation=_validation(),
            input_classification="cw_amplitude",
            source_path=source,
        )

    assert exc.value.code == "short_channel_distance_boundary_unsupported"
    assert not any(name == "short_channel_regression" for name, _ in calls)


@pytest.mark.parametrize("reverse_pair_order", [False, True])
def test_equally_near_short_regressors_fail_closed_independent_of_channel_order(
    reverse_pair_order: bool,
) -> None:
    pair_defs = [
        ("S1", "D1", (-0.01953125, 0.0, 0.0), (-0.01171875, 0.0, 0.0)),
        ("S2", "D2", (0.01171875, 0.0, 0.0), (0.01953125, 0.0, 0.0)),
        ("S3", "D3", (-0.0078125, 0.0, 0.0), (0.0078125, 0.0, 0.0)),
    ]
    if reverse_pair_order:
        pair_defs[:2] = reversed(pair_defs[:2])
    names = [f"{source}_{detector} {wavelength}" for source, detector, *_ in pair_defs for wavelength in (760, 850)]
    raw = FakeRaw(np.arange(1.0, 6 * 32 + 1).reshape(6, 32), ["fnirs_od"] * 6, names=names)
    for pair_index, (_, _, source, detector) in enumerate(pair_defs):
        for wavelength_index, wavelength in enumerate((760.0, 850.0)):
            channel = raw.info["chs"][pair_index * 2 + wavelength_index]
            channel["loc"][:3] = (np.asarray(source) + np.asarray(detector)) / 2
            channel["loc"][3:6] = source
            channel["loc"][6:9] = detector
            channel["loc"][9] = wavelength

    def should_not_run(*args: object, **kwargs: object) -> FakeRaw:
        raise AssertionError("The pinned regression must not choose arbitrarily between tied nearest regressors")

    with pytest.raises(fnirs.FnirsPreprocessingError) as exc:
        fnirs._wavelength_stratified_short_channel_regression(
            raw,
            should_not_run,
            expected_clock={},
            wavelengths_nm=(760.0, 850.0),
        )

    assert exc.value.code == "short_channel_nearest_regressor_ambiguous"


@pytest.mark.parametrize("bad_output", ["units", "pair"])
def test_haemoglobin_output_must_match_units_and_validated_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_output: str
) -> None:
    calls: list[tuple[str, dict]] = []
    _fake_api(monkeypatch, calls)
    source = tmp_path / "raw.snirf"
    source.write_bytes(b"synthetic source content")
    raw = _raw()
    raw.filenames = (str(source),)
    fns = fnirs._load_mne_functions()
    original = fns.beer_lambert_law

    def bad_beer(raw_od: FakeRaw, *, ppf: float) -> FakeRaw:
        output = original(raw_od, ppf=ppf)
        if bad_output == "units":
            output.info["chs"][0]["unit"] = 0
        else:
            output.ch_names[0] = "S99_D99 hbo"
            output.ch_names[1] = "S99_D99 hbr"
        return output

    monkeypatch.setattr(
        fnirs,
        "_load_mne_functions",
        lambda: fnirs._MneFns(
            fns.mne_version,
            fns.mne_nirs_version,
            fns.optical_density,
            fns.scalp_coupling_index,
            fns.tddr,
            fns.short_channel_regression,
            bad_beer,
            fns.molar_unit_code,
        ),
    )

    with pytest.raises(fnirs.FnirsPreprocessingError) as exc:
        fnirs.preprocess_fnirs(
            raw,
            validation=_validation(),
            input_classification="cw_amplitude",
            source_path=source,
        )

    assert exc.value.code == (
        "haemoglobin_units_unsupported"
        if bad_output == "units"
        else "haemoglobin_pair_identity_mismatch"
    )


def test_nonadmissible_or_mismatched_metadata_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []
    _fake_api(monkeypatch, calls)
    report = _validation()
    report["admissible"] = False

    with pytest.raises(fnirs.FnirsPreprocessingError) as exc:
        fnirs.preprocess_fnirs(_raw(), validation=report, input_classification="cw_amplitude")

    assert exc.value.code == "metadata_not_admissible"
    assert calls == []


def test_missing_mne_dependency_is_a_reason_coded_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def no_mne_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"mne", "mne_nirs"}:
            raise ImportError("simulated optional dependency absence")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mne_import)
    with pytest.raises(fnirs.FnirsPreprocessingError) as exc:
        fnirs._load_mne_functions()
    assert exc.value.code == "preprocessing_dependency_unavailable"


def test_real_beer_lambert_recovers_known_hbo_hbr_traces_in_molar_units(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Check the real MNE OD-to-Hb inversion against an independent fixture."""
    try:
        import mne
        from mne.preprocessing.nirs import beer_lambert_law
    except ImportError as exc:  # MNE is optional for the broader project suite.
        pytest.skip(f"optional MNE Beer-Lambert dependency unavailable: {exc}")

    assert mne.__version__ == fnirs.MNE_VERSION

    sfreq = 8.0
    sample_count = 64
    time = np.arange(sample_count, dtype=float) / sfreq
    wavelengths_nm = np.array([760.0, 850.0])
    short_distance_m = 0.009
    long_distance_m = 0.030
    ppf = fnirs.BEER_LAMBERT_PPF

    # Independently specified concentrations (M) and hemoglobin extinction
    # coefficients (M^-1 cm^-1) at the two fixture wavelengths. The forward
    # Beer-Lambert relation uses path length in cm and decadic absorbance.
    known_hbo_m = 2.0e-6 + 0.4e-6 * np.sin(2.0 * np.pi * 0.1 * time)
    known_hbr_m = 1.1e-6 - 0.25e-6 * np.cos(2.0 * np.pi * 0.1 * time)
    extinction_m_inv_cm_inv = np.array(
        [
            [586.00, 1548.52],
            [1058.00, 691.32],
        ],
        dtype=float,
    )  # rows: 760/850 nm; columns: HbO/HbR
    concentrations_m = np.vstack([known_hbo_m, known_hbr_m])
    optical_density_short = (
        np.log(10.0)
        * 100.0
        * short_distance_m
        * ppf
        * extinction_m_inv_cm_inv
        @ concentrations_m
    )
    optical_density_long = (
        np.log(10.0)
        * 100.0
        * long_distance_m
        * ppf
        * extinction_m_inv_cm_inv
        @ concentrations_m
    )
    optical_density = np.vstack([optical_density_short, optical_density_long])

    info = mne.create_info(
        ch_names=["S1_D1 760", "S1_D1 850", "S2_D2 760", "S2_D2 850"],
        sfreq=sfreq,
        ch_types=["fnirs_od"] * 4,
    )
    source_position_m = np.array([0.0, 0.0, 0.0])
    for index, (channel, wavelength_nm) in enumerate(zip(info["chs"], [760.0, 850.0, 760.0, 850.0])):
        distance_m = short_distance_m if index < 2 else long_distance_m
        detector_position_m = np.array([distance_m, 0.0, 0.0])
        channel["loc"][:3] = (source_position_m + detector_position_m) / 2
        channel["loc"][3:6] = source_position_m
        channel["loc"][6:9] = detector_position_m
        channel["loc"][9] = wavelength_nm
    raw_od = mne.io.RawArray(optical_density, info, verbose=False)

    source = tmp_path / "synthetic-known-concentrations.snirf"
    source.write_bytes(b"synthetic Beer-Lambert fixture")
    raw_od._filenames = (str(source),)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    validation = _validation(
        "optical_density",
        channel_count=4,
        sample_count=sample_count,
        sampling_frequency_hz=sfreq,
        time_start_s=0.0,
        time_end_s=float(time[-1]),
        median_sample_interval_s=1.0 / sfreq,
        measurement_pair_count=2,
        short_channel_pair_count=1,
        wavelengths_nm=wavelengths_nm.tolist(),
        channel_semantic_keys=[
            {"source": "S1", "detector": "D1", "wavelength_or_chromophore": 760.0},
            {"source": "S1", "detector": "D1", "wavelength_or_chromophore": 850.0},
            {"source": "S2", "detector": "D2", "wavelength_or_chromophore": 760.0},
            {"source": "S2", "detector": "D2", "wavelength_or_chromophore": 850.0},
        ],
        measurement_pair_keys=[{"source": "S1", "detector": "D1"}, {"source": "S2", "detector": "D2"}],
        snirf_sha256=source_digest,
    )

    monkeypatch.setattr(
        fnirs,
        "_load_mne_functions",
        lambda: fnirs._MneFns(
            mne_version=mne.__version__,
            mne_nirs_version=fnirs.MNE_NIRS_VERSION,
            optical_density=lambda raw: raw.copy(),
            scalp_coupling_index=lambda raw, *, l_freq, h_freq: np.ones(len(raw.ch_names)),
            tddr=lambda raw: raw.copy(),
            short_channel_regression=lambda raw, *, max_dist: raw.copy(),
            beer_lambert_law=beer_lambert_law,
            molar_unit_code=int(mne.io.constants.FIFF.FIFF_UNIT_MOL),
        ),
    )

    result = fnirs.preprocess_fnirs(
        raw_od,
        validation=validation,
        input_classification="optical_density",
        source_path=source,
    )

    assert result.raw_haemoglobin.ch_names == ["S1_D1 hbo", "S1_D1 hbr", "S2_D2 hbo", "S2_D2 hbr"]
    assert result.provenance["haemoglobin_units"] == "M (molar; MNE FIFF_UNIT_MOL)"
    assert all(channel["unit"] == int(mne.io.constants.FIFF.FIFF_UNIT_MOL) for channel in result.raw_haemoglobin.info["chs"])
    np.testing.assert_allclose(result.raw_haemoglobin.get_data()[2], known_hbo_m, rtol=3e-4, atol=1e-12)
    np.testing.assert_allclose(result.raw_haemoglobin.get_data()[3], known_hbr_m, rtol=3e-4, atol=1e-12)
    assert np.corrcoef(result.raw_haemoglobin.get_data()[2], known_hbo_m)[0, 1] > 0.999
    assert np.corrcoef(result.raw_haemoglobin.get_data()[1], known_hbr_m)[0, 1] > 0.999
