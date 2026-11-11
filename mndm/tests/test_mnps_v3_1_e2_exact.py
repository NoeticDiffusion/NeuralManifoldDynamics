"""Contract tests for the selected, still-unqualified E2 exact-N estimator."""

from __future__ import annotations

import numpy as np

import mndm.mnps.v3_1.eeg_metrics as metrics
from mndm.mnps.v3_1 import MNPS31WindowSupport, compute_e2_sensor_median_lz78_exact_multiset_ratio


FIXED_SET_ID = "anphy_epctl01_e1_g2_frozen_ordered_83_labels"
SOURCE_REFERENCE = "average_over_complete_fixed_83_channel_set"
LABELS = tuple(f"SYN_CH_{index:03d}" for index in range(1, 84))


def _support(**overrides: object) -> MNPS31WindowSupport:
    fields: dict[str, object] = {
        "window_start_sec": 20.0,
        "window_end_sec": 40.0,
        "sample_rate_hz": 250.0,
        "sample_count": 5000,
        "segment_id": "synthetic_e2_s2_segment",
        "source_is_continuous": True,
        "timestamps_uniform": True,
        "gap_free": True,
        "bad_interval_free": True,
        "within_recording_bounds": True,
        "effective_filter_support_valid": True,
    }
    fields.update(overrides)
    return MNPS31WindowSupport(**fields)  # type: ignore[arg-type]


def _signals() -> np.ndarray:
    base = np.tile(np.asarray([-1.0, 1.0]), 2500)
    return np.stack([np.roll(base, index % 2) for index in range(83)])


def _measure(signals: np.ndarray, *, support: MNPS31WindowSupport | None = None):
    return compute_e2_sensor_median_lz78_exact_multiset_ratio(
        signals,
        LABELS,
        support or _support(),
        source_identity="synthetic:e2_g1_s2_exact_5000_test",
        fixed_set_id=FIXED_SET_ID,
        source_reference=SOURCE_REFERENCE,
    )


def test_e2_estimator_is_exact_83_channel_median_and_replays_reference_draws(monkeypatch) -> None:
    # Keep the test focused on thresholding, seed/replay, histogram and
    # aggregation contracts; LZ78 parsing has separate established tests.
    monkeypatch.setattr(metrics, "_lz78_phrase_count", lambda symbols: len(set(symbols)))
    first = _measure(_signals())
    replay = _measure(_signals())

    assert first.computed and first.valid and not first.qualified
    assert first.value == 1.0
    assert first.details["qualified"] is False
    assert first.details["coordinate_assignment"] is None
    assert first.details["aggregate_formula"] == "median_of_all_83_supported_per_channel_ratios"
    first_channels = first.details["per_channel"]
    replay_channels = replay.details["per_channel"]
    assert len(first_channels) == 83
    assert first_channels == replay_channels
    assert len({channel["derived_seed"] for channel in first_channels}) == 83
    assert all(channel["tie_count"] == 0 for channel in first_channels)
    assert all(channel["symbol_counts"] == {"0": 2500, "1": 2500} for channel in first_channels)
    assert all(len(channel["reference_phrase_counts"]) == 32 for channel in first_channels)
    assert all(len(channel["reference_draw_sha256"]) == 32 for channel in first_channels)


def test_e2_any_constant_channel_rejects_the_whole_fixed_set(monkeypatch) -> None:
    monkeypatch.setattr(metrics, "_lz78_phrase_count", lambda symbols: len(set(symbols)))
    signals = _signals()
    signals[17] = 0.0

    result = _measure(signals)

    assert not result.computed and not result.valid and not result.qualified
    assert result.value is None
    assert result.reason == "symbol_sequence_has_fewer_than_two_symbols"
    assert result.details["failed_channel"] == LABELS[17]
    assert result.details["aggregation_failure_policy"] == "any_channel_unsupported_invalidates_whole_fixed_set"


def test_e2_fail_closed_support_and_input_contracts(monkeypatch) -> None:
    monkeypatch.setattr(metrics, "_lz78_phrase_count", lambda symbols: len(set(symbols)))
    signals = _signals()

    gap = _measure(signals, support=_support(gap_free=False))
    wrong_rate = _measure(signals, support=_support(sample_rate_hz=256.0, window_end_sec=39.53125))
    wrong_shape = _measure(signals[:82])
    nonfinite = _signals()
    nonfinite[0, 0] = np.nan
    invalid = _measure(nonfinite)

    assert gap.reason == "gap_or_segment_boundary" and gap.value is None
    assert wrong_rate.reason == "sampling_rate_mismatch" and wrong_rate.value is None
    assert wrong_shape.reason == "invalid_fixed_sensor_matrix_shape" and wrong_shape.value is None
    assert invalid.reason == "non_finite_input" and invalid.value is None


def test_lz78_phrase_parser_uses_incremental_dictionary_phrases() -> None:
    assert metrics._lz78_phrase_count((0, 1, 0, 1)) == 3
    assert metrics._lz78_phrase_count((0, 0, 0, 0)) == 2


def test_lz78_trie_parser_matches_tuple_dictionary_on_short_sequences() -> None:
    from itertools import product

    def reference_phrase_count(symbols: tuple[object, ...]) -> int:
        phrases: set[tuple[object, ...]] = set()
        cursor = 0
        while cursor < len(symbols):
            end = cursor + 1
            while end <= len(symbols) and symbols[cursor:end] in phrases:
                end += 1
            stop = min(end, len(symbols))
            phrase = symbols[cursor:stop]
            if not phrase:
                phrase = symbols[cursor : cursor + 1]
                stop = cursor + 1
            phrases.add(phrase)
            cursor = stop
        return len(phrases)

    sequences = [
        tuple(sequence)
        for length in range(13)
        for sequence in product((0, 1), repeat=length)
    ]
    sequences.extend(
        tuple(sequence)
        for length in range(1, 9)
        for sequence in product(("a", "b", "c"), repeat=length)
    )
    for sequence in sequences:
        assert metrics._lz78_phrase_count(sequence) == reference_phrase_count(sequence)


def test_binary_lz78_fast_path_matches_generic_parser_exactly() -> None:
    from itertools import product

    sequences = [
        np.asarray(sequence, dtype=np.uint8)
        for length in range(11)
        for sequence in product((0, 1), repeat=length)
    ]
    rng = np.random.default_rng(410018)
    sequences.extend(rng.integers(0, 2, size=size, dtype=np.uint8) for size in (32, 256, 1024, 5000))
    strided = np.tile(np.asarray([0, 1], dtype=np.uint8), 2048)[::2]
    sequences.append(strided)

    for sequence in sequences:
        expected = metrics._lz78_phrase_count(tuple(int(value) for value in sequence))
        assert metrics._lz78_phrase_count_binary(sequence) == expected


def test_binary_lz78_fast_path_rejects_nonbinary_or_wrong_dtype() -> None:
    import pytest

    with pytest.raises(ValueError, match="one-dimensional uint8"):
        metrics._lz78_phrase_count_binary(np.asarray([0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="only symbols 0 and 1"):
        metrics._lz78_phrase_count_binary(np.asarray([0, 2], dtype=np.uint8))
