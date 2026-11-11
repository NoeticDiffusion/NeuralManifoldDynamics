"""Binary LZ78 sequence-complexity estimator with exact-multiset references.

This preserves the repository's incremental LZ78 phrase semantics. Median
ties map to zero. A channel with a one-symbol sequence makes the complete
declared realization unsupported; channels are never silently dropped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Sequence

import numpy as np


SC_ESTIMATOR_ID = "SC.pipeline_native_window_lz78_exact_multiset_ratio.v1"
SC_SCHEMA_ID = "mndm.mnps_v3_2.sc_result.v1"
SC_SEED_DERIVATION_ID = "mndm.mnps_v3_1.pipeline_e2_seed_derivation.v1"
SC_MASTER_SEED = 310930


def _lz78_count_impl(symbols: np.ndarray) -> int:
    """Array-trie equivalent of the repository's incremental LZ78 parser."""
    size = symbols.size
    zero_child = np.full(size + 1, -1, dtype=np.int32)
    one_child = np.full(size + 1, -1, dtype=np.int32)
    terminal = np.zeros(size + 1, dtype=np.bool_)
    next_node = 1
    cursor = 0
    phrases = 0
    while cursor < size:
        node = 0
        end = cursor
        while end < size:
            symbol = int(symbols[end])
            child = zero_child[node] if symbol == 0 else one_child[node]
            if child < 0:
                child = next_node
                next_node += 1
                if symbol == 0:
                    zero_child[node] = child
                else:
                    one_child[node] = child
                terminal[child] = True
                phrases += 1
                end += 1
                break
            node = int(child)
            end += 1
            if not terminal[node]:
                terminal[node] = True
                phrases += 1
                break
        cursor = end
    return phrases


try:  # Optional speed-up; the estimator has a pure-Python-compatible fallback.
    from numba import njit

    _lz78_count = njit(cache=False)(_lz78_count_impl)
except Exception:  # pragma: no cover - depends on the optional local runtime
    _lz78_count = _lz78_count_impl


@dataclass(frozen=True)
class SCResult:
    estimator_id: str
    schema_id: str
    value: float | None
    units: str
    computed: bool
    valid: bool
    qualified: bool
    status: str
    reason: str | None
    channel_count: int
    sample_count: int
    reference_count_per_channel: int
    channel_labels: tuple[str, ...] | None
    channel_results: tuple[dict[str, object], ...]
    qc: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["channel_labels"] = list(self.channel_labels) if self.channel_labels is not None else None
        payload["channel_results"] = [dict(row) for row in self.channel_results]
        payload["qc"] = list(self.qc)
        return payload


def _unsupported(reason: str, *, channels: int = 0, samples: int = 0,
                 refs: int = 0, labels: tuple[str, ...] | None = None,
                 channel_results: tuple[dict[str, object], ...] = (),
                 computed: bool = False) -> SCResult:
    return SCResult(SC_ESTIMATOR_ID, SC_SCHEMA_ID, None, "1", computed, False,
                    False, "unsupported", reason, channels, samples, refs,
                    labels, channel_results, ())


def compute_sc_lz78_exact_multiset_ratio(
    signals: np.ndarray,
    channel_labels: Sequence[str] | None,
    *,
    source_identity: str,
    segment_id: str,
    start_seconds: float,
    end_seconds: float,
    reference_count: int = 32,
    master_seed: int = SC_MASTER_SEED,
) -> SCResult:
    """Compute median channel LZ78 phrase-count / shuffled-reference ratio.

    Each channel is thresholded at its arithmetic median (strictly greater is
    one; ties are zero). The denominator is the median LZ78 count from
    ``reference_count`` PCG64 permutations of that exact binary multiset.
    Seeds reproduce the existing v3.1 pipeline derivation. Each permutation
    index vector and shuffled sequence is hashed for audit.
    """
    try:
        raw = np.asarray(signals)
        if np.iscomplexobj(raw):
            return _unsupported("complex_signal_not_supported")
        values = np.asarray(raw, dtype=np.float64)
        start = float(start_seconds)
        end = float(end_seconds)
    except (TypeError, ValueError, OverflowError):
        return _unsupported("invalid_numeric_input")
    if values.ndim != 2:
        return _unsupported("signals_must_be_channels_by_samples")
    channels, samples = values.shape
    if channels < 1 or samples < 2:
        return _unsupported("insufficient_channel_or_sample_count", channels=channels, samples=samples)
    if not np.isfinite(values).all():
        return _unsupported("non_finite_sample", channels=channels, samples=samples)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return _unsupported("invalid_window_bounds", channels=channels, samples=samples)
    if isinstance(reference_count, bool) or not isinstance(reference_count, (int, np.integer)) or reference_count < 8:
        return _unsupported("reference_count_must_be_integer_at_least_8", channels=channels, samples=samples)
    if isinstance(master_seed, bool) or not isinstance(master_seed, (int, np.integer)):
        return _unsupported("master_seed_must_be_integer", channels=channels, samples=samples)
    if not isinstance(source_identity, str) or not source_identity.strip() or not isinstance(segment_id, str) or not segment_id.strip():
        return _unsupported("source_and_segment_identity_required", channels=channels, samples=samples)
    if channel_labels is None or isinstance(channel_labels, (str, bytes)):
        return _unsupported("channel_labels_required", channels=channels, samples=samples)
    labels = tuple(str(label).strip() for label in channel_labels)
    if len(labels) != channels or any(not label for label in labels):
        return _unsupported("channel_label_count_mismatch_or_empty", channels=channels, samples=samples, labels=labels)
    if len({label.casefold() for label in labels}) != len(labels):
        return _unsupported("duplicate_channel_labels", channels=channels, samples=samples, labels=labels)

    per_channel: list[dict[str, object]] = []
    ratios: list[float] = []
    any_unsupported = False
    for row, label in zip(values, labels, strict=True):
        threshold = float(np.median(row))
        symbols = np.asarray(row > threshold, dtype=np.uint8)
        zeros = int(symbols.size - np.count_nonzero(symbols))
        ones = int(np.count_nonzero(symbols))
        detail: dict[str, object] = {
            "channel_label": label,
            "threshold": threshold,
            "tie_count": int(np.count_nonzero(row == threshold)),
            "symbol_counts": {"0": zeros, "1": ones},
        }
        if zeros == 0 or ones == 0:
            detail.update({"status": "unsupported", "reason": "sequence_has_fewer_than_two_symbols",
                           "observed_phrase_count": None, "reference_phrase_counts": [],
                           "permutation_index_sha256": [], "permutation_symbols_sha256": [],
                           "ratio": None})
            per_channel.append(detail)
            any_unsupported = True
            continue

        observed = int(_lz78_count(symbols))
        seed_material = "\0".join((
            SC_SEED_DERIVATION_ID,
            str(int(master_seed)),
            source_identity,
            segment_id,
            float(start).hex(),
            float(end).hex(),
            label.strip().casefold(),
            str(int(reference_count)),
        )).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big", signed=False)
        rng = np.random.Generator(np.random.PCG64(seed))
        reference_counts: list[int] = []
        permutation_hashes: list[str] = []
        symbols_hashes: list[str] = []
        for _ in range(int(reference_count)):
            permutation = rng.permutation(symbols.size)
            permuted = np.ascontiguousarray(symbols[permutation], dtype=np.uint8)
            reference_counts.append(int(_lz78_count(permuted)))
            permutation_hashes.append(hashlib.sha256(np.asarray(permutation, dtype="<i8").tobytes(order="C")).hexdigest())
            symbols_hashes.append(hashlib.sha256(permuted.tobytes(order="C")).hexdigest())
        reference_median = float(np.median(np.asarray(reference_counts, dtype=np.float64)))
        if not math.isfinite(reference_median) or reference_median <= 0.0:
            detail.update({"status": "unsupported", "reason": "undefined_exact_multiset_reference",
                           "observed_phrase_count": observed, "reference_phrase_counts": reference_counts,
                           "permutation_index_sha256": permutation_hashes,
                           "permutation_symbols_sha256": symbols_hashes, "derived_seed": seed,
                           "reference_phrase_count_median": reference_median, "ratio": None})
            per_channel.append(detail)
            any_unsupported = True
            continue
        ratio = float(observed / reference_median)
        detail.update({"status": "computed_exploratory", "reason": None,
                       "observed_phrase_count": observed, "reference_phrase_counts": reference_counts,
                       "permutation_index_sha256": permutation_hashes,
                       "permutation_symbols_sha256": symbols_hashes, "derived_seed": seed,
                       "reference_phrase_count_median": reference_median, "ratio": ratio})
        per_channel.append(detail)
        ratios.append(ratio)

    if any_unsupported or len(ratios) != channels:
        first_reason = next((str(row["reason"]) for row in per_channel if row.get("status") == "unsupported"),
                            "one_or_more_channels_undefined")
        return _unsupported(first_reason, channels=channels, samples=samples,
                            refs=int(reference_count), labels=labels,
                            channel_results=tuple(per_channel), computed=True)
    value = float(np.median(np.asarray(ratios, dtype=np.float64)))
    valid = math.isfinite(value)
    return SCResult(
        SC_ESTIMATOR_ID, SC_SCHEMA_ID, value if valid else None, "1", True,
        valid, False, "computed_exploratory" if valid else "unsupported",
        None if valid else "non_finite_median_channel_ratio", channels, samples,
        int(reference_count), labels, tuple(per_channel),
        ("all_channels_required", "median_ties_symbolized_as_zero"),
    )


__all__ = [
    "SC_ESTIMATOR_ID", "SC_SCHEMA_ID", "SC_SEED_DERIVATION_ID", "SC_MASTER_SEED",
    "SCResult", "compute_sc_lz78_exact_multiset_ratio",
]
