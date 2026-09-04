"""Deterministic explicit perturbation outcomes for OD-TQ3."""

from __future__ import annotations

import numpy as np


QUALIFICATION_ID = "OD-TQ3-test-certificate"
QUALIFICATION_HASH = "od-tq3-protocol-serialization-v1"

PROTOCOL = {
    "perturbation_direction": "positive_x",
    "perturbation_time": "t0",
    "reference_attractor_or_state": "x_A",
    "return_criterion": "entered_reference_ball",
    "escape_criterion": "crossed_escape_boundary",
    "observation_horizon": 10.0,
    "non_return_is_escape": True,
}


def truth_outcomes() -> dict[str, np.ndarray | dict[str, object]]:
    """Return three 30-trial amplitude bins with known r50=2.0."""
    amplitudes = np.repeat(np.asarray([0.0, 1.0, 2.0]), 30)
    returned = np.concatenate(
        [
            np.ones(30, dtype=np.int8),
            np.concatenate([np.ones(18, dtype=np.int8), np.zeros(12, dtype=np.int8)]),
            np.concatenate([np.ones(9, dtype=np.int8), np.zeros(21, dtype=np.int8)]),
        ]
    )
    recovery_time = np.where(
        returned.astype(bool),
        0.5 + 0.25 * amplitudes,
        np.nan,
    )
    state = np.column_stack(
        [amplitudes, np.zeros_like(amplitudes), np.zeros_like(amplitudes)]
    ).astype(np.float32)
    return {
        "amplitudes": amplitudes.astype(np.float64),
        "returned": returned,
        "recovery_time_sec": recovery_time.astype(np.float64),
        "state": state,
        "time": np.arange(amplitudes.size, dtype=np.float64) * 0.1,
        "protocol": dict(PROTOCOL),
    }
