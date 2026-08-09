"""Metadata conventions for DANDI 000006 (mouse ALM delay-response task)."""

from __future__ import annotations

from ..registry import DatasetAdapter, register


DATASET_000006 = register(
    DatasetAdapter(
        dataset_id="dandi_000006",
        name="Mouse anterior lateral motor cortex in delay response task",
        event_columns={
            "trial_start": ("start_time",),
            "trial_end": ("stop_time",),
            "sample": ("pole_in_time", "cue_start_time", "sample_start_time", "sample_time", "stimulus_onset_time"),
            "delay": ("delay_start_time", "delay_time"),
            "response": ("go_cue_time", "response_time", "movement_time"),
        },
    )
)
