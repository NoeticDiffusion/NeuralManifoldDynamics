"""Metadata conventions for DANDI 000009 (ALM-thalamus persistent activity)."""

from __future__ import annotations

from ..registry import DatasetAdapter, register


DATASET_000009 = register(
    DatasetAdapter(
        dataset_id="dandi_000009",
        name="Maintenance of persistent activity in a frontal thalamocortical loop",
        event_columns={
            "trial_start": ("start_time",),
            "trial_end": ("stop_time",),
            "sample": ("sample_start_time", "sample_time", "stimulus_onset_time"),
            "delay": ("delay_start_time", "delay_time"),
            "response": ("go_cue_time", "response_time", "movement_time"),
            "photostimulation": ("photostim_start_time", "photostimulation_time"),
        },
    )
)
