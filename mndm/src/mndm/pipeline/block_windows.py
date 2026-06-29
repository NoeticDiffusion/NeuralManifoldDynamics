"""Block-native window generation from inferred stage-block intervals.

Given a list of :class:`~mndm.pipeline.stage_blocking.StageBlockInterval`
objects (already inferred by the block-source layer) and a
:class:`BlockWindowSpec` profile, this module generates analysis windows that
are grounded in exact block boundaries rather than a global epoch grid.

Window kinds
------------
- ``sliding``    — dense sliding windows over the full block (default).
- ``tail``       — sliding windows anchored at block end only.
- ``post_offset``— windows in named bins *after* the block ends.
- ``partitioned``— sliding windows inside named sub-intervals of the block.

All generated windows carry block-relative metadata: ``relative_time_in_block_sec``
(center from block start), ``distance_to_block_end_sec`` (center to block end),
and ``relative_pos_0_1`` (normalised 0–1 position within the block).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BlockWindowSpec:
    """Profile configuration for generating windows from a block interval.

    Attributes
    ----------
    kind:
        Generation mode: ``"sliding"``, ``"tail"``, ``"post_offset"``, or
        ``"partitioned"``.
    window_length_sec:
        Duration of each generated window in seconds.
    step_sec:
        Step size between consecutive window starts in seconds.
    tail_sec:
        In ``tail`` mode, the length of the block-end tail to cover.
        Windows are generated inside ``[block_end - tail_sec, block_end]``.
    post_offset_bins:
        In ``post_offset`` mode, a sequence of ``(name, lo_sec, hi_sec)``
        tuples defining named bins relative to ``block_end``.
    partitions:
        In ``partitioned`` mode, a sequence of ``(name, start_offset, end_offset)``
        tuples.  Positive offsets are relative to ``block_start``; negative
        offsets are relative to ``block_end`` (e.g. ``-8.0`` means 8 s before end).
    min_windows_per_block:
        Minimum number of windows a block must produce to be included.
        Blocks yielding fewer windows are discarded entirely.
    min_block_sec:
        Minimum block duration in seconds.  Shorter blocks are skipped before
        any window generation is attempted.
    emit_relative_position:
        If True, ``relative_pos_0_1`` is computed.  Has no effect on output
        shape — the field is always present — but documents intent.
    """

    kind: str = "sliding"
    window_length_sec: float = 4.0
    step_sec: float = 2.0
    emit_relative_position: bool = True
    tail_sec: float = 8.0
    post_offset_bins: Tuple[Tuple[str, float, float], ...] = field(default_factory=tuple)
    partitions: Tuple[Tuple[str, float, float], ...] = field(default_factory=tuple)
    min_windows_per_block: int = 1
    min_block_sec: float = 0.0


@dataclass(frozen=True)
class BlockWindowRow:
    """One analysis window generated from a block interval.

    Attributes
    ----------
    block_id:
        Integer identifier of the source block.
    window_id_within_block:
        Sequential index of this window within the block (0-based).
    stage_code:
        Integer stage/condition code inherited from the source block.
    block_start_sec, block_end_sec, block_duration_sec:
        Absolute boundaries and duration of the source block (seconds).
    window_start_sec, window_end_sec, window_center_sec:
        Absolute boundaries and center of this window (seconds).
    relative_time_in_block_sec:
        ``window_center_sec - block_start_sec``.
    distance_to_block_end_sec:
        ``block_end_sec - window_center_sec``.
    relative_pos_0_1:
        Normalised position of the window center inside the block, clamped
        to ``[0, 1]``.  ``0`` = block start, ``1`` = block end.
    partition_label:
        Name of the partition or post-offset bin, or ``""`` for plain sliding.
    is_post_offset:
        ``True`` when the window falls after ``block_end_sec``.
    """

    block_id: int
    window_id_within_block: int
    stage_code: int
    block_start_sec: float
    block_end_sec: float
    block_duration_sec: float
    window_start_sec: float
    window_end_sec: float
    window_center_sec: float
    relative_time_in_block_sec: float
    distance_to_block_end_sec: float
    relative_pos_0_1: float
    partition_label: str = ""
    is_post_offset: bool = False


def generate_block_windows(
    blocks: Sequence,
    spec: BlockWindowSpec,
) -> List[BlockWindowRow]:
    """Generate analysis windows from a list of block intervals.

    Parameters
    ----------
    blocks:
        Iterable of :class:`~mndm.pipeline.stage_blocking.StageBlockInterval`
        objects.  Any object with ``start_sec``, ``end_sec``, ``stage_code``,
        and ``block_id`` attributes works.
    spec:
        Window profile configuration.

    Returns
    -------
    List[BlockWindowRow]
        Flat list of generated windows ordered by ``block_id`` then
        ``window_id_within_block``.
    """
    rows: List[BlockWindowRow] = []

    for block in blocks:
        block_start = float(block.start_sec)
        block_end = float(block.end_sec)
        block_dur = block_end - block_start

        if block_dur < spec.min_block_sec:
            continue

        block_rows: List[BlockWindowRow] = []

        if spec.kind == "sliding":
            block_rows = _generate_sliding(block, spec, block_start, block_end, block_dur)
        elif spec.kind == "tail":
            block_rows = _generate_tail(block, spec, block_start, block_end, block_dur)
        elif spec.kind == "post_offset":
            block_rows = _generate_post_offset(block, spec, block_start, block_end, block_dur)
        elif spec.kind == "partitioned":
            block_rows = _generate_partitioned(block, spec, block_start, block_end, block_dur)

        if len(block_rows) >= spec.min_windows_per_block:
            rows.extend(block_rows)

    return rows


# ---------------------------------------------------------------------------
# Internal per-kind generators
# ---------------------------------------------------------------------------

def _make_row(
    block: object,
    win_id: int,
    win_start: float,
    win_end: float,
    block_start: float,
    block_end: float,
    block_dur: float,
    partition_label: str = "",
    is_post_offset: bool = False,
) -> BlockWindowRow:
    """Build one BlockWindowRow from raw timing values."""
    win_center = (win_start + win_end) * 0.5
    rel_time = win_center - block_start
    dist_to_end = block_end - win_center
    rel_pos = (rel_time / block_dur) if block_dur > 0.0 else 0.0
    rel_pos = max(0.0, min(1.0, rel_pos))
    return BlockWindowRow(
        block_id=int(getattr(block, "block_id", 0)),
        window_id_within_block=win_id,
        stage_code=int(getattr(block, "stage_code", 0)),
        block_start_sec=block_start,
        block_end_sec=block_end,
        block_duration_sec=block_dur,
        window_start_sec=win_start,
        window_end_sec=win_end,
        window_center_sec=win_center,
        relative_time_in_block_sec=rel_time,
        distance_to_block_end_sec=dist_to_end,
        relative_pos_0_1=rel_pos,
        partition_label=partition_label,
        is_post_offset=is_post_offset,
    )


def _sliding_over_interval(
    block: object,
    spec: BlockWindowSpec,
    interval_start: float,
    interval_end: float,
    block_start: float,
    block_end: float,
    block_dur: float,
    partition_label: str = "",
    is_post_offset: bool = False,
    win_id_offset: int = 0,
) -> List[BlockWindowRow]:
    """Slide windows over [interval_start, interval_end]."""
    rows: List[BlockWindowRow] = []
    win_len = spec.window_length_sec
    step = spec.step_sec
    if win_len <= 0.0 or step <= 0.0:
        return rows
    if interval_end - interval_start < win_len - 1e-9:
        return rows
    t = interval_start
    win_id = win_id_offset
    while t + win_len <= interval_end + 1e-9:
        rows.append(
            _make_row(
                block, win_id, t, t + win_len,
                block_start, block_end, block_dur,
                partition_label=partition_label,
                is_post_offset=is_post_offset,
            )
        )
        win_id += 1
        t += step
    return rows


def _generate_sliding(
    block: object,
    spec: BlockWindowSpec,
    block_start: float,
    block_end: float,
    block_dur: float,
) -> List[BlockWindowRow]:
    return _sliding_over_interval(
        block, spec, block_start, block_end,
        block_start, block_end, block_dur,
    )


def _generate_tail(
    block: object,
    spec: BlockWindowSpec,
    block_start: float,
    block_end: float,
    block_dur: float,
) -> List[BlockWindowRow]:
    tail_start = max(block_start, block_end - spec.tail_sec)
    return _sliding_over_interval(
        block, spec, tail_start, block_end,
        block_start, block_end, block_dur,
        partition_label="tail",
    )


def _generate_post_offset(
    block: object,
    spec: BlockWindowSpec,
    block_start: float,
    block_end: float,
    block_dur: float,
) -> List[BlockWindowRow]:
    rows: List[BlockWindowRow] = []
    for bin_name, lo_sec, hi_sec in spec.post_offset_bins:
        bin_start = block_end + lo_sec
        bin_end = block_end + hi_sec
        rows.extend(
            _sliding_over_interval(
                block, spec, bin_start, bin_end,
                block_start, block_end, block_dur,
                partition_label=bin_name,
                is_post_offset=True,
                win_id_offset=len(rows),
            )
        )
    return rows


def _generate_partitioned(
    block: object,
    spec: BlockWindowSpec,
    block_start: float,
    block_end: float,
    block_dur: float,
) -> List[BlockWindowRow]:
    if not spec.partitions:
        return _generate_sliding(block, spec, block_start, block_end, block_dur)
    rows: List[BlockWindowRow] = []
    for part_name, start_offset, end_offset in spec.partitions:
        # Negative offsets are relative to block_end; positive to block_start.
        part_start = (
            block_end + start_offset if start_offset < 0 else block_start + start_offset
        )
        part_end = (
            block_end + end_offset if end_offset <= 0 else block_start + end_offset
        )
        part_start = max(part_start, block_start)
        part_end = min(part_end, block_end)
        rows.extend(
            _sliding_over_interval(
                block, spec, part_start, part_end,
                block_start, block_end, block_dur,
                partition_label=part_name,
                win_id_offset=len(rows),
            )
        )
    return rows
