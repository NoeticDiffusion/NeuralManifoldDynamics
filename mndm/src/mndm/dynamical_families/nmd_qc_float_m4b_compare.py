"""Canonical M4B comparison helpers.

These helpers normalize two result encodings.  They do not execute either
evaluator and do not construct an expected answer from production output.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


def _trigger_identity(
    rule_id: str,
    reasons: list[str] | tuple[str, ...],
) -> list[str]:
    return sorted({f"{rule_id}:{reason}" for reason in reasons})


def _canonical_rule_flags(
    flags: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "channel_index": int(flag["channel_index"]),
                "segment_index": int(flag["segment_index"]),
                "rule_id": str(flag["rule_id"]),
                "status": str(flag["status"]),
                "trigger_identity": sorted(
                    str(value) for value in flag.get("trigger_identity", [])
                ),
            }
            for flag in flags
        ],
        key=lambda value: (
            value["channel_index"],
            value["segment_index"],
            value["rule_id"],
        ),
    )


def canonicalize_reference_qc(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the independent evaluator's already-canonical result."""
    channels = [
        {
            "channel_index": int(channel["channel_index"]),
            "channel_name": str(channel["channel_name"]),
            "status": str(channel["status"]),
            "invalid_reason_ids": sorted(
                str(value) for value in channel.get("invalid_reason_ids", [])
            ),
            "unresolved_reason_ids": sorted(
                str(value) for value in channel.get("unresolved_reason_ids", [])
            ),
            "finite_sample_count": int(channel["finite_sample_count"]),
        }
        for channel in result.get("channel_flags", [])
    ]
    segments = [
        {
            "channel_index": int(segment["channel_index"]),
            "segment_index": int(segment["segment_index"]),
            "start": int(segment["start"]),
            "stop": int(segment["stop"]),
            "status": str(segment["status"]),
            "finite_sample_count": int(segment["finite_sample_count"]),
        }
        for segment in result.get("segment_flags", [])
    ]
    return {
        "recording": {
            "recording_status": str(result["recording_status"]),
            "evaluated_support": result["evaluated_support"],
            "uncovered_support": result["uncovered_support"],
        },
        "channel": sorted(channels, key=lambda value: value["channel_index"]),
        "segment": sorted(
            segments,
            key=lambda value: (
                value["channel_index"],
                value["segment_index"],
            ),
        ),
        "rule": _canonical_rule_flags(list(result.get("rule_flags", []))),
    }


def canonicalize_production_qc(
    result: Mapping[str, Any],
    *,
    expected_channel_indices: list[int],
    expected_channel_names: list[str],
) -> dict[str, Any]:
    """Project production output into the frozen M4B comparison schema."""
    channels: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    for channel in result.get("channel_flags", []):
        channel_segments = list(channel.get("segment_flags", []))
        channels.append(
            {
                "channel_index": int(channel["channel_index"]),
                "channel_name": str(channel.get("channel_name")),
                "status": str(channel["status"]),
                "invalid_reason_ids": sorted(
                    str(value) for value in channel.get("invalid_reasons", [])
                ),
                "unresolved_reason_ids": sorted(
                    str(value) for value in channel.get("unresolved_reasons", [])
                ),
                "finite_sample_count": sum(
                    int(segment["finite_sample_count"])
                    for segment in channel_segments
                ),
            }
        )
        for segment in channel_segments:
            segments.append(
                {
                    "channel_index": int(segment["channel_index"]),
                    "segment_index": int(segment["segment_index"]),
                    "start": int(segment["start"]),
                    "stop": int(segment["stop"]),
                    "status": str(segment["status"]),
                    "finite_sample_count": int(segment["finite_sample_count"]),
                }
            )
            for rule_id, rule in segment.get("rule_statuses", {}).items():
                if rule_id in {"R5", "R8"}:
                    continue
                rules.append(
                    {
                        "channel_index": int(segment["channel_index"]),
                        "segment_index": int(segment["segment_index"]),
                        "rule_id": str(rule_id),
                        "status": str(rule["status"]),
                        "trigger_identity": _trigger_identity(
                            str(rule_id),
                            list(rule.get("reasons", [])),
                        ),
                    }
                )

    timebase = result.get("timebase_status", {})
    rules.append(
        {
            "channel_index": -1,
            "segment_index": -1,
            "rule_id": "R5",
            "status": str(timebase.get("status")),
            "trigger_identity": _trigger_identity(
                "R5",
                [str(timebase.get("reason"))],
            ),
        }
    )
    selection_matches = (
        list(result.get("required_channel_indices", []))
        == expected_channel_indices
        and list(result.get("required_channel_names", []))
        == expected_channel_names
    )
    rules.append(
        {
            "channel_index": -1,
            "segment_index": -1,
            "rule_id": "R8",
            "status": (
                "TECHNICALLY_ADMISSIBLE"
                if selection_matches
                else "TECHNICAL_STATUS_UNRESOLVED"
            ),
            "trigger_identity": (
                []
                if selection_matches
                else ["R8:CHANNEL_SELECTION_RESULT_MISMATCH"]
            ),
        }
    )
    return {
        "recording": {
            "recording_status": str(result["recording_status"]),
            "evaluated_support": result["evaluated_support"],
            "uncovered_support": result["uncovered_support"],
        },
        "channel": sorted(channels, key=lambda value: value["channel_index"]),
        "segment": sorted(
            segments,
            key=lambda value: (
                value["channel_index"],
                value["segment_index"],
            ),
        ),
        "rule": _canonical_rule_flags(rules),
    }


def compare_canonical_qc(
    reference: Mapping[str, Any],
    production: Mapping[str, Any],
) -> dict[str, Any]:
    """Return structured equality plus first-class mismatch paths."""
    if reference == production:
        return {"matches": True, "mismatches": []}
    mismatches: list[dict[str, Any]] = []

    def walk(left: Any, right: Any, path: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right), key=str):
                if key not in left or key not in right:
                    mismatches.append(
                        {
                            "path": f"{path}.{key}",
                            "reference": left.get(key),
                            "production": right.get(key),
                        }
                    )
                else:
                    walk(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                mismatches.append(
                    {
                        "path": path,
                        "reference": left,
                        "production": right,
                    }
                )
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                walk(left_item, right_item, f"{path}[{index}]")
            return
        if left != right:
            mismatches.append(
                {
                    "path": path,
                    "reference": left,
                    "production": right,
                }
            )

    walk(reference, production, "$")
    return {
        "matches": False,
        "mismatches": mismatches,
    }


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
