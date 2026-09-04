"""Outcome-blind methods audit for ds003670 native float rail semantics.

The audit reads only BIDS/EEGLAB metadata and, optionally, the decompressed
header prefix of one cited Zenodo CNT entry.  It never opens an FDT payload or
retains CNT signal samples.  Its purpose is to decide whether a native
float32 clipping/full-scale criterion can be bound before any FAR-EXT-002B
reconciliation.
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import json
from pathlib import Path
import re
import struct
import urllib.request
import zlib
from typing import Any, Mapping

import numpy as np


PROTOCOL_ID = "FAR-EXT-002C"
DATASET_ID = "ds003670"
ENTRY_PROTOCOL_ID = "FAR-EXT-002B"
ENTRY_STATUS = "NOT_TESTABLE"

NATIVE_RAIL_CRITERION_PASS = "NATIVE_RAIL_CRITERION_PASS"
NOT_TESTABLE = "NOT_TESTABLE"
AUDIT_SCOPE_FAILED = "AUDIT_SCOPE_FAILED"

ZENODO_RECORD_URL = "https://zenodo.org/api/records/4456079"
ZENODO_ARCHIVE_URL = (
    "https://zenodo.org/api/records/4456079/files/"
    "Exp1_Raw%20Data.zip/content"
)
ZENODO_ARCHIVE_SIZE = 13_368_162_568
ZENODO_ARCHIVE_MD5 = "7eea275088e87f1014e8353ce9e870a3"
ZENODO_CNT_ENTRY = (
    "Exp1_Raw Data/0101/GX_01_2019-09-24_15-45-53.cnt"
)
ZENODO_TAIL_BYTES = 8 * 1024 * 1024
ZENODO_HEADER_RANGE_BYTES = 256 * 1024
CNT_HEADER_MAX_DECOMPRESSED_BYTES = 1024 * 1024
EXPECTED_ENTRY_CERTIFICATE_SHA256 = (
    "9c4ec06d586e47e1cc235ce6c6b113118c810920c09251e2808c46caed4e8fcd"
)

PAPER_URL = "https://www.nature.com/articles/s41597-021-01046-y"
SUPPORTING_REPOSITORY_URL = (
    "https://github.com/ngebodh/GX_tES_EEG_Physio_Behavior"
)
SUPPORTING_REPOSITORY_REVISION = "358804f5b8b47f4753a51fc9990f60d8f37a5b27"
SUPPORTING_README_URL = (
    f"https://raw.githubusercontent.com/ngebodh/"
    f"GX_tES_EEG_Physio_Behavior/{SUPPORTING_REPOSITORY_REVISION}/README.md"
)
SUPPORTING_EXPORT_URL = (
    f"https://raw.githubusercontent.com/ngebodh/"
    f"GX_tES_EEG_Physio_Behavior/{SUPPORTING_REPOSITORY_REVISION}/"
    "GX_bids_export.m"
)
EEGO_DRIVER_MANUAL_URL = (
    "https://openvibe.inria.fr/openvibe/wp-content/uploads/2016/02/"
    "eego-openvibe-driver-manual-rev-1.0.pdf"
)
EEPROBE_FORMAT_URL = (
    "http://www.neuro.psychologie.uni-saarland.de/eeprobe/doc/cnt_riff.txt"
)
EEGO_FAQ_URL = "https://academy.ant-neuro.com/faq"


SOURCE_DOCUMENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "scientific_data_descriptor",
        "url": PAPER_URL,
        "evidence_targets": (
            "selected acquisition voltage range",
            "amplifier model",
            "export format",
        ),
        "scope": "acquisition_and_export_provenance",
    },
    {
        "id": "supporting_repository_readme",
        "url": SUPPORTING_README_URL,
        "revision": SUPPORTING_REPOSITORY_REVISION,
        "evidence_targets": (
            "raw CNT and BIDS SET formats",
            "export lineage",
        ),
        "scope": "metadata_and_export_provenance",
    },
    {
        "id": "supporting_repository_export_script",
        "url": SUPPORTING_EXPORT_URL,
        "revision": SUPPORTING_REPOSITORY_REVISION,
        "evidence_targets": (
            "CNT-to-SET/FDT conversion",
            "written BIDS manufacturer metadata",
            "explicit scale or unit transform",
        ),
        "scope": "export_implementation",
    },
    {
        "id": "eego_driver_manual",
        "url": EEGO_DRIVER_MANUAL_URL,
        "evidence_targets": (
            "referential input range options",
            "resolution",
            "mapping from acquisition values to exported float values",
        ),
        "scope": "generic_hardware_range",
    },
    {
        "id": "eeprobe_format",
        "url": EEPROBE_FORMAT_URL,
        "evidence_targets": (
            "channel calibration and unit fields",
            "CNT header layout",
        ),
        "scope": "format_scaling_semantics",
    },
    {
        "id": "eego_faq",
        "url": EEGO_FAQ_URL,
        "evidence_targets": (
            "export bit-depth choices",
            "native import/export semantics",
        ),
        "scope": "export_format_only",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap(value: object) -> object:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _field(value: object, name: str) -> object | None:
    value = _unwrap(value)
    if isinstance(value, Mapping):
        return value.get(name)
    try:
        return getattr(value, name)
    except AttributeError:
        return None


def _records(value: object) -> list[object]:
    value = _unwrap(value)
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return list(value.reshape(-1))
        if value.dtype == object:
            return list(value.reshape(-1))
    return [value]


def _text(value: object) -> str | None:
    value = _unwrap(value)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip() or None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.dtype.kind in {"U", "S"}:
            return "".join(str(item) for item in value.reshape(-1)).strip() or None
    text = str(value).strip()
    return text or None


def _load_set_header(path: Path) -> object:
    try:
        import scipy.io as sio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("scipy_required_for_set_header_audit") from exc
    try:
        variables = sio.whosmat(str(path))
    except Exception as exc:  # noqa: BLE001 - source-format boundary
        raise RuntimeError(f"set_header_inventory_unreadable:{path}") from exc
    variable_names = {name for name, _shape, _class in variables}
    if "EEG" in variable_names:
        raise RuntimeError(
            f"set_header_nested_struct_not_safely_auditable:{path}"
        )
    required = {"srate", "pnts", "nbchan", "data"}
    if not required.issubset(variable_names):
        raise RuntimeError(f"set_header_missing_required_fields:{path}")
    data_info = next(
        (item for item in variables if item[0] == "data"),
        None,
    )
    if data_info is None:
        raise RuntimeError(f"set_header_data_inventory_missing:{path}")
    _name, shape, matlab_class = data_info
    if matlab_class not in {"char", "string"} and np.prod(shape, dtype=int) > 1:
        raise RuntimeError(f"set_header_contains_embedded_samples:{path}")
    try:
        matrix = sio.loadmat(
            str(path),
            squeeze_me=True,
            struct_as_record=False,
            variable_names=sorted(required | {"chanlocs", "event", "reject"}),
        )
    except Exception as exc:  # noqa: BLE001 - source-format boundary
        raise RuntimeError(f"set_header_unreadable:{path}") from exc
    eeg = matrix.get("EEG")
    if eeg is not None:
        raise RuntimeError(f"set_header_nested_struct_not_safely_auditable:{path}")
    return {
        key: value
        for key, value in matrix.items()
        if key in required | {"chanlocs", "event", "reject"}
    }


def _header_field_names(value: object) -> list[str]:
    value = _unwrap(value)
    if isinstance(value, Mapping):
        return sorted(str(key) for key in value)
    if isinstance(value, np.void) and value.dtype.names:
        return sorted(str(key) for key in value.dtype.names)
    names = getattr(value, "_fieldnames", None)
    if names:
        return sorted(str(key) for key in names)
    return []


def _data_reference(value: object) -> str | None:
    for name in ("datfile", "data"):
        raw = _field(value, name)
        if (
            isinstance(raw, np.ndarray)
            and raw.size > 1
            and np.issubdtype(raw.dtype, np.number)
        ):
            raise RuntimeError("set_header_contains_embedded_samples")
        candidate = _text(raw)
        if candidate and candidate.lower().endswith(".fdt"):
            return candidate.replace("\\", "/")
    return None


def _channel_field_names(value: object) -> list[str]:
    names: set[str] = set()
    for record in _records(_field(value, "chanlocs")):
        names.update(_header_field_names(record))
    return sorted(names)


def _payload_allowlist(entry: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    binding = entry.get("payload_binding")
    if not isinstance(binding, Mapping) or binding.get("status") != "PASS":
        raise ValueError("far_ext_002b_payload_binding_missing")
    expected = binding.get("expected_payloads")
    if not isinstance(expected, list) or not all(
        isinstance(item, str) for item in expected
    ):
        raise ValueError("far_ext_002b_payload_allowlist_missing")
    normalized = {Path(item).as_posix() for item in expected}
    if any(
        path.startswith("/")
        or path.startswith("../")
        or "/../" in path
        or path == ".."
        for path in normalized
    ):
        raise ValueError("far_ext_002b_payload_allowlist_unsafe")
    set_paths = {path for path in normalized if path.lower().endswith(".set")}
    fdt_paths = {path for path in normalized if path.lower().endswith(".fdt")}
    if not set_paths or len(set_paths) != len(fdt_paths):
        raise ValueError("far_ext_002b_payload_allowlist_incomplete")
    return set_paths, fdt_paths


def _audit_set_headers(
    *,
    metadata_root: Path,
    eeg_root: Path,
    expected_set_paths: set[str],
    expected_fdt_paths: set[str],
) -> dict[str, Any]:
    actual_payloads = (
        {
            path.relative_to(eeg_root).as_posix()
            for path in eeg_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".set", ".fdt"}
        }
        if eeg_root.is_dir()
        else set()
    )
    expected_payloads = expected_set_paths | expected_fdt_paths
    inventory_errors = sorted(
        [
            *[f"missing:{path}" for path in expected_payloads - actual_payloads],
            *[f"unexpected:{path}" for path in actual_payloads - expected_payloads],
        ]
    )
    set_paths = sorted(
        path
        for path in (
            eeg_root.rglob("*.set") if eeg_root.is_dir() else []
        )
        if path.relative_to(eeg_root).as_posix() in expected_set_paths
    )
    if not set_paths:
        return {
            "status": AUDIT_SCOPE_FAILED,
            "reason": "no_set_headers_available",
            "set_header_count": 0,
            "fdt_payload_opened": False,
            "payload_inventory_errors": inventory_errors,
        }
    header_field_names: set[str] = set()
    channel_field_names: set[str] = set()
    references: set[str] = set()
    source_forms: set[str] = set()
    header_hashes: dict[str, str] = {}
    errors: list[str] = list(inventory_errors)
    for set_path in set_paths:
        relative = set_path.relative_to(eeg_root).as_posix()
        try:
            header = _load_set_header(set_path)
            source_forms.add(
                "nested_EEG_struct"
                if not isinstance(header, Mapping)
                else "flat_MATLAB_workspace"
            )
            header_field_names.update(_header_field_names(header))
            channel_field_names.update(_channel_field_names(header))
            reference = _data_reference(header)
            if reference is not None:
                references.add(reference)
            header_hashes[relative] = sha256_file(set_path)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{relative}:{type(exc).__name__}:{exc}")
    required_scale_fields = sorted(
        {
            "cal",
            "gain",
            "range",
            "unit",
            "scale",
        }.intersection(header_field_names | channel_field_names)
    )
    return {
        "status": "PASS" if not errors else AUDIT_SCOPE_FAILED,
        "reason": "set_headers_audited" if not errors else "set_header_errors",
        "metadata_root": str(metadata_root),
        "eeg_root": str(eeg_root),
        "set_header_count": len(set_paths),
        "set_header_paths": [
            path.relative_to(eeg_root).as_posix() for path in set_paths
        ],
        "set_header_sha256": header_hashes,
        "source_forms": sorted(source_forms),
        "header_fields": sorted(header_field_names),
        "channel_fields": sorted(channel_field_names),
        "scale_fields_present": required_scale_fields,
        "explicit_native_rail_binding": {},
        "fdt_references": sorted(references),
        "fdt_payload_opened": False,
        "payload_inventory_expected": sorted(expected_payloads),
        "payload_inventory_actual": sorted(actual_payloads),
        "payload_inventory_errors": inventory_errors,
        "errors": errors,
    }


def _audit_sidecars(
    metadata_root: Path,
    *,
    expected_set_paths: set[str],
) -> dict[str, Any]:
    expected_sidecar_paths = {
        str(Path(path).with_suffix(".json")).replace("\\", "/")
        for path in expected_set_paths
    }
    actual_sidecar_paths = (
        {
            path.relative_to(metadata_root).as_posix()
            for path in metadata_root.rglob("*_eeg.json")
            if path.is_file()
        }
        if metadata_root.is_dir()
        else set()
    )
    inventory_errors = sorted(
        f"missing:{path}"
        for path in expected_sidecar_paths - actual_sidecar_paths
    )
    ignored_sidecars = sorted(
        actual_sidecar_paths - expected_sidecar_paths
    )
    sidecar_paths = sorted(
        metadata_root / path
        for path in actual_sidecar_paths & expected_sidecar_paths
    )
    if not sidecar_paths:
        return {
            "status": AUDIT_SCOPE_FAILED,
            "reason": "no_eeg_sidecars_available",
            "sidecar_count": 0,
            "sidecar_inventory_errors": inventory_errors,
        }
    models: set[str] = set()
    counts: set[int] = set()
    software_filters: set[str] = set()
    channel_units: set[str] = set()
    errors: list[str] = list(inventory_errors)
    for path in sidecar_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            models.add(str(payload.get("ManufacturersModelName", "")))
            counts.add(int(payload["EEGChannelCount"]))
            software_filters.add(str(payload.get("SoftwareFilters", "")))
            channels_path = path.with_name(
                path.name.replace("_eeg.json", "_channels.tsv")
            )
            if channels_path.is_file():
                with channels_path.open(
                    encoding="utf-8",
                    newline="",
                ) as channels_handle:
                    for row in csv.DictReader(channels_handle, delimiter="\t"):
                        channel_units.add(str(row.get("units", "")))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path.relative_to(metadata_root).as_posix()}:{exc}")
    range_menu_values = sorted(
        {
            model
            for model in models
            if re.search(r"\b150\s*-\s*1000\s*mV\s*pp\b", model, re.I)
        }
    )
    return {
        "status": "PASS" if not errors else AUDIT_SCOPE_FAILED,
        "reason": "eeg_sidecars_audited" if not errors else "sidecar_errors",
        "sidecar_count": len(sidecar_paths),
        "sidecar_paths": [
            path.relative_to(metadata_root).as_posix() for path in sidecar_paths
        ],
        "sidecar_inventory_expected": sorted(expected_sidecar_paths),
        "sidecar_inventory_actual": sorted(actual_sidecar_paths),
        "sidecar_inventory_errors": inventory_errors,
        "ignored_sidecars_outside_payload_allowlist": ignored_sidecars,
        "manufacturer_model_values": sorted(models),
        "declared_eeg_channel_counts": sorted(counts),
        "software_filter_values": sorted(software_filters),
        "channel_unit_values": sorted(channel_units),
        "range_menu_values": range_menu_values,
        "range_menu_not_selected": bool(range_menu_values),
        "selected_input_range": None,
        "export_unit": None,
        "errors": errors,
    }


def _fetch_public_documents() -> list[dict[str, Any]]:
    marker_map = {
        "scientific_data_descriptor": (
            "acquisition voltages range",
            "1 v peak-to-peak",
            "eego sport",
        ),
        "supporting_repository_readme": (
            "raw eeg",
            ".cnt",
            ".set",
        ),
        "supporting_repository_export_script": (
            "bids_export",
            "ref input range",
            "copydata",
        ),
        "eego_driver_manual": (),
        "eeprobe_format": (
            "[basic channel data]",
            "calibration factor",
        ),
        "eego_faq": (
            "32 or 64bit .cnt",
            "export",
        ),
    }
    results: list[dict[str, Any]] = []
    for document in SOURCE_DOCUMENTS:
        try:
            request = urllib.request.Request(
                str(document["url"]),
                headers={"User-Agent": "NeuralManifoldDynamics-FAR-EXT-002C"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read(8 * 1024 * 1024)
                if response.read(1):
                    raise RuntimeError(
                        f"public_document_exceeds_limit:{document['id']}"
                    )
                decoded = body.decode("utf-8", errors="ignore").lower()
                markers = marker_map.get(str(document["id"]), ())
                results.append(
                    {
                        "id": document["id"],
                        "url": document["url"],
                        "revision": document.get("revision"),
                        "status": "PASS",
                        "final_url": response.geturl(),
                        "content_type": response.headers.get("Content-Type", ""),
                        "content_length": len(body),
                        "content_sha256": hashlib.sha256(body).hexdigest(),
                        "text_extractable": not body.startswith(b"%PDF"),
                        "expected_marker_presence": {
                            marker: marker in decoded for marker in markers
                        },
                        "evidence_targets": list(document["evidence_targets"]),
                    }
                )
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            http.client.HTTPException,
        ) as exc:
            results.append(
                {
                    "id": document["id"],
                    "url": document["url"],
                    "revision": document.get("revision"),
                    "status": AUDIT_SCOPE_FAILED,
                    "reason": f"{type(exc).__name__}:{exc}",
                    "evidence_targets": list(document["evidence_targets"]),
                }
            )
    return results


def _http_range(url: str, start: int, end: int) -> tuple[bytes, dict[str, str]]:
    if start < 0 or end < start:
        raise ValueError("invalid_http_range")
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "NeuralManifoldDynamics-FAR-EXT-002C",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read(end - start + 1)
        content_range = response.headers.get("Content-Range", "")
        range_parts = content_range.removeprefix("bytes ").split("/")
        if len(range_parts) != 2 or "-" not in range_parts[0]:
            raise RuntimeError(f"invalid_content_range:{content_range}")
        first, last = (
            int(value) for value in range_parts[0].split("-", 1)
        )
        if (
            response.status != 206
            or first != start
            or last != end
            or len(payload) != end - start + 1
        ):
            raise RuntimeError(f"unexpected_content_range:{content_range}")
        return payload, {
            "content_range": content_range,
            "content_length": response.headers.get("Content-Length", ""),
        }


def _zip64_extra(
    extra: bytes,
    *,
    uncompressed_overflow: bool,
    compressed_overflow: bool,
    offset_overflow: bool,
    disk_overflow: bool,
) -> dict[str, int]:
    position = 0
    while position + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, position)
        body = extra[position + 4:position + 4 + size]
        if tag == 0x0001:
            cursor = 0
            fields: dict[str, int] = {}
            for name, needed, width, format_string in (
                ("uncompressed_size", uncompressed_overflow, 8, "<Q"),
                ("compressed_size", compressed_overflow, 8, "<Q"),
                ("local_header_offset", offset_overflow, 8, "<Q"),
                ("disk_start", disk_overflow, 4, "<I"),
            ):
                if not needed:
                    continue
                if cursor + width > len(body):
                    raise RuntimeError("zip64_extra_truncated")
                fields[name] = struct.unpack_from(
                    format_string,
                    body,
                    cursor,
                )[0]
                cursor += width
            return fields
        position += 4 + size
    if (
        uncompressed_overflow
        or compressed_overflow
        or offset_overflow
        or disk_overflow
    ):
        raise RuntimeError("zip64_extra_missing")
    return {}


def _cnt_data_chunk_info(
    header: bytes | bytearray,
) -> tuple[int, int, bytes] | None:
    if len(header) < 12 or bytes(header[:4]) not in {b"RIFF", b"RF64"}:
        return None
    cursor = 12
    while cursor + 8 <= len(header):
        chunk_id = bytes(header[cursor:cursor + 4])
        chunk_size = int.from_bytes(
            header[cursor + 4:cursor + 8],
            "little",
        )
        if chunk_id in {b"CNT ", b"data", b"raw3", b"LIST", b"eeph"}:
            return cursor, chunk_size, chunk_id
        if chunk_id not in {b"fmt ", b"fact", b"JUNK", b"bext"}:
            raise RuntimeError(
                f"cnt_unknown_chunk_before_payload_boundary:{chunk_id!r}"
            )
        next_cursor = cursor + 8 + chunk_size + (chunk_size & 1)
        if next_cursor > len(header):
            return None
        cursor = next_cursor
    return None


def _remote_cnt_header_audit() -> dict[str, Any]:
    tail_start = ZENODO_ARCHIVE_SIZE - ZENODO_TAIL_BYTES
    tail, tail_meta = _http_range(
        ZENODO_ARCHIVE_URL,
        tail_start,
        ZENODO_ARCHIVE_SIZE - 1,
    )
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("zip_end_of_central_directory_missing")
    locator = eocd - 20
    if locator < 0 or tail[locator:locator + 4] != b"PK\x06\x07":
        raise RuntimeError("zip64_locator_missing")
    zip64_offset = struct.unpack_from("<Q", tail, locator + 8)[0]
    zip64_position = zip64_offset - tail_start
    if (
        zip64_position < 0
        or zip64_position + 56 > len(tail)
        or tail[zip64_position:zip64_position + 4] != b"PK\x06\x06"
    ):
        raise RuntimeError("zip64_eocd_missing")
    entry_count = struct.unpack_from("<Q", tail, zip64_position + 32)[0]
    central_size = struct.unpack_from("<Q", tail, zip64_position + 40)[0]
    central_offset = struct.unpack_from("<Q", tail, zip64_position + 48)[0]
    central_position = central_offset - tail_start
    if central_position < 0 or central_position + central_size > len(tail):
        raise RuntimeError("zip_central_directory_outside_range")
    record_format = "<4s6H3L5H2L"
    record_size = struct.calcsize(record_format)
    position = central_position
    selected: dict[str, Any] | None = None
    while position < central_position + central_size:
        values = struct.unpack_from(record_format, tail, position)
        name_length, extra_length, comment_length = values[10:13]
        name_start = position + record_size
        name = tail[name_start:name_start + name_length].decode(
            "utf-8",
            "replace",
        )
        extra_start = name_start + name_length
        extra = tail[extra_start:extra_start + extra_length]
        local_offset = values[16]
        compressed_size = values[8]
        uncompressed_size = values[9]
        zip64_fields = _zip64_extra(
            extra,
            uncompressed_overflow=uncompressed_size == 0xFFFFFFFF,
            compressed_overflow=compressed_size == 0xFFFFFFFF,
            offset_overflow=local_offset == 0xFFFFFFFF,
            disk_overflow=values[13] == 0xFFFF,
        )
        if uncompressed_size == 0xFFFFFFFF:
            uncompressed_size = zip64_fields["uncompressed_size"]
        if compressed_size == 0xFFFFFFFF:
            compressed_size = zip64_fields["compressed_size"]
        if local_offset == 0xFFFFFFFF:
            local_offset = zip64_fields["local_header_offset"]
        if name == ZENODO_CNT_ENTRY:
            selected = {
                "name": name,
                "compression_method": values[4],
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_header_offset": local_offset,
            }
            break
        position += record_size + name_length + extra_length + comment_length
    if selected is None:
        raise RuntimeError("zenodo_cnt_entry_missing")
    if (
        selected["local_header_offset"] is None
        or selected["compressed_size"] is None
        or selected["uncompressed_size"] is None
    ):
        raise RuntimeError("zip_entry_size_or_offset_unresolved")
    if selected["compression_method"] != 8:
        raise RuntimeError("zenodo_cnt_entry_not_deflated")
    local_start = int(selected["local_header_offset"])
    compressed_end = local_start + ZENODO_HEADER_RANGE_BYTES - 1
    prefix, prefix_meta = _http_range(
        ZENODO_ARCHIVE_URL,
        local_start,
        compressed_end,
    )
    if prefix[:4] != b"PK\x03\x04":
        raise RuntimeError("zip_local_header_missing")
    local_values = struct.unpack_from("<4s5H3L2H", prefix, 0)
    local_name_length, local_extra_length = local_values[9:11]
    data_start = 30 + local_name_length + local_extra_length
    compressed_prefix = prefix[data_start:]
    decompressor = zlib.decompressobj(-15)
    decompressed_header = bytearray()
    boundary_offset: int | None = None
    boundary_size: int | None = None
    boundary_fourcc: bytes | None = None
    pending = compressed_prefix
    while (
        pending
        and len(decompressed_header) < CNT_HEADER_MAX_DECOMPRESSED_BYTES
    ):
        output = decompressor.decompress(pending, max_length=1)
        pending = decompressor.unconsumed_tail
        if not output:
            if decompressor.eof:
                break
            continue
        decompressed_header.extend(output)
        chunk_info = _cnt_data_chunk_info(decompressed_header)
        if chunk_info is not None:
            boundary_offset, boundary_size, boundary_fourcc = chunk_info
            break
    if (
        boundary_offset is None
        or boundary_size is None
        or boundary_fourcc is None
    ):
        raise RuntimeError("cnt_payload_boundary_not_found_in_header_prefix")
    header_prefix = bytes(decompressed_header[:boundary_offset])
    signature = bytes(decompressed_header[:4])
    form_type = bytes(decompressed_header[8:12])
    markers = {
        "sampling_rate": b"[Sampling Rate]" in header_prefix,
        "basic_channel_data": b"[Basic Channel Data]" in header_prefix,
        "history_end": b"EOH" in header_prefix,
    }
    del pending, compressed_prefix, decompressor, decompressed_header
    return {
        "status": "PASS",
        "source": "zenodo_cnt_header_byte_range",
        "record_url": ZENODO_RECORD_URL,
        "archive_url": ZENODO_ARCHIVE_URL,
        "archive_size_bytes": ZENODO_ARCHIVE_SIZE,
        "archive_md5": ZENODO_ARCHIVE_MD5,
        "zip_entry_count": entry_count,
        "zip_tail_range": tail_meta,
        "cnt_entry": selected,
        "cnt_local_range": prefix_meta,
        "cnt_container": {
            "signature": signature.decode("ascii", "replace"),
            "form_type": form_type.decode("ascii", "replace"),
        "payload_boundary_fourcc": boundary_fourcc.decode(
            "ascii",
            "replace",
        ),
        "payload_boundary_offset": boundary_offset,
        "payload_boundary_size": boundary_size,
        "data_chunk_offset": (
            boundary_offset if boundary_fourcc == b"data" else None
        ),
        "data_chunk_size": (
            boundary_size if boundary_fourcc == b"data" else None
        ),
            "header_prefix_sha256": hashlib.sha256(header_prefix).hexdigest(),
            "header_prefix_bytes": len(header_prefix),
            "ascii_calibration_markers": markers,
        },
        "signal_bytes_retained": False,
        "signal_samples_inspected": False,
    }


def _criterion_assessment(
    *,
    sidecars: Mapping[str, Any],
    sets: Mapping[str, Any],
    documents: list[Mapping[str, Any]],
    remote: Mapping[str, Any],
) -> dict[str, Any]:
    fields = (
        "export_unit",
        "selected_input_range",
        "adc_or_export_transform",
        "rail_low",
        "rail_high",
        "rail_test",
    )
    candidates: dict[str, list[dict[str, Any]]] = {
        field: [] for field in fields
    }
    rejected: dict[str, list[dict[str, Any]]] = {
        field: [] for field in fields
    }
    evidence_sources: list[tuple[str, Mapping[str, Any]]] = [
        ("BIDS sidecars and SET headers", sidecars),
        ("SET header audit", sets),
        ("public documents", {"documents": documents}),
        ("remote CNT header audit", remote),
    ]
    for source_name, source in evidence_sources:
        binding = source.get("explicit_native_rail_binding", {})
        if not isinstance(binding, Mapping):
            continue
        for field in fields:
            value = binding.get(field)
            if value is not None:
                candidates[field].append(
                    {"source": source_name, "value": value}
                )
    for value in sidecars.get("range_menu_values", ()):
        rejected["selected_input_range"].append(
            {
                "source": "BIDS sidecars",
                "value": value,
                "reason": "selectable_menu_not_selected",
            }
        )
    for value in sidecars.get("channel_unit_values", ()):
        rejected["export_unit"].append(
            {
                "source": "BIDS channels.tsv",
                "value": value,
                "reason": "channel_unit_is_not_native_float_binding",
            }
        )
    if sets.get("scale_fields_present") == []:
        rejected["adc_or_export_transform"].append(
            {
                "source": "SET headers",
                "value": None,
                "reason": "no_scale_or_calibration_field_present",
            }
        )
    for document in documents:
        marker_presence = document.get("expected_marker_presence", {})
        if not isinstance(marker_presence, Mapping):
            continue
        for marker, present in marker_presence.items():
            if present:
                rejected["selected_input_range"].append(
                    {
                        "source": str(document.get("id", "public document")),
                        "value": marker,
                        "reason": "document_marker_without_complete_native_binding",
                    }
                )
    remote_markers = remote.get("cnt_container", {}).get(
        "ascii_calibration_markers",
        {},
    )
    if isinstance(remote_markers, Mapping):
        for marker, present in remote_markers.items():
            if present:
                rejected["rail_test"].append(
                    {
                        "source": "remote CNT header",
                        "value": marker,
                        "reason": "header_marker_without_native_float_rule",
                    }
                )
    field_ledger: dict[str, dict[str, Any]] = {}
    resolved: dict[str, Any] = {}
    for field in fields:
        values = {
            json.dumps(item["value"], sort_keys=True)
            for item in candidates[field]
        }
        unique = len(values) == 1 and bool(values)
        value = candidates[field][0]["value"] if unique else None
        if unique:
            resolved[field] = value
        field_ledger[field] = {
            "value": value,
            "status": "PASS" if unique else "UNRESOLVED",
            "unique": unique,
            "candidates": candidates[field],
            "rejected_candidates": rejected[field],
        }
    complete = len(resolved) == len(fields)
    return {
        **resolved,
        **{field: None for field in fields if field not in resolved},
        "field_ledger": field_ledger,
        "rejected_evidence": rejected,
        "range_menu_not_selected": bool(
            sidecars.get("range_menu_not_selected")
        ),
        "native_float_rail_status": (
            "PASS" if complete else "UNRESOLVED"
        ),
        "reason": (
            "native_float_rail_criterion_bound"
            if complete
            else "native_float_rail_criterion_unresolved"
        ),
    }


def audit_far_ext_002c(
    *,
    metadata_root: Path,
    eeg_root: Path,
    far002b_path: Path,
    fetch_documents: bool = True,
    probe_remote_header: bool = True,
) -> dict[str, Any]:
    if not far002b_path.is_file():
        return {
            "schema": "mndm.far_ext_002c_ds003670_native_rail_methods.v1",
            "protocol_id": PROTOCOL_ID,
            "dataset_id": DATASET_ID,
            "global_status": AUDIT_SCOPE_FAILED,
            "global_reason": "far_ext_002b_certificate_missing",
            "reconciliation_authorized": False,
            "audit_scope": {
                "fdt_payload_opened": False,
                "fdt_samples_inspected": False,
            },
        }
    entry = json.loads(far002b_path.read_text(encoding="utf-8"))
    if not isinstance(entry, dict):
        raise ValueError("far_ext_002b_certificate_not_object")
    entry_binding = {
        "path": str(far002b_path),
        "sha256": sha256_file(far002b_path),
        "expected_sha256": EXPECTED_ENTRY_CERTIFICATE_SHA256,
        "protocol_id": entry.get("protocol_id"),
        "dataset_id": entry.get("dataset_id"),
        "global_status": entry.get("global_status"),
    }
    if (
        entry_binding["sha256"] != EXPECTED_ENTRY_CERTIFICATE_SHA256
        or
        entry.get("protocol_id") != ENTRY_PROTOCOL_ID
        or entry.get("dataset_id") != DATASET_ID
        or entry.get("global_status") != ENTRY_STATUS
    ):
        return {
            "schema": "mndm.far_ext_002c_ds003670_native_rail_methods.v1",
            "protocol_id": PROTOCOL_ID,
            "dataset_id": DATASET_ID,
            "global_status": AUDIT_SCOPE_FAILED,
            "global_reason": "far_ext_002b_entry_binding_failed",
            "entry_far_ext_002b": entry_binding,
            "reconciliation_authorized": False,
            "audit_scope": {
                "fdt_payload_opened": False,
                "fdt_samples_inspected": False,
            },
        }
    try:
        expected_set_paths, expected_fdt_paths = _payload_allowlist(entry)
    except ValueError as exc:
        return {
            "schema": "mndm.far_ext_002c_ds003670_native_rail_methods.v1",
            "protocol_id": PROTOCOL_ID,
            "dataset_id": DATASET_ID,
            "global_status": AUDIT_SCOPE_FAILED,
            "global_reason": str(exc),
            "entry_far_ext_002b": entry_binding,
            "reconciliation_authorized": False,
            "audit_scope": {
                "fdt_payload_opened": False,
                "fdt_samples_inspected": False,
            },
        }
    sidecars = _audit_sidecars(
        metadata_root,
        expected_set_paths=expected_set_paths,
    )
    sets = _audit_set_headers(
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        expected_set_paths=expected_set_paths,
        expected_fdt_paths=expected_fdt_paths,
    )
    if fetch_documents:
        try:
            documents = _fetch_public_documents()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            documents = [
                {
                    "status": AUDIT_SCOPE_FAILED,
                    "reason": (
                        f"public_document_audit_failed:{type(exc).__name__}:"
                        f"{exc}"
                    ),
                }
            ]
    else:
        documents = [
            {
                "id": document["id"],
                "url": document["url"],
                "revision": document.get("revision"),
                "status": "NOT_RUN",
                "evidence_targets": list(document["evidence_targets"]),
            }
            for document in SOURCE_DOCUMENTS
        ]
    remote: dict[str, Any]
    if probe_remote_header:
        try:
            remote = _remote_cnt_header_audit()
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            struct.error,
            zlib.error,
            http.client.HTTPException,
        ) as exc:
            remote = {
                "status": AUDIT_SCOPE_FAILED,
                "reason": f"remote_cnt_header_audit_failed:{type(exc).__name__}:{exc}",
                "signal_bytes_retained": False,
                "signal_samples_inspected": False,
            }
    else:
        remote = {
            "status": "NOT_RUN",
            "reason": "remote_header_probe_not_requested",
            "signal_bytes_retained": False,
            "signal_samples_inspected": False,
        }
    scope_ok = (
        sidecars.get("status") == "PASS"
        and sets.get("status") == "PASS"
        and remote.get("status") in {"PASS", "NOT_RUN"}
        and all(
            document.get("status") in {"PASS", "NOT_RUN"}
            for document in documents
        )
    )
    criterion = _criterion_assessment(
        sidecars=sidecars,
        sets=sets,
        documents=documents,
        remote=remote,
    )
    criterion_pass = criterion["native_float_rail_status"] == "PASS"
    return {
        "schema": "mndm.far_ext_002c_ds003670_native_rail_methods.v1",
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "global_status": (
            NATIVE_RAIL_CRITERION_PASS
            if scope_ok and criterion_pass
            else NOT_TESTABLE
            if scope_ok
            else AUDIT_SCOPE_FAILED
        ),
        "global_reason": (
            criterion["reason"]
            if scope_ok
            else "audit_evidence_scope_unresolved"
        ),
        "entry_far_ext_002b": entry_binding,
        "source_documents": documents,
        "local_sidecar_audit": sidecars,
        "local_set_header_audit": sets,
        "remote_cnt_header_audit": remote,
        "criterion_assessment": criterion,
        "documentation_status": (
            "PASS"
            if scope_ok and criterion_pass
            else "PARTIAL"
            if scope_ok
            else "UNRESOLVED"
        ),
        "native_float_rail_status": criterion["native_float_rail_status"],
        "reconciliation_authorized": bool(scope_ok and criterion_pass),
        "far_003b_authorized": False,
        "audit_scope": {
            "metadata_opened": True,
            "set_headers_opened": True,
            "fdt_payload_opened": False,
            "fdt_samples_inspected": False,
            "cnt_signal_data_opened": False,
            "cnt_header_prefix_only": remote.get("status") == "PASS",
            "outcome_tables_opened": False,
            "nmd_outputs_opened": False,
            "mnps_calculated": False,
            "far_calculated": False,
        },
        "claim_boundary": (
            "FAR-EXT-002C audits only whether a native float32 rail criterion "
            "is documented. It establishes no post-ramp signal eligibility, "
            "MNPS trajectory, response magnitude, resilience value, or "
            "amplitude effect."
        ),
        "next_gate": (
            "Prepare a separate FAR-EXT-002B reconciliation preregistration "
            "only if every native float rail criterion field is uniquely bound."
        ),
    }
