"""HDF5 writer for MNPS tensor outputs.

Writes :class:`mndm.schema.MNPSPayload` to a structured HDF5 file (time series,
MNPS coordinates, Jacobians, labels, optional regional and extension groups).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None  # type: ignore
import numpy as np

from mndm.schema import MNPSPayload, normalize_payload

logger = logging.getLogger(__name__)


def _sanitize_h5_key(key: str) -> str:
    """Return a key safe for use as a flat HDF5 name."""
    return str(key).replace("/", "_").replace("\\", "_")


def _create_dataset(parent: Any, name: str, data, compression: str = "gzip", compression_opts: int = 4) -> None:
    """Create a dataset, handling scalar vs array data correctly.

    HDF5 does not support chunking/compression for true scalar datasets,
    so we disable compression in that case to avoid runtime errors.
    """

    # Normalize to numpy array to inspect shape
    arr = np.asarray(data)
    if arr.dtype.kind == "U":
        # h5py cannot store NumPy Unicode dtype (<U*) directly.
        # Use variable-length UTF-8 strings for compatibility.
        str_dtype = h5py.string_dtype(encoding="utf-8")
        parent.create_dataset(name, data=arr.astype(str).astype(object), dtype=str_dtype)
        return
    if arr.shape == ():
        # Scalar dataset – no compression/chunking allowed
        parent.create_dataset(name, data=arr)
        return

    kwargs: dict[str, Any] = {}
    # Heuristic: chunk/shuffle for larger arrays.
    if arr.size >= 10_000:
        kwargs["chunks"] = True
        kwargs["shuffle"] = True
    # Heuristic: gzip only when it is likely worthwhile.
    if arr.nbytes >= 256_000:
        kwargs["compression"] = compression
        kwargs["compression_opts"] = compression_opts
    parent.create_dataset(name, data=arr, **kwargs)


def _prepare_attr_value(value: Any) -> Any:
    """Internal helper: prepare attr value."""
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (bool, int, float, np.integer, np.floating, np.bool_)):
        return value
    if isinstance(value, np.ndarray):
        # Keep attrs lightweight and HDF5-compatible.
        if value.dtype == object:
            return json.dumps(value.tolist(), ensure_ascii=False)
        if value.size > 10_000:
            return f"<ndarray shape={tuple(value.shape)} dtype={value.dtype}>"
        if value.dtype.kind == "U":
            return value.astype("S")
        return value
    if isinstance(value, (list, tuple, set)):
        sequence = list(value)
        if not sequence:
            return np.asarray(sequence)
        if all(isinstance(v, (str, bytes)) for v in sequence):
            return np.asarray([str(v).encode("utf-8") for v in sequence], dtype="S")
        try:
            return np.asarray(sequence)
        except Exception:
            return json.dumps(sequence, ensure_ascii=False)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _write_json_string_dataset(parent: Any, name: str, value: Mapping[str, Any]) -> None:
    """Write a UTF-8 JSON dataset for nested metadata payloads."""
    text = json.dumps(value, ensure_ascii=False)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    parent.create_dataset(name, data=text, dtype=str_dtype)


def _write_extensions_group(h5: h5py.File, extensions: Mapping[str, Any]) -> None:
    """Write nested extension outputs under an ``extensions`` group.

    The ``extensions`` mapping is expected to have a structure like:

    .. code-block:: python

        {
            "e_kappa": {"time": ..., "energy": ..., "kappa": ...},
            "rfm": {"times": ..., "eigvals": ..., "eigvecs": ..., "dominance": ...},
            ...
        }

    Nested dicts become subgroups; arrays / scalars become datasets.
    """

    if not isinstance(extensions, Mapping) or not extensions:
        return

    root = h5.require_group("extensions")

    def _write_mapping(parent: h5py.Group, key: str, value: Any) -> None:
        """Internal helper: write mapping."""
        safe_key = _sanitize_h5_key(str(key))
        if isinstance(value, Mapping):
            sub = parent.require_group(safe_key)
            for k2, v2 in value.items():
                _write_mapping(sub, str(k2), v2)
        else:
            # Delegate dtype handling to _create_dataset
            _create_dataset(parent, safe_key, value)

    for name, payload in extensions.items():
        _write_mapping(root, str(name), payload)


def _write_feature_surface_group(
    h5: h5py.File,
    *,
    group_name: str,
    values: Optional[np.ndarray],
    names: Optional[list[str]],
    feature_metadata: Optional[Mapping[str, Any]],
    export_transform: str,
) -> None:
    """Internal helper: write feature surface group."""
    if values is None or np.size(values) == 0:
        return
    group = h5.require_group(group_name)
    _create_dataset(group, "values", np.asarray(values, dtype=np.float32))
    if names:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        names_utf8 = np.asarray([str(v) for v in names], dtype=object)
        group.create_dataset("names", data=names_utf8, dtype=str_dtype)
    if isinstance(feature_metadata, Mapping) and feature_metadata:
        metadata_group = group.require_group("metadata")
        for key, value in feature_metadata.items():
            _create_dataset(metadata_group, str(key), value)
    group.attrs["alignment"] = "per_timepoint"
    group.attrs["export_transform"] = export_transform
    group.attrs["feature_contract_version"] = "v1"
    if names:
        order_payload = json.dumps([str(v) for v in names], ensure_ascii=False, separators=(",", ":"))
        group.attrs["feature_order_hash"] = hashlib.sha256(order_payload.encode("utf-8")).hexdigest()
        group.attrs["n_features"] = int(len(names))


def write_h5(
    out_path: Path,
    dataset_id: str,
    payload: MNPSPayload,
    manifest: Optional[Mapping[str, Any]] = None,
    jacobian_diagnostics: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write MNPS tensors to HDF5 following the shared schema.

    Args:
        out_path: Output ``.h5`` file path (parent directories are created).
        dataset_id: Dataset identifier stored in file attributes.
        payload: Normalized MNPS payload (see :func:`mndm.schema.normalize_payload`).
        manifest: Optional run manifest dict serialized under ``manifest_json``.
        jacobian_diagnostics: Optional small Jacobian diagnostics arrays or scalars.

    Returns:
        ``out_path`` after a successful write.

    Raises:
        RuntimeError: If ``h5py`` is not installed.
    """
    if h5py is None:
        raise RuntimeError("h5py is required to write HDF5 outputs.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = normalize_payload(payload)

    with h5py.File(out_path, "w") as h5:
        h5.attrs["dataset_id"] = dataset_id

        # Canonical MNPS notation (v1.2 spec)
        h5.attrs["mnps_axis_names"] = _prepare_attr_value(["m", "d", "e"])

        for attr_key, attr_val in payload.attrs.items():
            # Don't serialize Python None as the string "None" into HDF5 attrs.
            # (Downstream code treats missing attrs differently from the literal string.)
            if attr_val is None:
                continue
            h5.attrs[attr_key] = _prepare_attr_value(attr_val)

        if manifest:
            manifest_json = json.dumps(manifest, ensure_ascii=False)
            str_dtype = h5py.string_dtype(encoding="utf-8")
            h5.create_dataset("manifest_json", data=manifest_json, dtype=str_dtype)
            h5.attrs["manifest_bytes"] = int(len(manifest_json.encode("utf-8")))
            # Backward-compatible small-manifest attr (avoid huge attrs).
            if h5.attrs["manifest_bytes"] <= 65535:
                h5.attrs["manifest"] = manifest_json

        # Convenience: expose participant meta and derived fields as top-level attrs
        # so downstream consumers don't need to parse the manifest/JSON.
        try:
            pmeta = payload.attrs.get("participant_meta", {}) if isinstance(payload.attrs, Mapping) else {}
            pmeta_source = payload.attrs.get("participant_meta_source", {}) if isinstance(payload.attrs, Mapping) else {}
            pmeta_mapped = payload.attrs.get("participant_mapped_meta", {}) if isinstance(payload.attrs, Mapping) else {}
            if isinstance(pmeta, Mapping):
                # Flatten common scalar fields under meta_*
                for k, v in pmeta.items():
                    if isinstance(v, (str, int, float, bool, np.integer, np.floating, np.bool_)):
                        h5.attrs[f"meta_{k}"] = _prepare_attr_value(v)

                participant_grp = h5.require_group("participant")
                for k, v in pmeta.items():
                    if isinstance(v, (str, int, float, bool, np.integer, np.floating, np.bool_)):
                        participant_grp.attrs[f"field_{_sanitize_h5_key(str(k))}"] = _prepare_attr_value(v)
                _write_json_string_dataset(participant_grp, "row_json", pmeta)

                if isinstance(pmeta_source, Mapping):
                    for k, v in pmeta_source.items():
                        if v is None:
                            continue
                        participant_grp.attrs[f"source_{_sanitize_h5_key(str(k))}"] = _prepare_attr_value(v)
                    if pmeta_source:
                        _write_json_string_dataset(participant_grp, "source_json", pmeta_source)

                if isinstance(pmeta_mapped, Mapping):
                    for k, v in pmeta_mapped.items():
                        if v is None:
                            continue
                        participant_grp.attrs[f"mapped_{_sanitize_h5_key(str(k))}"] = _prepare_attr_value(v)
                    if pmeta_mapped:
                        _write_json_string_dataset(participant_grp, "mapped_json", pmeta_mapped)

                # Derive 'group' (normalized) from common keys
                group_raw = None
                if isinstance(pmeta_mapped, Mapping):
                    mapped_group = pmeta_mapped.get("group")
                    if isinstance(mapped_group, (str, int, float)):
                        group_raw = str(mapped_group).strip()
                if not group_raw:
                    for key in ("Group", "group", "Diagnosis", "diagnosis", "Cohort", "cohort", "condition", "Condition"):
                        if key in pmeta and isinstance(pmeta[key], (str, int, float)):
                            group_raw = str(pmeta[key]).strip()
                            break
                if group_raw:
                    norm_map = {
                        "pd": "Parkinson",
                        "parkinson": "Parkinson",
                        "ctl": "Control",
                        "control": "Control",
                        "hc": "Healthy",
                        "healthy": "Healthy",
                        "ad": "AD",
                        "alzheimer": "AD",
                        "patient": "Patient",
                        "case": "Patient",
                        "subject": "Subject",
                    }
                    key_norm = group_raw.lower().replace("-", "").replace("_", "")
                    # Only set if the payload didn't already provide a canonical group.
                    if "group" not in h5.attrs:
                        h5.attrs["group"] = _prepare_attr_value(norm_map.get(key_norm, group_raw))

                # Derive 'condition' (e.g., medication status) if per-session fields exist
                condition_val = None
                if isinstance(pmeta_mapped, Mapping):
                    mapped_condition = pmeta_mapped.get("condition")
                    if isinstance(mapped_condition, (str, int, float)):
                        condition_val = mapped_condition
                # Infer session index from dataset_id like dsXXXX:sub-001[:ses-01]
                ses_tag = None
                try:
                    parts = str(dataset_id).split(":")
                    if len(parts) >= 3 and parts[2].startswith("ses-"):
                        ses_tag = parts[2]
                except Exception:
                    ses_tag = None
                # Map session tag to meta field names (dataset-dependent; best-effort)
                if ses_tag == "ses-01" and "sess1_Med" in pmeta:
                    condition_val = pmeta.get("sess1_Med")
                elif ses_tag == "ses-02" and "sess2_Med" in pmeta:
                    condition_val = pmeta.get("sess2_Med")
                else:
                    # Fallback to generic fields
                    for key in ("condition", "Condition", "Medication", "Med", "Status"):
                        if key in pmeta and isinstance(pmeta[key], (str, int, float)):
                            condition_val = pmeta[key]
                            break
                if condition_val is not None:
                    # Only set if the payload didn't already provide a canonical condition
                    # (e.g. ds003059 uses session_map → condition to avoid overwrites).
                    if "condition" not in h5.attrs:
                        h5.attrs["condition"] = _prepare_attr_value(condition_val)
                if "task" not in h5.attrs and isinstance(pmeta_mapped, Mapping):
                    mapped_task = pmeta_mapped.get("task")
                    if isinstance(mapped_task, (str, int, float)):
                        h5.attrs["task"] = _prepare_attr_value(mapped_task)
        except Exception:
            # Do not fail writing if convenience attrs derivation fails
            pass

        # Ensure a stable subject_id attr exists (downstream often expects it).
        # Prefer explicit payload attr; fall back to parsing dataset_id "dsXXXX:sub-YYY:..."
        if "subject_id" not in h5.attrs:
            try:
                parts = str(dataset_id).split(":")
                if len(parts) >= 2 and parts[1]:
                    h5.attrs["subject_id"] = _prepare_attr_value(parts[1])
            except Exception:
                pass

        _create_dataset(h5, "time", payload.time)
        _create_dataset(h5, "mnps_3d", payload.x)
        _create_dataset(h5, "mnps_3d_dot", payload.x_dot)

        _write_feature_surface_group(
            h5,
            group_name="features_raw",
            values=getattr(payload, "features_raw_values", None),
            names=getattr(payload, "features_raw_names", None),
            feature_metadata=getattr(payload, "feature_metadata", None),
            export_transform="none",
        )
        _write_feature_surface_group(
            h5,
            group_name="features_robust_z",
            values=getattr(payload, "features_robust_z_values", None),
            names=getattr(payload, "features_robust_z_names", None),
            feature_metadata=getattr(payload, "feature_metadata", None),
            export_transform="strict_robust_z",
        )

        if payload.z is not None and payload.z.size:
            _create_dataset(h5, "z", payload.z)

        if payload.stage is not None and payload.stage.size:
            labels_grp = h5.require_group("labels")
            _create_dataset(labels_grp, "stage", payload.stage)
            # Add staging metadata for downstream tools (viz/splitters)
            labels_grp.attrs["alignment"] = "per_timepoint"
            if "stage_source" in payload.attrs:
                labels_grp.attrs["stage_source"] = _prepare_attr_value(payload.attrs["stage_source"])
            if "stage_column" in payload.attrs:
                labels_grp.attrs["stage_column"] = _prepare_attr_value(payload.attrs["stage_column"])
            if "stage_codebook" in payload.attrs:
                labels_grp.attrs["stage_codebook"] = json.dumps(payload.attrs["stage_codebook"])

        if payload.labels:
            labels_grp = h5.require_group("labels")
            for name, arr in payload.labels.items():
                _create_dataset(labels_grp, name, arr)

        if payload.window_start is not None:
            _create_dataset(h5, "window_start", payload.window_start)
        if payload.window_end is not None:
            _create_dataset(h5, "window_end", payload.window_end)

        if payload.events:
            events_grp = h5.require_group("events")
            for name, arr in payload.events.items():
                _create_dataset(events_grp, name, arr)

        if payload.nn_indices is not None and payload.nn_indices.size:
            nn_grp = h5.require_group("nn")
            _create_dataset(nn_grp, "indices", payload.nn_indices)

        jacobian_grp = h5.require_group("jacobian")
        if payload.jacobian is not None and payload.jacobian.size:
            _create_dataset(jacobian_grp, "J_hat", payload.jacobian)
        if payload.jacobian_dot is not None and payload.jacobian_dot.size:
            _create_dataset(jacobian_grp, "J_dot", payload.jacobian_dot)
        if payload.jacobian_centers is not None and payload.jacobian_centers.size:
            _create_dataset(jacobian_grp, "centers", payload.jacobian_centers)
        if isinstance(jacobian_diagnostics, Mapping) and jacobian_diagnostics:
            diag_grp = jacobian_grp.require_group("diagnostics")
            for diag_key, diag_value in jacobian_diagnostics.items():
                safe_key = _sanitize_h5_key(str(diag_key))
                if isinstance(diag_value, np.ndarray):
                    _create_dataset(diag_grp, safe_key, diag_value)
                    continue
                if isinstance(diag_value, (list, tuple)):
                    _create_dataset(diag_grp, safe_key, np.asarray(diag_value))
                    continue
                if isinstance(diag_value, (bool, int, float, np.integer, np.floating, np.bool_)):
                    if isinstance(diag_value, (float, np.floating)) and not np.isfinite(float(diag_value)):
                        diag_grp.attrs[safe_key] = np.nan
                    else:
                        diag_grp.attrs[safe_key] = diag_value
                    continue
                if isinstance(diag_value, (str, bytes)):
                    diag_grp.attrs[safe_key] = _prepare_attr_value(diag_value)

        # Optional Stratified MNPS Jacobians (v2)
        if getattr(payload, "jacobian_9D", None) is not None and payload.jacobian_9D.size:
            jac_v2_grp = h5.require_group("jacobian_9D")
            _create_dataset(jac_v2_grp, "J_hat", payload.jacobian_9D)
            if getattr(payload, "jacobian_9D_dot", None) is not None and payload.jacobian_9D_dot.size:
                _create_dataset(jac_v2_grp, "J_dot", payload.jacobian_9D_dot)
            if getattr(payload, "jacobian_9D_centers", None) is not None and payload.jacobian_9D_centers.size:
                _create_dataset(jac_v2_grp, "centers", payload.jacobian_9D_centers)
            # Optional cross-partials extracted from the v2 Jacobian, stored as [W2] series
            cross = getattr(payload, "jacobian_9D_cross_partials", None)
            if isinstance(cross, Mapping) and cross:
                cp_grp = jac_v2_grp.require_group("cross_partials")
                for name, arr in cross.items():
                    # Avoid nested paths; use a flat dataset name
                    safe = str(name).replace("/", "_")
                    _create_dataset(cp_grp, safe, np.asarray(arr, dtype=np.float32))

        # Optional stratified MNPS coordinates (typically 9D)
        coords_9d = getattr(payload, "coords_9d", None)
        if coords_9d is not None and np.size(coords_9d) > 0:
            coords_9d_grp = h5.require_group("coords_9d")
            _create_dataset(coords_9d_grp, "values", coords_9d)
            names = getattr(payload, "coords_9d_names", None)
            if names:
                str_dtype = h5py.string_dtype(encoding="utf-8")
                names_utf8 = np.asarray([str(v) for v in names], dtype=object)
                coords_9d_grp.create_dataset("names", data=names_utf8, dtype=str_dtype)
            coords_9d_grp.attrs["version"] = "9d"

        # Optional extended coordinates (E-Kappa, RFM, O-Koh, TIG, ...)
        extensions = getattr(payload, "extensions", None)
        if isinstance(extensions, Mapping) and extensions:
            _write_extensions_group(h5, extensions)

        # Optional raw regional signals (e.g. fMRI ROI×time) used as
        # supporting inputs for some downstream analyses. These live under
        # /regions, but the canonical modality-agnostic regional output lives
        # under /regional_mnps below.
        regions_bold = getattr(payload, "regions_bold", None)
        if regions_bold is not None and np.size(regions_bold) > 0:
            regions_grp = h5.require_group("regions")
            _create_dataset(regions_grp, "bold", regions_bold)

            regions_names = getattr(payload, "regions_names", None)
            if regions_names:
                str_dtype = h5py.string_dtype(encoding="utf-8")
                names_utf8 = np.asarray([str(v) for v in regions_names], dtype=object)
                regions_grp.create_dataset(
                    "names",
                    data=names_utf8,
                    dtype=str_dtype,
                )

            regions_sfreq = getattr(payload, "regions_sfreq", None)
            if regions_sfreq is not None:
                regions_grp.attrs["sfreq"] = float(regions_sfreq)

        # Canonical modality-agnostic regional MNPS/MNJ outputs per network
        regional_mnps = getattr(payload, "regional_mnps", None)
        if isinstance(regional_mnps, Mapping) and regional_mnps:
            reg_mnps_grp = h5.require_group("regional_mnps")
            for network_label, network_data in regional_mnps.items():
                net_grp = reg_mnps_grp.require_group(network_label)
                if isinstance(network_data, Mapping):
                    # Write MNPS coordinates
                    mnps = network_data.get("mnps")
                    if mnps is not None and np.size(mnps) > 0:
                        _create_dataset(net_grp, "mnps", np.asarray(mnps, dtype=np.float32))
                    # Write MNPS derivatives
                    mnps_dot = network_data.get("mnps_dot")
                    if mnps_dot is not None and np.size(mnps_dot) > 0:
                        _create_dataset(net_grp, "mnps_dot", np.asarray(mnps_dot, dtype=np.float32))
                    # Write Jacobian
                    jacobian = network_data.get("jacobian")
                    if jacobian is not None and np.size(jacobian) > 0:
                        _create_dataset(net_grp, "jacobian", np.asarray(jacobian, dtype=np.float32))
                    # Write metrics as attributes
                    metrics = network_data.get("metrics")
                    if isinstance(metrics, Mapping):
                        for metric_name, metric_val in metrics.items():
                            if metric_val is not None:
                                try:
                                    metric_float = float(metric_val)
                                    # Preserve inf (e.g. strat9_condition_number) as
                                    # a very large sentinel so the physics signal is
                                    # not silently destroyed.  NaN stays NaN.
                                    if np.isnan(metric_float):
                                        net_grp.attrs[metric_name] = np.nan
                                    elif not np.isfinite(metric_float):
                                        # HDF5 h5py supports float inf natively.
                                        net_grp.attrs[metric_name] = metric_float
                                    else:
                                        net_grp.attrs[metric_name] = metric_float
                                except Exception:
                                    net_grp.attrs[metric_name] = _prepare_attr_value(metric_val)
                    # Write n_timepoints
                    n_tp = network_data.get("n_timepoints")
                    if n_tp is not None:
                        net_grp.attrs["n_timepoints"] = int(n_tp)
                    # Write 9-D stratified trajectory as a proper dataset when
                    # present (enables downstream trajectory analysis).
                    stratified = network_data.get("stratified")
                    if stratified is not None and np.size(stratified) > 0:
                        _create_dataset(
                            net_grp,
                            "stratified",
                            np.asarray(stratified, dtype=np.float32),
                        )

    logger.info("Wrote MNPS HDF5: %s", out_path)
    return out_path


