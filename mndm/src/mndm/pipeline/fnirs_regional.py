"""Geometry-derived fNIRS scalp-sector maps and descriptive window summaries.

This is a sensor-space pilot helper. It does not infer cortical regions, QC,
measurement validity, qualification, or MNPS coordinates.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import os
import uuid
from typing import Any, Mapping
from urllib.parse import quote
from pathlib import Path
from importlib import metadata as importlib_metadata

import numpy as np
import pandas as pd
from .. import __version__ as _MNDM_VERSION


REGIONAL_CONTRACT = "mndm.fnirs_sensor_sector.v0.1"
_CHROMOPHORES = ("hbo", "hbr")
_SUPPORT_TOLERANCE_REL = 1e-6
_SECTORS = (
    "left_anterior_scalp",
    "left_posterior_scalp",
    "right_anterior_scalp",
    "right_posterior_scalp",
)
_CHANNEL_RE = re.compile(r"^(?P<pair>[A-Za-z0-9]+_[A-Za-z0-9]+) (?P<chromophore>hbo|hbr)$", re.IGNORECASE)


class FnirsRegionalContractError(ValueError):
    """Input geometry or feature table does not satisfy the regional contract."""


def _validate_coordinatesystem(coordsystem: Mapping[str, Any]) -> None:
    if not isinstance(coordsystem, Mapping):
        raise FnirsRegionalContractError("NIRS coordinate-system metadata must be a JSON object.")
    if str(coordsystem.get("NIRSCoordinateSystem", "")).casefold() != "captrak":
        raise FnirsRegionalContractError("Sensor-sector mapping requires declared CapTrak coordinates.")
    if str(coordsystem.get("NIRSCoordinateUnits", "")).casefold() != "m":
        raise FnirsRegionalContractError("Sensor-sector mapping requires coordinates declared in metres.")
    description = str(coordsystem.get("NIRSCoordinateSystemDescription", "")).casefold()
    if not all(token in description for token in ("lpa", "rpa", "nasion")):
        raise FnirsRegionalContractError("CapTrak axes must document LPA, RPA and nasion orientation.")
    if "ras" not in description or description.find("lpa") > description.find("rpa"):
        raise FnirsRegionalContractError("CapTrak axis direction must agree with LPA-to-RPA RAS orientation.")
    if re.search(r"(?:positive|increasing|toward(?:s)?\s+the)\s+(?:y|posterior)|posterior\s+(?:positive|increasing)\s+y", description):
        raise FnirsRegionalContractError("CapTrak y-axis direction contradicts positive-anterior RAS orientation.")
    if re.search(r"(?:positive|increasing|toward(?:s)?\s+the)\s+(?:x|left)|left\s+(?:positive|increasing)\s+x", description):
        raise FnirsRegionalContractError("CapTrak x-axis direction contradicts positive-right RAS orientation.")


def _text_column(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        raise FnirsRegionalContractError(f"Required geometry column {column!r} is absent.")
    values = frame[column]
    if values.isna().any():
        raise FnirsRegionalContractError(f"Geometry column {column!r} contains missing values.")
    return [str(value).strip() for value in values]


def derive_fnirs_scalp_sector_map(
    channels: pd.DataFrame,
    optodes: pd.DataFrame,
    coordsystem: Mapping[str, Any],
    channel_map: list[Mapping[str, Any]],
    *,
    short_distance_mm: float = 10.0,
) -> pd.DataFrame:
    """Build a frozen, per-Hb-channel map to four CapTrak scalp sectors.

    Pairs shorter than ``short_distance_mm`` are marked short and excluded.
    The returned labels are geometric scalp sectors, never cortical labels.
    """
    _validate_coordinatesystem(coordsystem)
    if not math.isfinite(float(short_distance_mm)) or short_distance_mm <= 0:
        raise FnirsRegionalContractError("short_distance_mm must be finite and positive.")
    if float(short_distance_mm) != 10.0:
        raise FnirsRegionalContractError("G5 contract freezes the short-channel cutoff at 10.0 mm.")

    required_channel_columns = ("source", "detector", "wavelength_nominal")
    for column in required_channel_columns:
        if column not in channels.columns:
            raise FnirsRegionalContractError(f"channels.tsv is missing {column!r}.")
    source_labels = _text_column(channels, "source")
    detector_labels = _text_column(channels, "detector")
    try:
        wavelength = pd.to_numeric(channels["wavelength_nominal"], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise FnirsRegionalContractError("wavelength_nominal must be finite and numeric.") from exc
    if not np.isfinite(wavelength).all():
        raise FnirsRegionalContractError("wavelength_nominal must be finite and numeric.")
    sidecar_rows = list(zip(source_labels, detector_labels, wavelength))
    if len(set(sidecar_rows)) != len(sidecar_rows):
        raise FnirsRegionalContractError("channels.tsv contains duplicate source/detector/wavelength rows.")
    pair_keys = sorted({(source, detector) for source, detector in zip(source_labels, detector_labels)})

    for column in ("name", "type", "x", "y", "z"):
        if column not in optodes.columns:
            raise FnirsRegionalContractError(f"optodes.tsv is missing {column!r}.")
    names = _text_column(optodes, "name")
    types = _text_column(optodes, "type")
    if len(set(names)) != len(names):
        raise FnirsRegionalContractError("optode labels must be unique.")
    positions: dict[str, np.ndarray] = {}
    optode_types: dict[str, str] = {}
    for index, name in enumerate(names):
        try:
            point = np.asarray([optodes.iloc[index][axis] for axis in ("x", "y", "z")], dtype=float)
        except (TypeError, ValueError) as exc:
            raise FnirsRegionalContractError(f"Optode {name!r} has invalid coordinates.") from exc
        if point.shape != (3,) or not np.isfinite(point).all():
            raise FnirsRegionalContractError(f"Optode {name!r} coordinates must be finite 3D values.")
        positions[name] = point
        optode_types[name] = types[index].casefold()

    if not isinstance(channel_map, list) or not channel_map:
        raise FnirsRegionalContractError("The G3 channel map must be a nonempty JSON array.")
    feature_channels: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    seen_channel_names: set[str] = set()
    seen_column_keys: set[str] = set()
    for item in channel_map:
        if not isinstance(item, Mapping):
            raise FnirsRegionalContractError("Every G3 channel-map entry must be an object.")
        name = str(item.get("channel_name", ""))
        chromophore = str(item.get("chromophore", "")).casefold()
        match = _CHANNEL_RE.fullmatch(name)
        if not match or match.group("chromophore").casefold() != chromophore or chromophore not in _CHROMOPHORES:
            raise FnirsRegionalContractError(f"Unsupported semantic Hb channel name {name!r}.")
        pair = match.group("pair")
        if str(item.get("unit", "")) != "M" or str(item.get("column_key", "")) != quote(name, safe=""):
            raise FnirsRegionalContractError(f"Channel-map unit or key is invalid for {name!r}.")
        source, detector = pair.split("_", maxsplit=1)
        column_key = str(item["column_key"])
        pair_channels = feature_channels.setdefault((source, detector), {})
        if name in seen_channel_names or column_key in seen_column_keys or chromophore in pair_channels:
            raise FnirsRegionalContractError(f"Duplicate G3 channel-map entry for {name!r}.")
        seen_channel_names.add(name)
        seen_column_keys.add(column_key)
        pair_channels[chromophore] = item

    sidecar_pair_set = set(pair_keys)
    if set(feature_channels) != sidecar_pair_set:
        raise FnirsRegionalContractError("G3 channel pairs do not exactly match channels.tsv source/detector pairs.")

    records: list[dict[str, Any]] = []
    for source, detector in pair_keys:
        if source not in positions or detector not in positions:
            raise FnirsRegionalContractError(f"Pair {source}/{detector} references an undeclared optode.")
        if optode_types[source] != "source" or optode_types[detector] != "detector":
            raise FnirsRegionalContractError(f"Pair {source}/{detector} has inconsistent optode types.")
        if set(feature_channels[(source, detector)]) != set(_CHROMOPHORES):
            raise FnirsRegionalContractError(f"Pair {source}/{detector} needs exactly one HbO and one HbR feature channel.")
        source_xyz = positions[source]
        detector_xyz = positions[detector]
        distance_mm = float(np.linalg.norm(source_xyz - detector_xyz) * 1000.0)
        midpoint_mm = (source_xyz + detector_xyz) * 500.0
        if not np.isfinite(distance_mm) or distance_mm <= 0:
            raise FnirsRegionalContractError(f"Pair {source}/{detector} has invalid source-detector distance.")
        if distance_mm < short_distance_mm:
            distance_class = "short_excluded"
            sector = None
        else:
            side = "left" if midpoint_mm[0] < 0 else "right"
            ap = "anterior" if midpoint_mm[1] >= 0 else "posterior"
            sector = f"{side}_{ap}_scalp"
            distance_class = "long_included"
        for chromophore in _CHROMOPHORES:
            channel = feature_channels[(source, detector)][chromophore]
            records.append({
                "source": source,
                "detector": detector,
                "channel_name": str(channel["channel_name"]),
                "chromophore": chromophore,
                "column_key": str(channel["column_key"]),
                "unit": "M",
                "distance_mm": distance_mm,
                "distance_class": distance_class,
                "midpoint_x_mm": float(midpoint_mm[0]),
                "midpoint_y_mm": float(midpoint_mm[1]),
                "midpoint_z_mm": float(midpoint_mm[2]),
                "coordinate_system": "CapTrak",
                "region_name": sector,
                "region_kind": "geometric_scalp_sector" if sector else "excluded_short_channel",
                "map_contract": REGIONAL_CONTRACT,
            })

    result = pd.DataFrame.from_records(records)
    included = result[result["distance_class"] == "long_included"]
    counts = included.groupby(["region_name", "chromophore"])["source"].size()
    if set(included["region_name"].unique()) != set(_SECTORS):
        raise FnirsRegionalContractError("The fixed map does not cover all four declared scalp sectors.")
    if not all(int(counts.get((sector, chrom), 0)) > 0 for sector in _SECTORS for chrom in _CHROMOPHORES):
        raise FnirsRegionalContractError("Every scalp sector must contain long HbO and HbR channels.")
    return result.sort_values(["region_name", "source", "detector", "chromophore"], kind="stable").reset_index(drop=True)


def _feature_channel_map(features: pd.DataFrame) -> list[Mapping[str, Any]]:
    column = "fnirs_channel_map_json"
    if column not in features.columns or features.empty:
        raise FnirsRegionalContractError("Features need a nonempty fnirs_channel_map_json table.")
    values = features[column].dropna().astype(str).unique()
    if len(values) != 1:
        raise FnirsRegionalContractError("Every window must carry the same single G3 channel map.")
    try:
        parsed = json.loads(values[0])
    except (TypeError, json.JSONDecodeError) as exc:
        raise FnirsRegionalContractError("fnirs_channel_map_json is invalid JSON.") from exc
    if not isinstance(parsed, list):
        raise FnirsRegionalContractError("fnirs_channel_map_json must be an array.")
    names = [str(item.get("channel_name", "")) for item in parsed if isinstance(item, Mapping)]
    keys = [str(item.get("column_key", "")) for item in parsed if isinstance(item, Mapping)]
    if len(names) != len(parsed) or len(set(names)) != len(names) or len(set(keys)) != len(keys):
        raise FnirsRegionalContractError("G3 channel map has malformed or duplicate channel names/column keys.")
    return parsed


def _validate_features(features: pd.DataFrame) -> list[Mapping[str, Any]]:
    required = (
        "file", "epoch_id", "t_start", "t_end", "fnirs_window_id",
        "fnirs_window_start_s", "fnirs_window_end_s", "fnirs_actual_t_start_s",
        "fnirs_actual_t_end_s", "fnirs_actual_covered_duration_s",
        "fnirs_clock_support_fraction", "fnirs_min_valid_support_fraction",
        "fnirs_source_sha256", "fnirs_measurement_validity",
        "fnirs_qualification_status", "fnirs_qc_status", "qc_ok_fnirs",
    )
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise FnirsRegionalContractError("G3 feature table lacks required fields: " + ", ".join(missing))
    if features.empty or features["file"].isna().any() or features["fnirs_source_sha256"].isna().any():
        raise FnirsRegionalContractError("Regional pilot input must include nonempty source identity and checksum values.")
    source_files = features["file"].astype(str).str.strip()
    source_hashes = features["fnirs_source_sha256"].astype(str).str.strip().str.lower()
    if (source_files.eq("").any() or source_files.nunique() != 1
            or source_hashes.nunique() != 1
            or not re.fullmatch(r"[0-9a-f]{64}", source_hashes.iloc[0])):
        raise FnirsRegionalContractError("Regional pilot input must identify one source recording and one checksum.")
    if features["fnirs_window_id"].duplicated().any():
        raise FnirsRegionalContractError("G3 window IDs must be unique before regional aggregation.")
    if set(features["fnirs_measurement_validity"].dropna().astype(str)) != {"not_assessed"}:
        raise FnirsRegionalContractError("Regional pilot may not upgrade measurement validity.")
    if set(features["fnirs_qualification_status"].dropna().astype(str)) != {"not_assessed"}:
        raise FnirsRegionalContractError("Regional pilot may not upgrade qualification status.")
    if set(features["fnirs_qc_status"].dropna().astype(str)) != {"not_assessed"}:
        raise FnirsRegionalContractError("Regional pilot may not upgrade QC status.")
    if features["qc_ok_fnirs"].notna().any():
        raise FnirsRegionalContractError("Unknown fNIRS QC must remain unknown in the regional pilot.")
    return _feature_channel_map(features)


def _validate_region_map(region_map: pd.DataFrame) -> None:
    required = {
        "source", "detector", "channel_name", "chromophore", "distance_mm",
        "distance_class", "midpoint_x_mm", "midpoint_y_mm", "coordinate_system",
        "region_name", "region_kind", "map_contract",
    }
    missing = required.difference(region_map.columns)
    if missing or region_map.empty:
        raise FnirsRegionalContractError("Regional map lacks required geometry fields: " + ", ".join(sorted(missing)))
    if region_map["channel_name"].duplicated().any():
        raise FnirsRegionalContractError("Regional map contains duplicate semantic channels.")
    if set(region_map["map_contract"].astype(str)) != {REGIONAL_CONTRACT}:
        raise FnirsRegionalContractError("Regional map has an unsupported map contract.")
    if set(region_map["coordinate_system"].astype(str)) != {"CapTrak"}:
        raise FnirsRegionalContractError("Regional map coordinate system must remain CapTrak.")
    if not set(region_map["chromophore"].astype(str)).issubset(set(_CHROMOPHORES)):
        raise FnirsRegionalContractError("Regional map contains an unsupported chromophore.")
    for row in region_map.to_dict(orient="records"):
        try:
            distance = float(row["distance_mm"])
            x_mm = float(row["midpoint_x_mm"])
            y_mm = float(row["midpoint_y_mm"])
        except (TypeError, ValueError) as exc:
            raise FnirsRegionalContractError("Regional map geometry values must be numeric.") from exc
        if not all(math.isfinite(value) for value in (distance, x_mm, y_mm)) or distance <= 0:
            raise FnirsRegionalContractError("Regional map geometry values must be finite and positive where required.")
        if distance < 10.0:
            expected_class, expected_region, expected_kind = "short_excluded", None, "excluded_short_channel"
        else:
            side = "left" if x_mm < 0 else "right"
            ap = "anterior" if y_mm >= 0 else "posterior"
            expected_class, expected_region, expected_kind = "long_included", f"{side}_{ap}_scalp", "geometric_scalp_sector"
        if (str(row["distance_class"]) != expected_class
                or row["region_name"] != expected_region
                or str(row["region_kind"]) != expected_kind):
            raise FnirsRegionalContractError("Regional map labels do not match the frozen geometry rule.")


def aggregate_fnirs_scalp_sectors(features: pd.DataFrame, region_map: pd.DataFrame) -> pd.DataFrame:
    """Create equal-channel-weighted sector means with fixed full coverage.

    No regional value is produced if any expected long channel is missing,
    rejected, below the window's minimum support or non-finite.
    """
    channel_map = _validate_features(features)
    _validate_region_map(region_map)
    included = region_map[region_map["distance_class"] == "long_included"].copy()
    if included.empty:
        raise FnirsRegionalContractError("Regional map has no long channels.")
    expected_names = {str(item.get("channel_name")) for item in channel_map}
    if not set(included["channel_name"].astype(str)).issubset(expected_names):
        raise FnirsRegionalContractError("Regional map contains a channel absent from the feature-table channel map.")

    keyed: dict[str, Mapping[str, Any]] = {str(item["channel_name"]): item for item in channel_map}
    for name in included["channel_name"].astype(str):
        item = keyed[name]
        column_key = str(item["column_key"])
        for metric in ("mean_M", "valid_support_fraction", "rejection_reason"):
            column = f"fnirs_{metric}__{column_key}"
            if column not in features.columns:
                raise FnirsRegionalContractError(f"Feature channel {name!r} lacks {metric!r}.")

    output: list[dict[str, Any]] = []
    for _, row in features.iterrows():
        try:
            minimum_support = float(row["fnirs_min_valid_support_fraction"])
            if not math.isfinite(minimum_support) or not 0 < minimum_support <= 1:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise FnirsRegionalContractError("Every window needs a valid minimum channel support threshold.") from exc
        for sector in _SECTORS:
            for chromophore in _CHROMOPHORES:
                members = included[
                    (included["region_name"] == sector) & (included["chromophore"] == chromophore)
                ]
                valid_values: list[float] = []
                for channel in members["channel_name"].astype(str):
                    key = str(keyed[channel]["column_key"])
                    try:
                        value = float(row[f"fnirs_mean_M__{key}"])
                    except (TypeError, ValueError):
                        value = np.nan
                    try:
                        support = float(row[f"fnirs_valid_support_fraction__{key}"])
                    except (TypeError, ValueError) as exc:
                        raise FnirsRegionalContractError(f"Channel {channel!r} has invalid valid-support metadata.") from exc
                    if not math.isfinite(support) or support < 0.0 or support > 1.0:
                        raise FnirsRegionalContractError(f"Channel {channel!r} valid-support fraction must be in [0, 1].")
                    if not math.isfinite(value):
                        continue
                    reason = row[f"fnirs_rejection_reason__{key}"]
                    no_rejection = pd.isna(reason) or str(reason).strip() == ""
                    support_tolerance = max(minimum_support * _SUPPORT_TOLERANCE_REL, 1e-12)
                    if math.isfinite(value) and math.isfinite(support) and support >= minimum_support - support_tolerance and no_rejection:
                        valid_values.append(value)
                expected = int(len(members))
                valid_count = len(valid_values)
                complete = expected > 0 and valid_count == expected
                output.append({
                    "file": str(row["file"]),
                    "source_sha256": str(row["fnirs_source_sha256"]),
                    "epoch_id": int(row["epoch_id"]),
                    "window_id": int(row["fnirs_window_id"]),
                    "t_start_s": float(row["fnirs_window_start_s"]),
                    "t_end_s": float(row["fnirs_window_end_s"]),
                    "actual_t_start_s": row["fnirs_actual_t_start_s"],
                    "actual_t_end_s": row["fnirs_actual_t_end_s"],
                    "actual_covered_duration_s": float(row["fnirs_actual_covered_duration_s"]),
                    "clock_support_fraction": float(row["fnirs_clock_support_fraction"]),
                    "min_valid_support_fraction": minimum_support,
                    "region_name": sector,
                    "region_kind": "geometric_scalp_sector",
                    "chromophore": chromophore,
                    "region_mean_M": float(np.mean(valid_values)) if complete else np.nan,
                    "channel_count_expected": expected,
                    "channel_count_valid": valid_count,
                    "channel_coverage_fraction": valid_count / expected if expected else 0.0,
                    "region_status": "computed" if complete else "incomplete_channel_coverage",
                    "unit": "M",
                    "qc_ok_fnirs": pd.NA,
                    "qc_status": "not_assessed",
                    "measurement_validity": "not_assessed",
                    "qualification_status": "not_assessed",
                    "is_qualified": 0,
                    "claim_status": "no_biological_claim",
                    "regional_contract": REGIONAL_CONTRACT,
                })
    result = pd.DataFrame.from_records(output)
    result["qc_ok_fnirs"] = pd.array(result["qc_ok_fnirs"], dtype="Int8")
    return result


def compute_channel_loss_sensitivity(features: pd.DataFrame, region_map: pd.DataFrame) -> pd.DataFrame:
    """Report leave-one-source/detector-pair-out changes for each sector."""
    baseline = aggregate_fnirs_scalp_sectors(features, region_map)
    included = region_map[region_map["distance_class"] == "long_included"]
    records: list[dict[str, Any]] = []
    pairs = included[["source", "detector"]].drop_duplicates().sort_values(["source", "detector"])
    for source, detector in pairs.itertuples(index=False, name=None):
        omitted_map = region_map[
            ~((region_map["source"] == source) & (region_map["detector"] == detector))
        ]
        perturbed = aggregate_fnirs_scalp_sectors(features, omitted_map)
        joined = baseline.merge(
            perturbed,
            on=["window_id", "region_name", "chromophore"],
            suffixes=("_baseline", "_perturbed"),
            validate="one_to_one",
        )
        for (sector, chromophore), group in joined.groupby(["region_name", "chromophore"], sort=True):
            left = pd.to_numeric(group["region_mean_M_baseline"], errors="coerce").to_numpy(float)
            right = pd.to_numeric(group["region_mean_M_perturbed"], errors="coerce").to_numpy(float)
            valid = np.isfinite(left) & np.isfinite(right)
            delta = np.abs(right[valid] - left[valid])
            records.append({
                "source": source,
                "detector": detector,
                "omitted_pair": f"{source}_{detector}",
                "region_name": sector,
                "chromophore": chromophore,
                "windows_compared": int(valid.sum()),
                "max_abs_change_M": float(np.max(delta)) if delta.size else np.nan,
                "median_abs_change_M": float(np.median(delta)) if delta.size else np.nan,
                "sensitivity_status": "computed" if delta.size else "no_complete_windows",
                "contract": REGIONAL_CONTRACT,
            })
    return pd.DataFrame.from_records(records)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_fnirs_scalp_sector_pilot(
    features_path: str | Path,
    channels_tsv: str | Path,
    optodes_tsv: str | Path,
    coordsystem_json: str | Path,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export the G5 sensor-sector pilot artifacts and reproducibility manifest."""
    features_path = Path(features_path)
    channels_tsv = Path(channels_tsv)
    optodes_tsv = Path(optodes_tsv)
    coordsystem_json = Path(coordsystem_json)
    out_dir = Path(out_dir)
    inputs = {
        "features_table": features_path,
        "channels_tsv": channels_tsv,
        "optodes_tsv": optodes_tsv,
        "coordsystem_json": coordsystem_json,
    }
    absent = [str(path) for path in inputs.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError("G5 input file(s) not found: " + ", ".join(absent))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "regional_features": out_dir / "regional_features.csv",
        "channel_region_map": out_dir / "channel_region_map.json",
        "channel_loss_sensitivity": out_dir / "channel_loss_sensitivity.csv",
        "manifest": out_dir / "g5_manifest.json",
    }
    if not overwrite:
        collisions = [str(path) for path in output_paths.values() if path.exists()]
        if collisions:
            raise FileExistsError("G5 output already exists; choose a new output directory or use overwrite: " + ", ".join(collisions))

    from ..parallel import _read_feature_table

    features = _read_feature_table(features_path)
    channel_map = _validate_features(features)
    channels = pd.read_csv(channels_tsv, sep="\t", encoding="utf-8-sig")
    optodes = pd.read_csv(optodes_tsv, sep="\t", encoding="utf-8-sig")
    with coordsystem_json.open("r", encoding="utf-8-sig") as handle:
        coordsystem = json.load(handle)
    region_map = derive_fnirs_scalp_sector_map(channels, optodes, coordsystem, channel_map)
    regional = aggregate_fnirs_scalp_sectors(features, region_map)
    sensitivity = compute_channel_loss_sensitivity(features, region_map)
    module_path = Path(__file__).resolve()
    contract_path = module_path.parents[4] / "project" / "fNIRS" / "g5_contract.md"

    temp_paths: dict[str, Path] = {}
    backup_paths: dict[str, Path] = {}
    committed: list[str] = []
    try:
        for key, destination in output_paths.items():
            temp = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
            temp_paths[key] = temp
        regional.to_csv(temp_paths["regional_features"], index=False, float_format="%.17g")
        region_json = {
            "contract": REGIONAL_CONTRACT,
            "coordinate_system": coordsystem,
            "rule": {
                "pair_location": "Euclidean source-detector midpoint",
                "left_right": "midpoint_x_mm < 0 is left; otherwise right",
                "anterior_posterior": "midpoint_y_mm >= 0 is anterior; otherwise posterior",
                "short_pair_cutoff_mm": 10.0,
                "short_pair_policy": "exclude from regional candidate series",
                "labels_are": "geometric scalp sectors, not anatomical/cortical regions",
            },
            "channel_map": region_map.to_dict(orient="records"),
        }
        temp_paths["channel_region_map"].write_text(
            json.dumps(region_json, sort_keys=True, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        sensitivity.to_csv(temp_paths["channel_loss_sensitivity"], index=False, float_format="%.17g")
        manifest: dict[str, Any] = {
            "schema": REGIONAL_CONTRACT,
            "measurement_validity": "not_assessed",
            "qualification_status": "not_assessed",
            "qc_status": "not_assessed",
            "claim_status": "no_biological_claim",
            "source_sha256": str(features["fnirs_source_sha256"].iloc[0]),
            "inputs": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in inputs.items()},
            "implementation": {
                "module_path": str(module_path),
                "module_sha256": _sha256(module_path),
                "contract_path": str(contract_path),
                "contract_sha256": _sha256(contract_path) if contract_path.is_file() else None,
            },
            "coordinate_system": str(coordsystem.get("NIRSCoordinateSystem")),
            "map_contract": "g5_contract.md",
            "counts": {
                "windows": int(features["fnirs_window_id"].nunique()),
                "mapped_channel_chromophore_rows": int(len(region_map)),
                "long_channel_chromophore_rows": int((region_map["distance_class"] == "long_included").sum()),
                "short_channel_chromophore_rows": int((region_map["distance_class"] == "short_excluded").sum()),
                "regional_rows": int(len(regional)),
                "complete_regional_rows": int((regional["region_status"] == "computed").sum()),
                "channel_loss_sensitivity_rows": int(len(sensitivity)),
            },
            "regional_coverage": {
                f"{region}/{chromophore}": int((region_map["region_name"].eq(region) & region_map["chromophore"].eq(chromophore) & region_map["distance_class"].eq("long_included")).sum())
                for region in _SECTORS for chromophore in _CHROMOPHORES
            },
            "dependencies": {
                "mndm": _MNDM_VERSION,
                "numpy": importlib_metadata.version("numpy"),
                "pandas": importlib_metadata.version("pandas"),
            },
            "outputs": {},
            "interpretation_limit": "one-recording sensor-space descriptors only; no anatomy, QC, qualification, MNPS or biological claim",
        }
        manifest["outputs"] = {
            key: {"path": str(output_paths[key]), "sha256": _sha256(temp_paths[key])}
            for key in ("regional_features", "channel_region_map", "channel_loss_sensitivity")
        }
        temp_paths["manifest"].write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # All artifacts are staged before replacement. Backups allow rollback if
        # any member of the multi-file output set fails to commit.
        try:
            for key in ("regional_features", "channel_region_map", "channel_loss_sensitivity", "manifest"):
                destination = output_paths[key]
                if destination.exists():
                    if not overwrite:
                        raise FileExistsError(f"G5 output appeared during export: {destination}")
                    backup = destination.with_name(destination.name + f".{uuid.uuid4().hex}.bak")
                    os.replace(destination, backup)
                    backup_paths[key] = backup
                os.replace(temp_paths[key], destination)
                committed.append(key)
        except Exception as exc:
            for key in reversed(committed):
                output_paths[key].unlink(missing_ok=True)
            rollback_errors: list[str] = []
            for key, backup in backup_paths.items():
                if backup.exists():
                    try:
                        os.replace(backup, output_paths[key])
                    except OSError as rollback_exc:
                        rollback_errors.append(f"{backup} -> {output_paths[key]}: {rollback_exc}")
            if rollback_errors:
                raise RuntimeError("G5 export failed and backup restoration also failed; backups retained: " + "; ".join(rollback_errors)) from exc
            raise
        for backup in backup_paths.values():
            backup.unlink(missing_ok=True)
        return manifest
    finally:
        for temp in temp_paths.values():
            temp.unlink(missing_ok=True)

