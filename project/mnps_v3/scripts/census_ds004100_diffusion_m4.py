"""Census ds004100 dynamical-family HDF5 against the v3 D-M4 prereg.

Read-only. Does not interpret ictal/interictal biology.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def _decode(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            item = value.item()
            if isinstance(item, bytes):
                return item.decode("utf-8")
            return None if item is None else str(item)
        return None
    return str(value)


def _read_scalar(group, path: str) -> str | None:
    if path not in group:
        return None
    return _decode(group[path][()])


def _all_nan(group, path: str) -> str:
    if path not in group:
        return "missing"
    data = np.asarray(group[path][:], dtype=float)
    if data.size == 0:
        return "empty"
    finite = np.isfinite(data)
    if not np.any(finite):
        return "all_nan"
    if np.all(data[finite] == 0.0):
        return "all_zero"
    return "has_finite"


def census_file(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        canonical = "dynamical_families" in handle
        legacy = "orthogonal_dynamics" in handle
        if canonical:
            root = handle["dynamical_families"]
            diffusion = root["diffusion/v1"] if "diffusion/v1" in root else None
            destination = root["destination/v1"] if "destination/v1" in root else None
            resilience = root["resilience/v1"] if "resilience/v1" in root else None
            namespace = "canonical"
        elif legacy:
            root = handle["orthogonal_dynamics"]
            diffusion = (
                root["diffusion_geometry/v1"] if "diffusion_geometry/v1" in root else None
            )
            destination = root["committor/v1"] if "committor/v1" in root else None
            resilience = (
                root["finite_amplitude_resilience/v1"]
                if "finite_amplitude_resilience/v1" in root
                else None
            )
            namespace = "legacy_orthogonal_dynamics"
        else:
            return {"file": str(path), "namespace": "absent"}

        def family_row(group) -> dict:
            if group is None:
                return {"present": False}
            row = {
                "present": True,
                "computation_status": _read_scalar(group, "computation_status"),
                "measurement_validity": _read_scalar(group, "measurement_validity"),
                "claim_status": _read_scalar(group, "claim_status"),
                "failure_reason": _read_scalar(group, "failure_reason"),
                "grain_biological_unit": _read_scalar(group, "grain/biological_unit"),
                "grain_direct_between_subject": _read_scalar(
                    group, "grain/direct_between_subject_inference"
                ),
                "qualification_id": _read_scalar(group, "provenance/qualification_id"),
                "validation_level": _read_scalar(group, "provenance/validation_level"),
            }
            return row

        diffusion_row = family_row(diffusion)
        if diffusion is not None:
            diffusion_row["A_bD_computation_status"] = _read_scalar(
                diffusion, "summary/A_bD_computation_status"
            )
            diffusion_row["R_b_over_a_computation_status"] = _read_scalar(
                diffusion, "summary/R_b_over_a_computation_status"
            )
            diffusion_row["drift_alignment_failure_reason"] = _read_scalar(
                diffusion, "summary/drift_alignment_failure_reason"
            )
            diffusion_row["A_bD_series"] = _all_nan(diffusion, "series/A_bD")
            diffusion_row["R_b_over_a_series"] = _all_nan(diffusion, "series/R_b_over_a")
            if "series/a_hat" in diffusion:
                diffusion_row["a_hat_shape"] = list(diffusion["series/a_hat"].shape)

        return {
            "file": path.name,
            "namespace": namespace,
            "mnps_3d": "mnps_3d" in handle,
            "jacobian": "jacobian" in handle and "J_hat" in handle["jacobian"],
            "support_signature": "support_signature" in handle,
            "stochastic_reachability": "stochastic_reachability" in handle,
            "diffusion": diffusion_row,
            "destination": family_row(destination),
            "resilience": family_row(resilience),
        }


def summarize(rows: list[dict]) -> dict:
    def count(path: tuple[str, ...]) -> Counter:
        values: Counter = Counter()
        for row in rows:
            node: object = row
            for key in path:
                if not isinstance(node, dict):
                    node = None
                    break
                node = node.get(key)
            values[str(node)] += 1
        return values

    return {
        "n_files": len(rows),
        "namespace": dict(count(("namespace",))),
        "mnps_3d": dict(count(("mnps_3d",))),
        "jacobian": dict(count(("jacobian",))),
        "support_signature": dict(count(("support_signature",))),
        "stochastic_reachability": dict(count(("stochastic_reachability",))),
        "diffusion_computation_status": dict(count(("diffusion", "computation_status"))),
        "diffusion_measurement_validity": dict(
            count(("diffusion", "measurement_validity"))
        ),
        "diffusion_claim_status": dict(count(("diffusion", "claim_status"))),
        "diffusion_A_bD_status": dict(count(("diffusion", "A_bD_computation_status"))),
        "diffusion_R_b_over_a_status": dict(
            count(("diffusion", "R_b_over_a_computation_status"))
        ),
        "diffusion_alignment_reason": dict(
            count(("diffusion", "drift_alignment_failure_reason"))
        ),
        "diffusion_A_bD_series": dict(count(("diffusion", "A_bD_series"))),
        "diffusion_R_b_over_a_series": dict(count(("diffusion", "R_b_over_a_series"))),
        "destination_computation_status": dict(
            count(("destination", "computation_status"))
        ),
        "destination_failure_reason": dict(count(("destination", "failure_reason"))),
        "resilience_computation_status": dict(
            count(("resilience", "computation_status"))
        ),
        "resilience_failure_reason": dict(count(("resilience", "failure_reason"))),
        "grain_biological_unit": dict(count(("diffusion", "grain_biological_unit"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    files = sorted(args.run_dir.rglob("*.h5"))
    rows = [census_file(path) for path in files]
    report = {"run_dir": str(args.run_dir), "summary": summarize(rows), "files": rows}
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
