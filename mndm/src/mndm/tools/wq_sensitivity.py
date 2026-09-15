"""Bounded offline sensitivity diagnostics for Gate F W_Q horizons.

This module is diagnostic-only. It reuses the production reachability
recurrence for each bounded prefix and records perturbation response with
explicit support and failure states. It does not alter Gate F configuration,
qualification, or biological claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .wq_diagnostic_input import load_gate_e_arrays
from ..dynamics.stochastic_reachability import compute_stochastic_reachability


HORIZONS = (1, 2, 4, 8, 16, 32)
N_STARTS = 8
EPSILONS = (1e-10, 1e-8, 1e-6)
SEEDS = (0, 1, 2)
PRODUCTION_PSD_FLOOR = 1e-8
BRANCHES = ("phi_only", "q_only")


def _code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _safe_scalar(value: Any) -> Any:
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    return _safe_scalar(value)


def _typed_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": arr.dtype.str, "shape": list(arr.shape)}, sort_keys=True).encode())
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def _metric_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    covariance = result.get("w_q")
    if covariance is None:
        return {
            "computation_status": result.get("computation_status"),
            "failure_reason": result.get("failure_reason"),
        }
    matrix = np.asarray(covariance, dtype=np.float64)
    return {
        "computation_status": result.get("computation_status"),
        "failure_reason": result.get("failure_reason"),
        "w_q_maxabs": float(np.max(np.abs(matrix))) if matrix.size else 0.0,
        "v_norm": result.get("v_norm"),
        "d_eff": result.get("d_eff"),
        "c_1_q": result.get("c_1_q"),
        "a_q": result.get("a_q"),
    }


def _direction_overlap(base: np.ndarray, perturbed: np.ndarray) -> float | None:
    if base.shape[0] < 2 or perturbed.shape[0] < 2:
        return None
    base_values, base_vectors = np.linalg.eigh(0.5 * (base + base.T))
    pert_values, pert_vectors = np.linalg.eigh(0.5 * (perturbed + perturbed.T))
    base_values = base_values[::-1]
    pert_values = pert_values[::-1]
    base_vectors = base_vectors[:, ::-1]
    pert_vectors = pert_vectors[:, ::-1]
    scale = max(abs(float(base_values[0])), abs(float(pert_values[0])), np.finfo(float).tiny)
    if (float(base_values[0] - base_values[1]) / scale) < 1e-6:
        return None
    if (float(pert_values[0] - pert_values[1]) / scale) < 1e-6:
        return None
    return float(abs(np.dot(base_vectors[:, 0], pert_vectors[:, 0])) ** 2)


def _unit_frobenius_noise(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(shape)
    norm = float(np.linalg.norm(noise.ravel(), ord=2))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("could not construct finite perturbation direction")
    return noise / norm


def _perturb_phi(phi: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    result = np.asarray(phi, dtype=np.float64).copy()
    for edge in range(result.shape[0]):
        direction = _unit_frobenius_noise(result[edge].shape, rng)
        magnitude = float(np.linalg.norm(result[edge].ravel(), ord=2))
        result[edge] += epsilon * magnitude * direction
    return result


def _perturb_q(q: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    dimension = q.shape[0]
    direction = _unit_frobenius_noise((dimension, dimension), rng)
    congruence = np.eye(dimension) + epsilon * direction
    perturbed = congruence @ q @ congruence.T
    return 0.5 * (perturbed + perturbed.T)


def _run_one(
    phi: np.ndarray,
    q: np.ndarray,
    q_metadata: Mapping[str, Any],
    horizon: int,
    start: int,
    perturbation_epsilon: float,
    branch: str,
    seed: int,
) -> dict[str, Any]:
    if start < 0 or start + horizon > len(phi):
        return {
            "status": "insufficient_support",
            "horizon": horizon,
            "start": start,
            "perturbation_epsilon": perturbation_epsilon,
            "branch": branch,
            "seed": seed,
        }
    branch_code = 0 if branch == "phi_only" else 1
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(start), branch_code]))
    window = np.asarray(phi[start : start + horizon], dtype=np.float64)
    q_contract = dict(q_metadata)
    q_contract.update({"covariance": q, "computation_status": "computed"})
    baseline = compute_stochastic_reachability(
        list(window), q, q_contract=q_contract, epsilon=PRODUCTION_PSD_FLOOR, precision="float64"
    )
    if baseline.get("computation_status") != "computed":
        return {
            "status": "baseline_failure",
            "baseline": _metric_summary(baseline),
            "base_w_q": baseline.get("w_q"),
            "base_w_q_projection_qc": baseline.get("w_q_projection_qc"),
            "horizon": horizon,
            "start": start,
            "perturbation_epsilon": perturbation_epsilon,
            "branch": branch,
            "seed": seed,
        }
    if branch == "phi_only":
        perturbed_phi = _perturb_phi(window, perturbation_epsilon, rng)
        perturbed_q = q
    elif branch == "q_only":
        perturbed_phi = window
        perturbed_q = _perturb_q(q, perturbation_epsilon, rng)
    else:
        raise ValueError(f"unknown perturbation branch: {branch}")
    perturbed_contract = dict(q_contract)
    perturbed_contract["covariance"] = perturbed_q
    input_phi_distortion = float(
        np.linalg.norm(perturbed_phi - window) / max(np.linalg.norm(window), np.finfo(float).tiny)
    )
    input_q_distortion = float(
        np.linalg.norm(perturbed_q - q) / max(np.linalg.norm(q), np.finfo(float).tiny)
    )
    perturbed = compute_stochastic_reachability(
        list(perturbed_phi), perturbed_q, q_contract=perturbed_contract,
        epsilon=PRODUCTION_PSD_FLOOR, precision="float64"
    )
    record: dict[str, Any] = {
        "status": "computed" if perturbed.get("computation_status") == "computed" else "perturbed_failure",
        "horizon": horizon,
        "start": start,
        "perturbation_epsilon": perturbation_epsilon,
        "branch": branch,
        "seed": seed,
        "baseline": _metric_summary(baseline),
        "perturbed": _metric_summary(perturbed),
        "base_w_q": baseline.get("w_q"),
        "variant_w_q": perturbed.get("w_q"),
        "base_w_q_projection_qc": baseline.get("w_q_projection_qc"),
        "relative_phi_distortion": input_phi_distortion,
        "relative_q_distortion": input_q_distortion,
        "variant_w_q_projection_qc": perturbed.get("w_q_projection_qc"),
    }
    if perturbed.get("computation_status") != "computed":
        return record
    base_w = np.asarray(baseline["w_q"], dtype=np.float64)
    pert_w = np.asarray(perturbed["w_q"], dtype=np.float64)
    denominator = max(
        float(np.max(np.abs(base_w))),
        float(np.max(np.abs(pert_w))),
        np.finfo(float).tiny,
    )
    record.update({
        "base_w_q": base_w,
        "variant_w_q": pert_w,
        "delta_w_maxabs_normalized": float(
            np.max(np.abs(pert_w / denominator - base_w / denominator))
        ),
        "delta_v_norm_abs": abs(float(perturbed["v_norm"]) - float(baseline["v_norm"])),
        "delta_d_eff_abs": abs(float(perturbed["d_eff"]) - float(baseline["d_eff"])),
        "delta_c_1_q_abs": abs(float(perturbed["c_1_q"]) - float(baseline["c_1_q"])),
        "delta_a_q_abs": abs(float(perturbed["a_q"]) - float(baseline["a_q"])),
        "top_direction_overlap_sq": _direction_overlap(base_w, pert_w),
        "variant_w_q_projection_qc": perturbed.get("w_q_projection_qc"),
    })
    return record


def sensitivity_from_arrays(
    phi: np.ndarray,
    q: np.ndarray,
    q_metadata: Mapping[str, Any],
    *,
    horizons: Sequence[int] = HORIZONS,
    n_starts: int = N_STARTS,
    epsilons: Sequence[float] = EPSILONS,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    phi = np.asarray(phi, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if phi.ndim != 3 or phi.shape[1] != phi.shape[2] or not np.all(np.isfinite(phi)):
        raise ValueError("phi must be finite square matrices")
    if q.ndim != 2 or q.shape != phi.shape[1:] or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite matrix matching phi dimension")
    horizons = tuple(int(h) for h in horizons)
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("horizons must be positive")
    if max(horizons) > len(phi):
        raise ValueError("horizon exceeds available transition support")
    starts = np.linspace(0, len(phi) - max(horizons), int(n_starts), dtype=int)
    starts = tuple(dict.fromkeys(int(value) for value in starts))
    records = [
        _run_one(phi, q, q_metadata, horizon, start, float(epsilon), branch, int(seed))
        for horizon in horizons
        for start in starts
        for epsilon in epsilons
        for branch in BRANCHES
        for seed in seeds
    ]
    protocol = {
        "horizons": list(horizons),
        "start_indices": list(starts),
        "epsilons": [float(v) for v in epsilons],
        "seeds": [int(v) for v in seeds],
        "branches": list(BRANCHES),
        "precision": "float64",
        "production_psd_floor": PRODUCTION_PSD_FLOOR,
        "q_perturbation": "PSD_congruence",
        "qualification_status": "not_assessed",
        "claim_status": "no_biological_claim",
    }
    invalid_counts: dict[str, int] = {}
    for row in records:
        status = str(row.get("status"))
        invalid_counts[status] = invalid_counts.get(status, 0) + 1
    source_payload = {
        **dict(q_metadata),
        "phi_sha256": _typed_hash(phi),
        "q_sha256": _typed_hash(q),
    }
    source_payload = _safe_json(source_payload)
    source_payload["producer_q"] = {
        "dtype": q.dtype.name,
        "shape": list(q.shape),
        "maxabs": float(np.max(np.abs(q))),
        "q_sha256": source_payload["q_sha256"],
    }
    source_payload["metadata_sha256"] = hashlib.sha256(
        json.dumps(source_payload, sort_keys=True).encode()
    ).hexdigest()
    return {
        "schema_version": "mndm.wq_sensitivity.v1",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "code_sha256": _code_hash(),
        "source": source_payload,
        "n_phi_steps": int(len(phi)),
        "matrix_dimension": int(phi.shape[1]),
        "invalid_counts": invalid_counts,
        "records": _safe_json(records),
    }


def run_h5_sensitivity(h5_path: Path, output_dir: Path, branch: str = "primary") -> Path:
    phi, q, source = load_gate_e_arrays(h5_path, branch=branch)
    q_metadata = dict(source)
    q_metadata.update(source["q_metadata"])
    q_metadata.update({
        "schema_version": source["q_metadata"]["schema_version"],
        "q_dt_sec": source["q_dt_sec"],
    })
    result = sensitivity_from_arrays(phi, q, q_metadata)
    result["source"].update({"h5_path": str(h5_path), "branch": branch, "source_sha256": source["source_sha256"]})
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{Path(h5_path).stem}_{branch}_wq_sensitivity.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    table = output.with_suffix(".csv")
    records = result["records"]
    fields = sorted({key for row in records for key in row})
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(row.get(key)) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in fields} for row in records)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", default="primary")
    args = parser.parse_args(argv)
    print(run_h5_sensitivity(args.h5, args.output_dir, args.branch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
