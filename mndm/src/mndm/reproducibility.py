"""Helpers for reproducibility policy and seed resolution."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple


DEFAULT_SEED = 42

STOCHASTIC_COMPONENTS: tuple[str, ...] = (
    "feature_worker_rng",
    "robustness_bootstrap",
    "null_sanity_tests",
    "epoch_stage_stratified_sampling",
    "preprocess_ica",
    "fmri_louvain_modularity",
)

DETERMINISTIC_OUTPUTS: tuple[str, ...] = (
    "mnps_3d",
    "mnps_3d_dot",
    "coords_9d",
    "nn_indices",
    "jacobian.J_hat",
    "jacobian.J_dot",
    "jacobian_9D.J_hat",
    "jacobian_9D.J_dot",
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int_seed(value: Any, default: int = DEFAULT_SEED) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _resolve_reproducibility_cfg(
    config: Mapping[str, Any] | None,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    root = _as_mapping(_as_mapping(config).get("reproducibility"))
    merged: Dict[str, Any] = {k: v for k, v in root.items() if k != "datasets"}
    ds_map = _as_mapping(root.get("datasets"))
    if dataset_id and dataset_id in ds_map and isinstance(ds_map.get(dataset_id), Mapping):
        merged.update(dict(ds_map.get(dataset_id) or {}))
    return merged


def _legacy_base_seed(
    config: Mapping[str, Any] | None,
    dataset_id: Optional[str] = None,
) -> Tuple[int, str]:
    cfg = _as_mapping(config)
    robustness_cfg = _as_mapping(cfg.get("robustness"))
    if "seed" in robustness_cfg:
        return _int_seed(robustness_cfg.get("seed")), "robustness.seed"

    epoching_cfg = _as_mapping(cfg.get("epoching"))
    sampling_cfg = _as_mapping(epoching_cfg.get("sampling"))
    if "seed" in sampling_cfg:
        return _int_seed(sampling_cfg.get("seed")), "epoching.sampling.seed"
    ds_epoching = _as_mapping(_as_mapping(epoching_cfg.get("datasets")).get(dataset_id)) if dataset_id else {}
    ds_sampling = _as_mapping(ds_epoching.get("sampling"))
    if "seed" in ds_sampling:
        return _int_seed(ds_sampling.get("seed")), f"epoching.datasets.{dataset_id}.sampling.seed"

    preprocess_cfg = _as_mapping(cfg.get("preprocess"))
    artifacts_cfg = _as_mapping(preprocess_cfg.get("artifacts"))
    if "ica_random_state" in artifacts_cfg:
        return _int_seed(artifacts_cfg.get("ica_random_state")), "preprocess.artifacts.ica_random_state"
    ds_artifacts = _as_mapping(_as_mapping(artifacts_cfg.get("datasets")).get(dataset_id)) if dataset_id else {}
    if "ica_random_state" in ds_artifacts:
        return _int_seed(ds_artifacts.get("ica_random_state")), f"preprocess.artifacts.datasets.{dataset_id}.ica_random_state"

    return int(DEFAULT_SEED), "default"


def resolve_base_seed(
    config: Mapping[str, Any] | None,
    dataset_id: Optional[str] = None,
) -> Tuple[int, str]:
    repro_cfg = _resolve_reproducibility_cfg(config, dataset_id)
    if "seed" in repro_cfg:
        return _int_seed(repro_cfg.get("seed")), "reproducibility.seed"
    return _legacy_base_seed(config, dataset_id)


def resolve_component_seed(
    config: Mapping[str, Any] | None,
    *,
    dataset_id: Optional[str] = None,
    fallback_seed: Any = None,
    fallback_source: Optional[str] = None,
    offset: int = 0,
) -> Tuple[int, str]:
    repro_cfg = _resolve_reproducibility_cfg(config, dataset_id)
    if "seed" in repro_cfg:
        return _int_seed(repro_cfg.get("seed")) + int(offset), "reproducibility.seed"
    if fallback_seed is not None:
        return _int_seed(fallback_seed) + int(offset), str(fallback_source or "component.seed")
    base_seed, source = resolve_base_seed(config, dataset_id)
    return int(base_seed) + int(offset), source


def resolve_reproducibility_policy(
    config: Mapping[str, Any] | None,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    seed, source = resolve_base_seed(config, dataset_id)
    return {
        "seed": int(seed),
        "seed_source": str(source),
        "stochastic_components": list(STOCHASTIC_COMPONENTS),
        "deterministic_outputs": list(DETERMINISTIC_OUTPUTS),
    }
