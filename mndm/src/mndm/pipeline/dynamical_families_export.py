"""Opt-in orchestration for standard dynamical-family contracts."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..dynamical_families import (
    estimate_committor_local_law_dense_grid_o2b,
    estimate_local_diffusion_geometry,
    summarize_finite_amplitude_resilience,
)
from ..dynamical_families.chart_drift import resolve_ingest_chart_drift
from ..dynamical_families.contracts import (
    COMMITTOR_SCHEMA_VERSION,
    FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
    MNDM_TRANSLATION_VALIDATED,
    unavailable_result,
)
from ..measurement_certificate import attach_certificate
from ..inferential_grain import GRAIN_BY_FAMILY_ID, attach_grain
from ..dynamical_families.registry import (
    LEGACY_FAMILY_CONFIG_KEYS,
    family_config,
)

LEGACY_CONFIG_KEY = "orthogonal_dynamics"


def reject_legacy_family_config_key(config: Mapping[str, Any] | None) -> None:
    """Refuse pre-v3 YAML keys. No alias is provided."""
    if not isinstance(config, Mapping):
        return
    if LEGACY_CONFIG_KEY in config:
        raise ValueError(
            "Config key 'orthogonal_dynamics' was renamed to 'dynamical_families'. "
            "Update the overlay; no YAML alias is provided."
        )
    root = config.get("dynamical_families", {})
    if not isinstance(root, Mapping):
        return
    if "spread" in root:
        raise ValueError(
            "Family 'spread' is gate_closed (Gate F). No YAML enable flag is "
            "provided and W_Q is not emitted."
        )
    for old, new in LEGACY_FAMILY_CONFIG_KEYS.items():
        if old in root:
            raise ValueError(
                f"Config key 'dynamical_families.{old}' was renamed to "
                f"'dynamical_families.{new}'. No YAML alias is provided."
            )


def _family_root(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    reject_legacy_family_config_key(config)
    root = config.get("dynamical_families", {})
    return root if isinstance(root, Mapping) else {}


def _method_tags_from_qualification(qualification: Mapping[str, Any]) -> dict[str, str]:
    """Copy adapter method tags when both id and hash are already recorded.

    Tags do not authorize computation and do not set measurement_validity.
    """
    qid = str(qualification.get("qualification_id") or "").strip()
    qhash = str(qualification.get("qualification_contract_hash") or "").strip()
    if not qid or not qhash:
        return {}
    return {
        "qualification_id": qid,
        "qualification_contract_hash": qhash,
        "validation_level": MNDM_TRANSLATION_VALIDATED,
    }


def _stamp_computed_method_tags(
    result: Mapping[str, Any],
    *,
    qualification: Mapping[str, Any],
    family_id: str,
    family_object: str,
) -> dict[str, Any]:
    out = dict(result)
    if out.get("computation_status") != "computed":
        return out
    provenance = dict(out.get("provenance") or {}) if isinstance(out.get("provenance"), Mapping) else {}
    provenance.update({"family_id": family_id, "family_object": family_object})
    provenance.update(_method_tags_from_qualification(qualification))
    out["provenance"] = provenance
    return out


def _certify_family_result(result: Mapping[str, Any], family_id: str) -> dict[str, Any]:
    provenance = result.get("provenance") if isinstance(result.get("provenance"), Mapping) else {}
    # Diffusion: OD-TQ1 is a method tag in provenance, not regime validity.
    # Destination / resilience still use recorded TQ id+hash as translation_qualified.
    translation_qualified = (
        family_id != "diffusion"
        and result.get("computation_status") == "computed"
        and bool(str(provenance.get("qualification_id") or "").strip())
        and bool(str(provenance.get("qualification_contract_hash") or "").strip())
    )
    certified = attach_certificate(result, translation_qualified=translation_qualified)
    spec = GRAIN_BY_FAMILY_ID.get(family_id)
    if spec is None:
        raise ValueError(f"No inferential grain registered for family: {family_id}")
    return attach_grain(certified, **spec)


def build_dynamical_families_export(
    *,
    config: Mapping[str, Any],
    state: np.ndarray,
    time: np.ndarray,
    stage: np.ndarray | None,
    segment_id: np.ndarray | None,
    coordinate_layer: str,
    coordinate_names: list[str],
    reaction_coordinate: np.ndarray | None = None,
    reaction_coordinate_name: str = "reaction_coordinate",
    perturbation_amplitudes: np.ndarray | None = None,
    returned_to_reference: np.ndarray | None = None,
    perturbation_survived: np.ndarray | None = None,
    recovery_time_sec: np.ndarray | None = None,
    perturbation_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build explicitly requested exports without changing MNPS."""
    root = _family_root(config)
    if not bool(root.get("enabled", False)):
        return {}
    export: dict[str, Any] = {}
    diffusion_cfg = family_config(root, "diffusion")
    if bool(diffusion_cfg.get("enabled", False)):
        neighborhood = diffusion_cfg.get("neighborhood", {})
        neighborhood = neighborhood if isinstance(neighborhood, Mapping) else {}
        qualification = diffusion_cfg.get("translation_qualification", {})
        qualification = qualification if isinstance(qualification, Mapping) else {}
        # Compute when the estimator has support. YAML TQ is a provenance
        # method tag, not an on/off gate. Registry forbid:
        # jacobian_derivative_residual_as_diffusion. MNPS x_dot is not an
        # independently qualified SDE drift estimator. C1 alignment is
        # library-only in M1–M2; ingest never residualizes increments and
        # never passes payload.x_dot.
        drift_resolution = resolve_ingest_chart_drift(diffusion_cfg)
        diffusion_result = estimate_local_diffusion_geometry(
            state,
            time,
            drift=drift_resolution.field,
            residualize_increments=False,
            drift_source=drift_resolution.source,
            segment_id=segment_id,
            coordinate_layer=coordinate_layer,
            coordinate_names=coordinate_names,
            neighborhood_k=int(neighborhood.get("k", 20)),
            min_samples=int(diffusion_cfg.get("min_samples", 30)),
            min_neighborhood_samples=int(diffusion_cfg.get("min_neighborhood_samples", 10)),
            max_gap_sec=diffusion_cfg.get("max_gap_sec"),
            max_dt_relative_deviation=float(diffusion_cfg.get("max_dt_relative_deviation", 0.05)),
            min_valid_fraction=float(diffusion_cfg.get("min_valid_fraction", 0.1)),
            psd_floor=float(diffusion_cfg.get("psd_floor", 1e-8)),
        )
        if (
            diffusion_result.get("computation_status") == "computed"
            and drift_resolution.field is None
            and drift_resolution.failure_reason
        ):
            summary = dict(diffusion_result.get("summary") or {})
            if summary.get("A_bD_computation_status") == "not_testable":
                summary["drift_alignment_failure_reason"] = drift_resolution.failure_reason
            provenance = dict(diffusion_result.get("provenance") or {})
            settings = dict(provenance.get("settings") or {})
            settings["drift_source"] = drift_resolution.source
            settings["drift_mode"] = drift_resolution.mode
            provenance["settings"] = settings
            diffusion_result = {
                **diffusion_result,
                "summary": summary,
                "provenance": provenance,
            }
        export["diffusion"] = _stamp_computed_method_tags(
            diffusion_result,
            qualification=qualification,
            family_id="diffusion",
            family_object="diffusion_geometry",
        )
    destination_cfg = family_config(root, "destination")
    if bool(destination_cfg.get("enabled", False)):
        explicit_label_source = (
            str(destination_cfg.get("regime_source", "")).strip() == "explicit_first_hit_labels"
            and str(destination_cfg.get("label_key", "")).strip() == "stage"
        )
        reaction_coordinate_spec = destination_cfg.get("reaction_coordinate")
        dense_reaction_coordinate = (
            isinstance(reaction_coordinate_spec, Mapping)
            and str(reaction_coordinate_spec.get("source", "")).strip() == "explicit_column"
            and bool(str(reaction_coordinate_spec.get("key", "")).strip())
            and reaction_coordinate is not None
        )
        destination_qualification = destination_cfg.get("translation_qualification", {})
        destination_qualification = (
            destination_qualification
            if isinstance(destination_qualification, Mapping)
            else {}
        )
        if stage is None or not explicit_label_source or not dense_reaction_coordinate:
            export["destination"] = unavailable_result(
                COMMITTOR_SCHEMA_VERSION,
                status="not_testable",
                failure_reason="explicit_A_B_reaction_coordinate_contract_not_configured",
                coordinate_layer=coordinate_layer,
                coordinate_names=coordinate_names,
            )
        elif str(destination_cfg.get("estimator", "")).strip() != "local_law_dense_grid_o2b":
            export["destination"] = unavailable_result(
                COMMITTOR_SCHEMA_VERSION,
                status="not_testable",
                failure_reason="unsupported_committor_estimator",
                coordinate_layer=coordinate_layer,
                coordinate_names=coordinate_names,
            )
        elif not bool(destination_qualification.get("qualified", False)):
            export["destination"] = unavailable_result(
                COMMITTOR_SCHEMA_VERSION,
                status="not_testable",
                failure_reason="mndm_committor_adapter_translation_qualification_required",
                coordinate_layer=coordinate_layer,
                coordinate_names=coordinate_names,
            )
        elif not str(destination_qualification.get("qualification_id", "")).strip() or not str(
            destination_qualification.get("qualification_contract_hash", "")
        ).strip():
            export["destination"] = unavailable_result(
                COMMITTOR_SCHEMA_VERSION,
                status="not_testable",
                failure_reason="committor_qualification_metadata_required",
                coordinate_layer=coordinate_layer,
                coordinate_names=coordinate_names,
            )
        else:
            boundaries = reaction_coordinate_spec.get("boundaries", ())
            boundaries = list(boundaries) if isinstance(boundaries, (list, tuple)) else []
            if len(boundaries) != 2:
                export["destination"] = unavailable_result(
                    COMMITTOR_SCHEMA_VERSION,
                    status="invalid",
                    failure_reason="reaction_coordinate_boundaries_required",
                    coordinate_layer=coordinate_layer,
                    coordinate_names=coordinate_names,
                )
            else:
                result = estimate_committor_local_law_dense_grid_o2b(
                    state,
                    time,
                    reaction_coordinate,
                    stage,
                    set_A=destination_cfg.get("set_A", []),
                    set_B=destination_cfg.get("set_B", []),
                    grid_min=float(boundaries[0]),
                    grid_max=float(boundaries[1]),
                    diffusion_coefficient=float(destination_cfg.get("diffusion_coefficient", np.nan)),
                    segment_id=segment_id,
                    coordinate_layer=coordinate_layer,
                    coordinate_names=coordinate_names,
                    reaction_coordinate_name=reaction_coordinate_name,
                    grid_resolution=int(destination_cfg.get("grid_resolution", 65)),
                    min_samples=int(destination_cfg.get("min_samples", 30)),
                    min_support_per_grid=int(destination_cfg.get("min_support_per_grid", 30)),
                    min_transition_segments=int(destination_cfg.get("min_transition_segments", 5)),
                    max_dt_relative_deviation=float(
                        destination_cfg.get("max_dt_relative_deviation", 0.05)
                    ),
                )
                if (
                    result.get("computation_status") == "computed"
                    and isinstance(result.get("provenance"), Mapping)
                ):
                    result["provenance"].update(
                        {
                            "qualification_id": destination_qualification.get("qualification_id"),
                            "qualification_contract_hash": destination_qualification.get(
                                "qualification_contract_hash"
                            ),
                            "validation_level": MNDM_TRANSLATION_VALIDATED,
                            "family_id": "destination",
                            "family_object": "committor",
                        }
                    )
                export["destination"] = result
    resilience_cfg = family_config(root, "resilience")
    if bool(resilience_cfg.get("enabled", False)):
        protocol_source = str(resilience_cfg.get("protocol_source", "")).strip()
        resilience_qualification = resilience_cfg.get("translation_qualification", {})
        resilience_qualification = (
            resilience_qualification
            if isinstance(resilience_qualification, Mapping)
            else {}
        )
        explicit_outcomes = (
            protocol_source == "explicit_perturbation_outcomes"
            and perturbation_amplitudes is not None
            and returned_to_reference is not None
        )
        protocol = perturbation_protocol if isinstance(perturbation_protocol, Mapping) else {}
        required_protocol_fields = (
            "perturbation_direction",
            "perturbation_time",
            "reference_attractor_or_state",
            "return_criterion",
            "escape_criterion",
            "observation_horizon",
        )
        complete_protocol = (
            all(str(protocol.get(field, "")).strip() for field in required_protocol_fields)
            and isinstance(protocol.get("non_return_is_escape"), (bool, np.bool_))
        )
        if not explicit_outcomes:
            export["resilience"] = unavailable_result(
                FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
                status="not_testable",
                failure_reason="no_perturbation_protocol",
                coordinate_layer=coordinate_layer,
                coordinate_names=coordinate_names,
            )
        elif not complete_protocol:
            export["resilience"] = unavailable_result(
                FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
                status="not_testable",
                failure_reason="incomplete_perturbation_protocol",
                coordinate_layer=coordinate_layer,
                coordinate_names=coordinate_names,
            )
        elif str(resilience_cfg.get("estimator", "")).strip() != "observed_perturbation_outcome_summary":
            export["resilience"] = unavailable_result(
                FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
                status="not_testable",
                failure_reason="unsupported_resilience_estimator",
                coordinate_layer=coordinate_layer,
                coordinate_names=coordinate_names,
            )
        elif not bool(resilience_qualification.get("qualified", False)):
            export["resilience"] = unavailable_result(
                FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
                status="not_testable",
                failure_reason="mndm_resilience_adapter_translation_qualification_required",
                coordinate_layer=coordinate_layer,
                coordinate_names=coordinate_names,
            )
        elif not str(resilience_qualification.get("qualification_id") or "").strip() or not str(
            resilience_qualification.get("qualification_contract_hash") or ""
        ).strip():
            export["resilience"] = unavailable_result(
                FINITE_AMPLITUDE_RESILIENCE_SCHEMA_VERSION,
                status="not_testable",
                failure_reason="resilience_qualification_metadata_required",
                coordinate_layer=coordinate_layer,
                coordinate_names=coordinate_names,
            )
        else:
            result = summarize_finite_amplitude_resilience(
                perturbation_amplitudes,
                returned_to_reference,
                survived=perturbation_survived,
                recovery_time_sec=recovery_time_sec,
                coordinate_layer=coordinate_layer,
                coordinate_names=coordinate_names,
                min_trials_per_amplitude=int(
                    resilience_cfg.get("min_trials_per_amplitude", 20)
                ),
                protocol=protocol,
            )
            if (
                result.get("computation_status") == "computed"
                and isinstance(result.get("provenance"), Mapping)
            ):
                result["provenance"].update(
                    {
                        "qualification_id": resilience_qualification.get(
                            "qualification_id"
                        ),
                        "qualification_contract_hash": resilience_qualification.get(
                            "qualification_contract_hash"
                        ),
                        "validation_level": MNDM_TRANSLATION_VALIDATED,
                        "family_id": "resilience",
                        "family_object": "finite_amplitude_resilience",
                    }
                )
            export["resilience"] = result
    return {
        key: _certify_family_result(value, key)
        for key, value in export.items()
        if isinstance(value, Mapping)
    }
