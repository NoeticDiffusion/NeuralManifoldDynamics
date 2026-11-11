"""Experimental MNPS 3.2 namespace.

Status: partial implementation. Experimental formula kernels live in their
explicit ``v3_2`` modules and are not re-exported here. No candidate registry,
active pipeline, or export integration is provided.
Plan: project/new_mnps_v.3.2/IMPLEMENTATION_PLAN.md
Contracts: project/new_mnps_v.3.2/CONTRACT_AND_DECISIONS.md
Validation: project/new_mnps_v.3.2/VALIDATION_PLAN.md

PCV and activity-vector AR now have low-level formula implementations; their
end-to-end EEG realizations remain unresolved. Do not register them as active
pipeline estimators until their support and realization contracts exist.
"""
