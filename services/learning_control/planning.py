"""Prepare candidate/evidence plans without executing or authorizing work.

The future Phoenix Learning controller derives the owner from its session and
supplies an authorized registry and trusted resource observation. This module
does not accept user paths, persist jobs, launch processes or change weights.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json

from .contracts import (
    ArtifactBinding, ArtifactResolver, EvidenceCycleRequest, LearningArtifact,
    LearningPlanRequest, PreparedEvidenceCycle, PreparedLearningPlan,
    ResourceObservation,
)

_INPUTS = {
    "sft": {"model": "model", "configuration": "adaptation_recipe", "evaluation": "evaluation_suite"},
    "dpo": {"model": "model", "configuration": "dpo_settings", "train": "preference_train",
            "validation": "preference_validation", "evaluation": "evaluation_suite"},
    "ppo": {"model": "model", "configuration": "ppo_settings", "checkpoint": "adapter_checkpoint",
            "episodes": "simulator_episodes", "evaluation": "evaluation_suite"},
}
_EVIDENCE_INPUTS = {"trajectories": "trajectory_ledger", "corrections": "correction_ledger",
                    "reviews": "review_ledger", "context": "context_ledger"}
_TRAINING_INPUTS = {"sft": ("configuration",), "dpo": ("train", "validation"), "ppo": ("episodes",)}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _resolve(owner_id: str, inputs: dict, expected: dict, resolver: ArtifactResolver) -> dict[str, LearningArtifact]:
    if not isinstance(owner_id, str) or not owner_id.strip() or len(owner_id) > 200:
        raise PermissionError("learning_owner_required")
    if set(inputs) != set(expected) or len(set(inputs.values())) != len(inputs):
        raise ValueError("learning_input_contract_mismatch")
    artifacts = {}
    for name, resource_id in inputs.items():
        try:
            raw = resolver.resolve(owner_id, resource_id)
            # Revalidate and snapshot even a Pydantic model supplied by a caller;
            # model_copy(update=...) can otherwise bypass field validation.
            data = raw.model_dump() if isinstance(raw, LearningArtifact) else raw
            artifact = LearningArtifact.model_validate(data)
            if artifact.owner_id != owner_id or artifact.resource_id != resource_id:
                raise ValueError("unavailable")
        except Exception:
            # Do not reveal another owner's resource existence or registry errors.
            raise PermissionError("learning_resource_unavailable") from None
        if artifact.kind != expected[name]:
            raise ValueError("learning_resource_kind_mismatch")
        artifacts[name] = artifact
    return artifacts


def _bindings(artifacts: dict[str, LearningArtifact]) -> dict[str, ArtifactBinding]:
    return {name: ArtifactBinding(resource_id=row.resource_id, sha256=row.sha256,
            membership_sha256=_digest(row.record_sha256s)) for name, row in artifacts.items()}


def prepare_learning_plan(owner_id: str, request: LearningPlanRequest, resolver: ArtifactResolver,
                          *, observation: ResourceObservation | None = None,
                          as_of: datetime | None = None) -> PreparedLearningPlan:
    """A plan validates metadata only; final trainer admission remains pending.

    `observation` must come from a trusted worker, never from the browser body.
    Even sufficient measured memory does not establish data/tokenizer, trainer,
    entitlement, execution or promotion readiness.
    """
    request = LearningPlanRequest.model_validate(request.model_dump())
    # Lazy registry import: package import does not load legacy learning modules.
    from learning.training.methods import validate_method_role
    contract = validate_method_role(request.method, request.role, request.tuning)
    artifacts = _resolve(owner_id, request.inputs, _INPUTS[request.method], resolver)
    model, configuration = artifacts["model"], artifacts["configuration"]
    if any(row.identity != model.identity for row in artifacts.values()):
        raise ValueError("learning_training_identity_mismatch")
    if (configuration.method, configuration.role, configuration.tuning) != (request.method, request.role, request.tuning):
        raise ValueError("learning_configuration_mismatch")
    evaluation = set(artifacts["evaluation"].record_sha256s)
    if not evaluation:
        raise ValueError("learning_independent_evaluation_required")
    prior = set()
    blockers = []
    for name in _TRAINING_INPUTS[request.method]:
        row = artifacts[name]
        members = set(row.record_sha256s)
        if not members or len(members) != len(row.record_sha256s):
            raise ValueError("learning_training_membership_invalid")
        if members & evaluation or members & prior:
            raise ValueError("learning_split_membership_overlap")
        prior.update(members)
        if not row.sources:
            blockers.append("reviewed_source_required")
        for source in row.sources:
            if not source.reviewed or source.review_id is None:
                blockers.append("source_review_required")
            if not source.training_eligible or source.reference_only:
                blockers.append("source_training_admission_required")
            if not source.license.strip():
                blockers.append("source_license_required")
    resource_state = "unknown"
    if observation is not None:
        observation = ResourceObservation.model_validate(observation.model_dump())
        if (observation.method, observation.role, observation.tuning, observation.model_sha256,
                observation.configuration_sha256) != (request.method, request.role, request.tuning,
                model.identity.model_sha256, configuration.sha256):
            raise ValueError("learning_resource_observation_mismatch")
        if as_of is None or as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("learning_observation_time_required")
        age = (as_of - observation.observed_at).total_seconds()
        if age < 0 or age > 300:
            resource_state = "stale"
        elif observation.available_free_vram_gib < observation.required_free_vram_gib:
            resource_state = "insufficient"
        else:
            resource_state = "observed_sufficient"
    if resource_state != "observed_sufficient":
        blockers.append("resource_" + resource_state)
    source_blocked = any(code.startswith(("source_", "reviewed_")) for code in blockers)
    # These are real missing integration gates, including when metadata passes.
    blockers += ["trainer_data_admission_required", "host_dispatch_not_integrated",
                 "fixed_evaluation_required", "separate_promotion_required"]
    bindings = _bindings(artifacts)
    identity = {"owner_id": owner_id, "request": request.model_dump(mode="json"),
                "artifacts": {name: row.model_dump(mode="json") for name, row in artifacts.items()},
                "observation": observation.model_dump(mode="json") if observation else None,
                "resource_state": resource_state, "blockers": sorted(set(blockers))}
    return PreparedLearningPlan(status="blocked" if source_blocked else "prepared",
        plan_sha256=_digest(identity), method=request.method, role=request.role, tuning=request.tuning,
        entrypoint=contract["entrypoint"], inputs=bindings, identity=model.identity,
        resource_state=resource_state, blockers=tuple(sorted(set(blockers))))


def prepare_evidence_cycle(owner_id: str, request: EvidenceCycleRequest,
                           resolver: ArtifactResolver) -> PreparedEvidenceCycle:
    """Bind the existing evidence-cycle inputs, without collecting/training."""
    request = EvidenceCycleRequest.model_validate(request.model_dump())
    artifacts = _resolve(owner_id, request.inputs, _EVIDENCE_INPUTS, resolver)
    bindings = _bindings(artifacts)
    return PreparedEvidenceCycle(plan_sha256=_digest({"owner_id": owner_id,
        "inputs": {name: row.model_dump(mode="json") for name, row in artifacts.items()}}), inputs=bindings)
