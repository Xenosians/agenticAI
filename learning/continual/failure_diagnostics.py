from __future__ import annotations

import hashlib
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from learning.continual.storage import (
    canonical_json,
    immutable_write_json,
    immutable_write_jsonl,
    sha256_file,
)
from learning.curation.corpus_analysis import load_trajectories
from learning.evidence.types import LearningTrajectory, TrajectoryStep
from learning.paths import RUNTIME_LEARNING_ROOT, TRAJECTORIES_PATH


DEFAULT_FAILURE_MINING_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "continual"
    / "failure-mining"
)

CandidateComponent = Literal[
    "hub_router",
    "specialist",
    "tool_gateway",
    "provider",
    "unknown",
]


class FailureDiagnosis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(
        default="failure-diagnosis.v1",
        alias="schema",
    )
    diagnosis_id: str
    created_at: str
    trajectory_id: str
    task_id: str | None = None
    component: CandidateComponent
    failure_code: str
    agent: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    detail: str
    auto_fix_candidate: bool = False


class GroundedBindingCandidate(BaseModel):
    """
    Review-only candidate for a Hub grounding correction.

    This object is deliberately NOT a CorrectionEvent and is NEVER
    dataset eligible by itself. It exists only to make deterministic
    runtime evidence reviewable without allowing runtime failures to
    mutate training data automatically.
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(
        default="grounded-binding-candidate.v1",
        alias="schema",
    )
    candidate_id: str
    created_at: str
    trajectory_id: str
    task_id: str
    user_request: str
    agent: str
    tool: str
    failure_code: str
    missing_bindings: dict[str, list[str]]
    rejected_semantic_intent: dict[str, Any]
    proposed_semantic_intent: dict[str, Any]
    evidence: dict[str, Any]
    review_required: bool = True
    training_eligible: bool = False


class FailureMiningManifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(
        default="failure-mining-manifest.v1",
        alias="schema",
    )
    run_id: str
    created_at: str
    output_directory: str
    trajectories_path: str
    trajectories_sha256: str | None = None
    trajectory_count: int
    diagnosis_count: int
    candidate_count: int
    failure_counts: dict[str, int] = Field(default_factory=dict)
    candidate_failure_counts: dict[str, int] = Field(default_factory=dict)
    training_executed: bool = False
    automatic_promotion_performed: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return f"{prefix}-{digest[:24]}"


def _semantic_guard_code(step: TrajectoryStep) -> str | None:
    decision = step.semantic_guard_decision
    if not isinstance(decision, dict):
        return None
    code = decision.get("decision_code")
    if not isinstance(code, str) or not code.strip():
        return None
    return code.strip()


def _tool_from_step(step: TrajectoryStep) -> str | None:
    if isinstance(step.proposed_tool, str) and step.proposed_tool.strip():
        return step.proposed_tool.strip()
    return None


def _value_literals(value: Any) -> list[str] | None:
    if isinstance(value, str):
        if not value or value != value.strip():
            return None
        return [value]

    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                return None
            if not item or item != item.strip():
                return None
            result.append(item)
        return result if result else None

    return None


def _literal_is_explicit_in_request(*, value: str, user_request: str) -> bool:
    """
    Conservative literal evidence check.

    We do not fuzzy-match, normalize identifiers, translate values,
    or infer aliases. A candidate may be proposed only when the exact
    model-proposed grounded value visibly appears in the original user
    request. Human/trusted review is still mandatory afterwards.
    """

    return value in user_request


def _diagnosis_for_step(
    *,
    trajectory: LearningTrajectory,
    step: TrajectoryStep,
) -> FailureDiagnosis | None:
    code = _semantic_guard_code(step)

    if code is not None:
        component: CandidateComponent = "hub_router"
        detail = (
            "SemanticGuard rejected the specialist proposal against "
            "the Hub semantic contract."
        )

        if code in {
            "semantic_tool_not_available",
            "semantic_tool_forbidden",
            "semantic_tool_not_allowed",
        }:
            component = "hub_router"
            detail = (
                "The selected/proposed capability did not match the "
                "validated Hub capability boundary."
            )
        elif code in {
            "semantic_argument_unbound",
            "semantic_argument_not_allowed",
            "semantic_bound_argument_missing",
            "semantic_grounded_argument_invalid",
            "semantic_argument_forbidden",
        }:
            component = "hub_router"
            detail = (
                "The Hub/specialist grounded-argument contract did "
                "not agree with the proposed capability arguments."
            )
        elif code in {
            "semantic_effect_mismatch",
            "semantic_effect_unknown",
        }:
            component = "hub_router"
            detail = (
                "The Hub semantic effect did not agree with trusted "
                "capability metadata."
            )

        payload = {
            "trajectory_id": trajectory.trajectory_id,
            "task_id": step.task_id,
            "failure_code": code,
            "agent": step.agent,
            "tool": _tool_from_step(step),
        }
        return FailureDiagnosis(
            diagnosis_id=_stable_id("diagnosis", payload),
            created_at=_utc_now(),
            trajectory_id=trajectory.trajectory_id,
            task_id=step.task_id,
            component=component,
            failure_code=code,
            agent=step.agent,
            tool=_tool_from_step(step),
            arguments=dict(step.proposed_arguments or {}),
            detail=detail,
            auto_fix_candidate=(code == "semantic_argument_unbound"),
        )

    gateway = step.gateway_decision
    if isinstance(gateway, dict):
        code = gateway.get("decision_code")
        status = gateway.get("status")
        if isinstance(code, str) and code.strip():
            payload = {
                "trajectory_id": trajectory.trajectory_id,
                "task_id": step.task_id,
                "failure_code": code.strip(),
                "agent": step.agent,
                "tool": _tool_from_step(step),
            }
            return FailureDiagnosis(
                diagnosis_id=_stable_id("diagnosis", payload),
                created_at=_utc_now(),
                trajectory_id=trajectory.trajectory_id,
                task_id=step.task_id,
                component="tool_gateway",
                failure_code=code.strip(),
                agent=step.agent,
                tool=_tool_from_step(step),
                arguments=dict(step.proposed_arguments or {}),
                detail=(
                    "ToolGateway emitted a machine-readable runtime "
                    f"decision (status={status!r})."
                ),
                auto_fix_candidate=False,
            )

    if step.status == "error" and step.outcome_code:
        payload = {
            "trajectory_id": trajectory.trajectory_id,
            "task_id": step.task_id,
            "failure_code": step.outcome_code,
            "agent": step.agent,
            "tool": _tool_from_step(step),
        }
        return FailureDiagnosis(
            diagnosis_id=_stable_id("diagnosis", payload),
            created_at=_utc_now(),
            trajectory_id=trajectory.trajectory_id,
            task_id=step.task_id,
            component="unknown",
            failure_code=step.outcome_code,
            agent=step.agent,
            tool=_tool_from_step(step),
            arguments=dict(step.proposed_arguments or {}),
            detail="Runtime specialist failure without a supported deterministic repair rule.",
            auto_fix_candidate=False,
        )

    return None


def _grounded_binding_candidate(
    *,
    trajectory: LearningTrajectory,
    step: TrajectoryStep,
) -> GroundedBindingCandidate | None:
    if _semantic_guard_code(step) != "semantic_argument_unbound":
        return None

    intent = step.semantic_intent
    arguments = step.proposed_arguments
    tool = _tool_from_step(step)

    if not isinstance(intent, dict):
        return None
    if not isinstance(arguments, dict) or not arguments:
        return None
    if tool is None:
        return None

    allowed_tools = intent.get("allowed_tools")
    if not isinstance(allowed_tools, list) or allowed_tools != [tool]:
        # Do not repair arguments if tool identity itself is not already
        # exactly and uniquely bound by the Hub contract.
        return None

    allowed_arguments = intent.get("allowed_arguments")
    if not isinstance(allowed_arguments, dict):
        allowed_arguments = {}

    missing: dict[str, list[str]] = {}

    for name, raw_value in arguments.items():
        if name in allowed_arguments and allowed_arguments.get(name):
            continue

        literals = _value_literals(raw_value)
        if literals is None:
            continue

        if not all(
            _literal_is_explicit_in_request(
                value=value,
                user_request=trajectory.user_request,
            )
            for value in literals
        ):
            continue

        missing[name] = literals

    if not missing:
        return None

    proposed_intent = json.loads(json.dumps(intent))
    proposed_allowed = dict(proposed_intent.get("allowed_arguments") or {})
    for name, literals in missing.items():
        proposed_allowed[name] = list(literals)
    proposed_intent["allowed_arguments"] = proposed_allowed

    payload = {
        "trajectory_id": trajectory.trajectory_id,
        "task_id": step.task_id,
        "agent": step.agent,
        "tool": tool,
        "failure_code": "semantic_argument_unbound",
        "missing_bindings": missing,
        "rejected_semantic_intent": intent,
        "proposed_semantic_intent": proposed_intent,
    }

    decision = step.semantic_guard_decision or {}

    return GroundedBindingCandidate(
        candidate_id=_stable_id("candidate", payload),
        created_at=_utc_now(),
        trajectory_id=trajectory.trajectory_id,
        task_id=step.task_id,
        user_request=trajectory.user_request,
        agent=step.agent,
        tool=tool,
        failure_code="semantic_argument_unbound",
        missing_bindings=missing,
        rejected_semantic_intent=json.loads(json.dumps(intent)),
        proposed_semantic_intent=proposed_intent,
        evidence={
            "semantic_guard_decision": json.loads(json.dumps(decision)),
            "proposed_arguments": json.loads(json.dumps(arguments)),
            "exact_literal_check": True,
            "candidate_rule": "explicit-proposed-grounded-value-missing-from-hub-binding.v1",
        },
        review_required=True,
        training_eligible=False,
    )


def diagnose_trajectory(
    trajectory: LearningTrajectory,
) -> tuple[list[FailureDiagnosis], list[GroundedBindingCandidate]]:
    diagnoses: list[FailureDiagnosis] = []
    candidates: list[GroundedBindingCandidate] = []

    for step in trajectory.steps:
        diagnosis = _diagnosis_for_step(
            trajectory=trajectory,
            step=step,
        )
        if diagnosis is not None:
            diagnoses.append(diagnosis)

        candidate = _grounded_binding_candidate(
            trajectory=trajectory,
            step=step,
        )
        if candidate is not None:
            candidates.append(candidate)

    return diagnoses, candidates


def mine_failures(
    trajectories: list[LearningTrajectory],
) -> tuple[list[FailureDiagnosis], list[GroundedBindingCandidate]]:
    diagnoses: list[FailureDiagnosis] = []
    candidates: list[GroundedBindingCandidate] = []

    for trajectory in trajectories:
        item_diagnoses, item_candidates = diagnose_trajectory(trajectory)
        diagnoses.extend(item_diagnoses)
        candidates.extend(item_candidates)

    # Deterministic de-duplication by stable IDs.
    diagnoses = list({item.diagnosis_id: item for item in diagnoses}.values())
    candidates = list({item.candidate_id: item for item in candidates}.values())

    diagnoses.sort(key=lambda item: item.diagnosis_id)
    candidates.sort(key=lambda item: item.candidate_id)
    return diagnoses, candidates


def run_failure_mining(
    *,
    trajectories_path: Path = TRAJECTORIES_PATH,
    output_root: Path = DEFAULT_FAILURE_MINING_ROOT,
) -> FailureMiningManifest:
    trajectories_path = trajectories_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    trajectories = load_trajectories(trajectories_path)
    diagnoses, candidates = mine_failures(trajectories)

    identity = {
        "trajectories_sha256": sha256_file(trajectories_path),
        "diagnosis_ids": [item.diagnosis_id for item in diagnoses],
        "candidate_ids": [item.candidate_id for item in candidates],
        "algorithm": "failure-mining-v1",
    }
    run_id = _stable_id("failure-mining", identity)
    directory = output_root / run_id
    manifest_path = directory / "manifest.json"

    if manifest_path.is_file():
        return FailureMiningManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )

    directory.mkdir(parents=True, exist_ok=False)
    immutable_write_jsonl(directory / "diagnoses.jsonl", diagnoses)
    immutable_write_jsonl(directory / "candidates.jsonl", candidates)

    manifest = FailureMiningManifest(
        run_id=run_id,
        created_at=_utc_now(),
        output_directory=str(directory),
        trajectories_path=str(trajectories_path),
        trajectories_sha256=sha256_file(trajectories_path),
        trajectory_count=len(trajectories),
        diagnosis_count=len(diagnoses),
        candidate_count=len(candidates),
        failure_counts=dict(Counter(item.failure_code for item in diagnoses)),
        candidate_failure_counts=dict(Counter(item.failure_code for item in candidates)),
        training_executed=False,
        automatic_promotion_performed=False,
    )
    immutable_write_json(manifest_path, manifest)
    return manifest
