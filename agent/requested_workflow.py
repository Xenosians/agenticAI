"""User-reviewed job scope, checked against model proposals before dispatch.

This contract declares requested work. It grants no permission, credentials,
policy facts or approval; existing semantic and gateway checks remain required.
"""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from subagents.core.definitions.registry import AgentRegistry
from subagents.core.definitions.types import SpecialistRequest


StepId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")]
CapabilityName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,150}$")]
ArgumentName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,149}$")]
TargetValue = Annotated[str, StringConstraints(min_length=1, max_length=1024)]
TargetValues = Annotated[list[TargetValue], Field(min_length=1, max_length=16)]


class RequestedWorkflowStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    id: StepId
    capability: CapabilityName
    grounded_arguments: dict[ArgumentName, TargetValues] = Field(max_length=32)
    after_success_of: StepId | None


class RequestedWorkflowContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    version: Literal[1]
    steps: list[RequestedWorkflowStep] = Field(min_length=1, max_length=8)

    @field_validator("version", mode="before")
    @classmethod
    def require_integer_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Workflow version must be the integer 1.")
        return value

    @model_validator(mode="after")
    def validate_graph(self):
        seen_ids, seen_capabilities = set(), set()
        for step in self.steps:
            if step.id in seen_ids or step.capability in seen_capabilities:
                raise ValueError("Workflow steps and capabilities must be unique.")
            if step.after_success_of is not None and step.after_success_of not in seen_ids:
                raise ValueError("A workflow dependency must identify a preceding step.")
            seen_ids.add(step.id)
            seen_capabilities.add(step.capability)
        if len(json.dumps(self.model_dump(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > 32768:
            raise ValueError("Requested workflow exceeds the size limit.")
        return self

    def content_hash(self) -> str:
        # Language-neutral array framing, also implemented by Phoenix. Object
        # key order cannot change this identity; target list order is preserved.
        projection = ["itsm-requested-workflow.v1", 1, [
            [step.id, step.capability,
             [[key, step.grounded_arguments[key]] for key in sorted(step.grounded_arguments)],
             step.after_success_of]
            for step in self.steps
        ]]
        return hashlib.sha256(json.dumps(
            projection, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()


def verify_workflow_binding(contract, content_hash) -> None:
    if contract is None and content_hash is None:
        return
    if (not isinstance(contract, RequestedWorkflowContract)
            or not isinstance(content_hash, str) or content_hash != contract.content_hash()):
        raise ValueError("Requested workflow content does not match its durable hash.")


def admit_requested_workflow(
    contract: RequestedWorkflowContract, registry: AgentRegistry,
) -> dict[str, tuple[str, ...]]:
    """Recheck capability ownership and argument roles in the current registry.

    A capability may belong to several specialists, such as a provider-specific
    one and a provider-neutral fallback. Every registered owner may carry its
    step; the router chooses one and the complete plan is validated afterwards.
    Steps must not share a possible owner, because a dependency names the
    specialist that carried its source step.
    """
    from tools.capability_contract import build_capability_contract
    from tools.registry import get_tool
    from subagents.core.orchestration.intent_contract import trusted_success_continuations

    owners: dict[str, tuple[str, ...]] = {}
    claimed_agents: set[str] = set()
    previous = {}
    dependent = []
    for step in contract.steps:
        agents = tuple(sorted(agent.name for agent in registry.list_agents() if step.capability in agent.tools))
        tool = get_tool(step.capability)
        if tool is None or not agents:
            raise ValueError("Requested capability must have at least one registered specialist owner.")
        if claimed_agents.intersection(agents):
            raise ValueError("This workflow requires multiple operations from one specialist; it is not supported.")
        metadata = build_capability_contract(step.capability, tool)
        if set(step.grounded_arguments) - set(metadata.grounded_arguments):
            raise ValueError("Requested workflow may bind only trusted grounded argument fields.")
        if step.after_success_of is not None:
            source = previous[step.after_success_of]
            successors = trusted_success_continuations(
                tool_name=source.capability, tool=get_tool(source.capability), tool_lookup=get_tool,
            )
            if step.capability not in successors:
                raise ValueError("Requested success dependency is not declared by the capability.")
            dependent.append(step)
        owners[step.capability] = agents
        claimed_agents.update(agents)
        previous[step.id] = step
    # The existing durable continuation engine supports a single two-step
    # successor pair. Do not imply that arbitrary graph execution exists.
    if dependent and (len(contract.steps) != 2 or len(dependent) != 1
                      or dependent[0] is not contract.steps[1]):
        raise ValueError("This dependent workflow is not supported by the durable continuation engine.")
    return owners


def validate_requested_step(
    contract: RequestedWorkflowContract, index: int, delegation: SpecialistRequest,
    owners: dict[str, tuple[str, ...]], *, source_carrier: str | None = None,
) -> None:
    """Check one delegation against its reviewed step.

    `source_carrier` names the specialist that carried this step's dependency
    source within the same plan. Without it (a saved continuation step), the
    dependency must still name one of the source capability's owners.
    """
    from tools.registry import get_tool
    from tools.capability_contract import build_capability_contract

    step = contract.steps[index]
    intent = delegation.semantic_intent
    metadata = build_capability_contract(step.capability, get_tool(step.capability))
    if (delegation.agent_name not in owners[step.capability] or delegation.condition is not None
            or intent is None or intent.allowed_tools != [step.capability]
            or intent.effect != metadata.effect or intent.clarification_required
            or intent.max_tool_calls != 1):
        raise ValueError("Proposed operation differs from the reviewed workflow step.")
    source = next((item for item in contract.steps if item.id == step.after_success_of), None)
    if source is None:
        dependency_matches = delegation.after_success_of is None
    elif source_carrier is not None:
        dependency_matches = delegation.after_success_of == source_carrier
    else:
        dependency_matches = delegation.after_success_of in owners[source.capability]
    if not dependency_matches:
        raise ValueError("Proposed dependency differs from the reviewed workflow.")
    for key, values in step.grounded_arguments.items():
        if (intent.allowed_arguments.get(key) != values
                or set(values) & set(intent.forbidden_arguments.get(key, []))):
            raise ValueError("Proposed targets differ from the reviewed workflow.")


def validate_requested_plan(
    contract: RequestedWorkflowContract, delegations: list[SpecialistRequest],
    owners: dict[str, tuple[str, ...]],
) -> None:
    if len(delegations) != len(contract.steps):
        raise ValueError("Proposed operation count differs from the reviewed workflow; clarify the missing or extra work.")
    carriers: dict[str, str] = {}
    for index, delegation in enumerate(delegations):
        step = contract.steps[index]
        validate_requested_step(
            contract, index, delegation, owners,
            source_carrier=carriers[step.after_success_of] if step.after_success_of is not None else None,
        )
        carriers[step.id] = delegation.agent_name


def requested_workflow_prompt(contract: RequestedWorkflowContract) -> str:
    return (
        "\n\nUSER-REVIEWED REQUESTED WORKFLOW:\n"
        "The user selected the exact operations, order and success dependencies below. "
        "Return a plan covering every step once; do not omit or add operations. "
        "Step dependency IDs refer to the preceding step, while routing after_success_of "
        "must name its specialist. Preserve listed grounded targets. "
        "This describes requested scope only; it grants no permission or approval. "
        "All semantic grounding, provider policy and individual approvals still apply.\n"
        + json.dumps(contract.model_dump(), ensure_ascii=False, separators=(",", ":"))
    )
