from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentDefinition:
    name: str
    description: str

    tools: list[str] = field(
        default_factory=list
    )

    model: str = "local-qwen"
    max_steps: int = 3
    system_prompt: str = ""
    
@dataclass
class AgentTask:
    task_id: str
    agent_name: str
    user_request: str

    instructions: str | None = None

    context: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class AgentResult:
    task_id: str
    agent_name: str
    status: str

    answer: str | None = None
    proposed_tool: str | None = None
    proposed_arguments: dict[str, Any] | None = None
    error: str | None = None
    approval_id: str | None = None


@dataclass
class HubResult:
    status: str
    user_request: str

    routes: list[str] = field(
        default_factory=list
    )

    results: list[AgentResult] = field(
        default_factory=list
    )

    answer: str | None = None
    error: str | None = None


# ============================================================
# Planner types
# ============================================================


@dataclass
class HubTask:
    """
    A single task proposed by the Hub planner.

    This is a planning object only. It does not grant
    permission to execute anything.
    """

    id: str
    agent: str
    instruction: str

    identifiers: list[str] = field(
        default_factory=list
    )

    depends_on: list[str] = field(
        default_factory=list
    )

    condition: str | None = None


@dataclass
class HubPlan:
    """
    Validated task plan produced by the Hub planner.
    """

    tasks: list[HubTask] = field(
        default_factory=list
    )