from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentDefinition:
    """
    Static configuration describing a sub-agent.
    """

    name: str
    description: str
    tools: list[str] = field(default_factory=list)
    model: str = "local-qwen"
    max_steps: int = 3
    system_prompt: str = ""


@dataclass
class AgentTask:
    """
    A task sent from the hub/orchestrator to a worker agent.
    """

    task_id: str
    agent_name: str
    user_request: str
    instructions: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """
    Structured result returned by a worker agent.
    """

    task_id: str
    agent_name: str
    status: str

    answer: str | None = None

    proposed_tool: str | None = None
    proposed_arguments: dict[str, Any] | None = None

    error: str | None = None
    
@dataclass
class HubResult:
    """
    Result returned by the hub/orchestrator.
    """

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