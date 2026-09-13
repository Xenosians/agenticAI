from dataclasses import (
    dataclass,
    field,
)

from typing import (
    Any,
)


@dataclass
class AgentDefinition:
    name: str

    description: str

    model: str

    tools: list[str] = field(
        default_factory=list
    )

    max_steps: int = 3

    system_prompt: str = ""


@dataclass
class SpecialistRequest:
    """
    Structured Hub -> specialist delegation.

    The Hub may describe what it wants the specialist to inspect,
    but this object is NOT authorization.

    Trusted tool authorization still belongs to ToolGateway.
    """

    agent_name: str

    instructions: str


@dataclass
class AgentTask:
    task_id: str

    agent_name: str

    user_request: str

    instructions: (
        str | None
    ) = None

    context: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


@dataclass
class AgentResult:
    task_id: str

    agent_name: str

    status: str

    answer: (
        str | None
    ) = None

    proposed_tool: (
        str | None
    ) = None

    proposed_arguments: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    error: (
        str | None
    ) = None

    approval_id: (
        str | None
    ) = None


@dataclass
class HubResult:
    status: str

    user_request: str

    routes: list[str] = field(
        default_factory=list
    )

    results: list[
        AgentResult
    ] = field(
        default_factory=list
    )

    answer: (
        str | None
    ) = None

    error: (
        str | None
    ) = None