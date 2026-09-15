from __future__ import annotations

import json

from subagents.core.capabilities import (
    build_agent_capability_catalog,
)

from subagents.core.types import (
    AgentDefinition,
)

from subagents.prompts.prompt_loader import (
    load_prompt,
)


def build_worker_system_prompt(
    agent: AgentDefinition,
) -> str:
    """
    Build the worker prompt from the specialist definition and
    trusted runtime capability catalog.

    The specialist definition describes behavioral role/context.

    Capability truth comes from the trusted tool registry.

    The worker therefore reasons over what capabilities actually
    exist instead of relying on capability descriptions manually
    copied into prompts.
    """

    capabilities = (
        build_agent_capability_catalog(
            agent,
            include_arguments=True,
        )
    )

    capability_json = (
        json.dumps(
            capabilities,
            indent=2,
        )
    )

    template = (
        load_prompt(
            "worker_tool_protocol.txt"
        )
    )

    return (
        template
        .replace(
            "{{AGENT_SYSTEM_PROMPT}}",
            agent.system_prompt,
        )
        .replace(
            "{{TOOLS_JSON}}",
            capability_json,
        )
    )