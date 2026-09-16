from __future__ import annotations

import json

from typing import (
    Any,
)

from subagents.core.tooling.capabilities import (
    build_agent_capability_catalog,
)

from subagents.core.definitions.types import (
    AgentDefinition,
)

from subagents.prompts.prompt_loader import (
    load_prompt,
)


def build_worker_system_prompt(
    agent: AgentDefinition,
    *,
    capability_catalog: (
        list[
            dict[
                str,
                Any,
            ]
        ]
        | None
    ) = None,
) -> str:
    """
    Build the worker prompt from the specialist definition and
    trusted runtime capability catalog.

    Callers may provide a prebuilt capability catalog.

    AgentRuntime does this deliberately so the exact SAME catalog
    object is used for:

        prompt construction
        provenance hashing

    This prevents dynamic capability metadata from being resolved
    independently at two different moments.

    Other callers remain backward compatible: when no catalog is
    supplied, it is built normally.
    """

    capabilities = (
        capability_catalog

        if capability_catalog
        is not None

        else (
            build_agent_capability_catalog(
                agent,
                include_arguments=True,
            )
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