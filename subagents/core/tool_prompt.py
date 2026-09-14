import json

from tools.registry import (
    get_tool,
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
    Build the worker system prompt containing only the
    capabilities explicitly allowed for the specialist.

    Registry metadata remains the trusted source of tool
    descriptions and argument schemas.

    The model receives a simplified capability representation
    rather than the internal registry structure.
    """

    tool_specs = []

    for tool_name in (
        agent.tools
    ):
        tool = (
            get_tool(
                tool_name
            )
        )

        if tool is None:
            raise ValueError(
                f"Agent '{agent.name}' references "
                f"unknown tool '{tool_name}'."
            )

        description = (
            tool.get(
                "description"
            )
        )

        if not isinstance(
            description,
            str,
        ):
            raise ValueError(
                f"Tool '{tool_name}' has "
                "an invalid description."
            )

        argument_schema = (
            tool.get(
                "parameters",
                {},
            )
        )

        if not isinstance(
            argument_schema,
            dict,
        ):
            raise ValueError(
                f"Tool '{tool_name}' has "
                "an invalid argument schema."
            )

        tool_specs.append(
            {
                "name":
                    tool_name,

                "description":
                    description,

                "argument_schema":
                    argument_schema,
            }
        )

    tool_json = (
        json.dumps(
            tool_specs,
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
            tool_json,
        )
    )