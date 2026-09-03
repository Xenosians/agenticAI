import json

from tools.registry import get_tool

from subagents.core.types import AgentDefinition
from subagents.prompts.prompt_loader import load_prompt


def build_worker_system_prompt(
    agent: AgentDefinition,
) -> str:
    """
    Build the worker system prompt including only
    the tools explicitly allowed for that agent.
    """

    tool_specs = []

    for tool_name in agent.tools:
        tool = get_tool(tool_name)

        if tool is None:
            raise ValueError(
                f"Agent '{agent.name}' references "
                f"unknown tool '{tool_name}'."
            )

        tool_specs.append(
            {
                "name": tool_name,
                "description": tool["description"],
                "parameters": tool.get(
                    "parameters",
                    {},
                ),
            }
        )

    tool_json = json.dumps(
        tool_specs,
        indent=2,
    )

    template = load_prompt(
        "worker_tool_protocol.txt"
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