import json

from tools.registry import get_tool

from subagents.core.types import AgentDefinition


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

    return (
        f"{agent.system_prompt}\n\n"
        "FUNCTION CALLING PROTOCOL:\n"
        "Given the user's request and the available tools below, "
        "respond with ONLY a JSON array containing the required "
        "function call.\n\n"
        "Each call must contain:\n"
        '- "name": the exact tool name\n'
        '- "arguments": a JSON object containing the arguments\n\n'
        "Do not output markdown or explanations.\n"
        "Do not invent identifiers.\n"
        "Use only the tools listed below.\n\n"
        "AVAILABLE TOOLS:\n"
        f"{tool_json}"
    )