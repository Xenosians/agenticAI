import json
from typing import Any


def parse_tool_calls(
    response: str,
) -> list[dict[str, Any]]:
    """
    Parse structured tool calls produced by a worker model.
    """

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Worker returned invalid JSON."
        ) from exc

    if not isinstance(parsed, list):
        raise ValueError(
            "Worker response must be a JSON array."
        )

    if not parsed:
        raise ValueError(
            "Worker returned an empty tool-call list."
        )

    validated_calls = []

    for call in parsed:
        if not isinstance(call, dict):
            raise ValueError(
                "Each tool call must be a JSON object."
            )

        name = call.get("name")
        arguments = call.get("arguments")

        if not isinstance(name, str) or not name:
            raise ValueError(
                "Tool call is missing a valid name."
            )

        if not isinstance(arguments, dict):
            raise ValueError(
                f"Tool '{name}' has invalid arguments."
            )

        validated_calls.append(
            {
                "name": name,
                "arguments": arguments,
            }
        )

    return validated_calls