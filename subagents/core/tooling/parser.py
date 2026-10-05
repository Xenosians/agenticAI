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
        normalized = response.strip()

        # Narrow deterministic recovery for one observed small-model
        # serialization failure:
        #
        #   [{"arguments": {...}, "name": "tool"}
        #
        # The object itself is complete, but the outer JSON array is
        # missing exactly one closing bracket.
        #
        # We never infer or modify:
        #   - tool name
        #   - arguments
        #   - braces
        #   - quotes
        #   - semantic content
        #
        # Existing validation below remains authoritative.
        repaired = None

        if (
            normalized.startswith("[")
            and normalized.endswith("}")
        ):
            candidate = normalized + "]"

            try:
                candidate_parsed = json.loads(
                    candidate
                )
            except json.JSONDecodeError:
                candidate_parsed = None

            if isinstance(
                candidate_parsed,
                list,
            ):
                repaired = candidate_parsed

        if repaired is None:
            raise ValueError(
                "Worker returned invalid JSON."
            ) from exc

        parsed = repaired

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