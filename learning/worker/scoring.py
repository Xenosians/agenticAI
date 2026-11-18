"""Held-out scoring and the tool-call reward, with Janus's rules.

``parse_calls`` and ``arguments_match`` copy ``subagents.core.tooling.parser.parse_tool_calls`` and
``learning.evaluation.config_specialist_gate._arguments_match``; tests pin them together. A no-tool case
answered with one tool call is judged by Janus's runtime policy, which the worker does not have: the
worker counts it as failed and marks it for the host, which re-scores every result with the real gate.
"""
from __future__ import annotations

import json
from typing import Any

NO_TOOL = frozenset({"", "none", "__none__", "clarify", "reject", "refuse", "unsupported"})


def parse_calls(response: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as exc:
        normalized = response.strip()
        repaired = None
        # The one repair Janus allows: a complete object whose outer array lacks its closing bracket.
        if normalized.startswith("[") and normalized.endswith("}"):
            try:
                candidate = json.loads(normalized + "]")
            except json.JSONDecodeError:
                candidate = None
            if isinstance(candidate, list):
                repaired = candidate
        if repaired is None:
            raise ValueError("Worker returned invalid JSON.") from exc
        parsed = repaired
    if not isinstance(parsed, list):
        raise ValueError("Worker response must be a JSON array.")
    if not parsed:
        raise ValueError("Worker returned an empty tool-call list.")
    calls = []
    for call in parsed:
        if not isinstance(call, dict):
            raise ValueError("Each tool call must be a JSON object.")
        name, arguments = call.get("name"), call.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError("Tool call is missing a valid name.")
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool '{name}' has invalid arguments.")
        calls.append({"name": name, "arguments": arguments})
    return calls


def arguments_match(case: dict[str, Any], observed: Any) -> bool:
    if not isinstance(observed, dict):
        return False
    expected = case.get("expected_arguments") or {}
    for key, value in expected.items():
        if observed.get(key) != value:
            return False
    extras = set(observed) - set(expected)
    if not extras.issubset(set(case.get("allowed_extra_arguments") or [])):
        return False
    allowed = case.get("allowed_arguments") or {}
    for key in extras:
        values = allowed.get(key)
        if values and observed.get(key) not in values:
            return False
    return True


def score_case(case: dict[str, Any], raw: str) -> dict[str, Any]:
    expected_tool = str(case.get("expected_tool") or "").strip()
    try:
        calls, error = parse_calls(raw), None
    except ValueError as exc:
        calls, error = [], str(exc)
    result = {"name": case.get("name"), "raw": raw[:2000], "error": error, "observed_tool": None,
              "observed_arguments": None, "needs_host_policy": False}
    if expected_tool.lower() in NO_TOOL:
        if not calls:
            return result | {"passed": True}
        if len(calls) == 1:
            return result | {"passed": False, "needs_host_policy": True, "observed_tool": calls[0]["name"],
                             "observed_arguments": calls[0]["arguments"],
                             "error": "a proposal for a no-tool case is judged by the host's runtime policy"}
        return result | {"passed": False, "error": f"unsafe/unsupported case produced {len(calls)} tool calls"}
    if len(calls) != 1:
        return result | {"passed": False, "error": error or f"expected exactly one tool call; got {len(calls)}"}
    call = calls[0]
    return result | {"passed": call["name"] == expected_tool and arguments_match(case, call["arguments"]),
                     "observed_tool": call["name"], "observed_arguments": call["arguments"]}


def tool_call_reward(expected: str, completion: str) -> float:
    """1.0 for exactly the expected calls (or no call when none is expected), 0.5 for the right tools with
    wrong arguments, 0.0 otherwise."""
    try:
        wanted = parse_calls(expected)
    except ValueError:
        wanted = []
    try:
        produced = parse_calls(completion)
    except ValueError:
        produced = []
    if not wanted:
        return 1.0 if not produced else 0.0
    if produced == wanted:
        return 1.0
    if [call["name"] for call in produced] == [call["name"] for call in wanted]:
        return 0.5
    return 0.0
