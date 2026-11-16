"""Narrow, observable repair of a routing plan's format, never of its content.

The local Hub sometimes writes a complete plan in a slightly wrong format: the
closing brackets at the very end in the wrong order, an intent without the
fields whose only valid value the contract fixes, or one argument value written
without its list. Each repair here has exactly one possible result and cannot
change an agent, capability, argument value, effect or clarification. Anything else stays malformed and takes the one trusted repair
generation. The caller records every repair in the routing trace.
"""
from __future__ import annotations

import json
from typing import Any

_CLOSER = {"{": "}", "[": "]"}


def _closing_order(text: str) -> str | None:
    """The text with its trailing closers in the only valid order, or None.

    Applies only when everything from the first unmatched closer on is closers
    and there are at least as many of each kind as the open containers need:
    every key and value was already written inside the container that was open
    when it was written, so only one order of the final closers is valid. Fewer
    closers could mean a cut-off plan; that is never completed here.
    """
    stack: list[str] = []
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in _CLOSER:
            stack.append(char)
        elif char in "}]":
            if stack and _CLOSER[stack[-1]] == char:
                stack.pop()
                continue
            tail = "".join(text[index:].split())
            needed = "".join(_CLOSER[opener] for opener in reversed(stack))
            if (set(tail) - set("}]")
                    or any(tail.count(closer) < needed.count(closer) for closer in "}]")):
                return None
            return text[:index] + needed
    return None


def _intent_constants(plan: Any) -> list[str]:
    """Add only the omitted intent fields that have one valid value.

    max_tool_calls is always 1 in the current one-call runtime. An intent that
    names exactly one capability with a read or mutation effect is, by the
    protocol, not a clarification (a clarification names none and uses effect
    "unknown"); missing required values still force clarification later.
    Present values are never replaced.
    """
    filled: list[str] = []
    delegations = plan.get("delegations") if isinstance(plan, dict) else None
    for delegation in delegations if isinstance(delegations, list) else ():
        intent = delegation.get("intent") if isinstance(delegation, dict) else None
        if not isinstance(intent, dict):
            continue
        if "max_tool_calls" not in intent:
            intent["max_tool_calls"] = 1
            filled.append("max_tool_calls")
        tools, effect = intent.get("allowed_tools"), intent.get("effect")
        if ("clarification_required" not in intent and isinstance(tools, list) and len(tools) == 1
                and isinstance(effect, str) and effect.strip().lower() in {"read", "mutation"}):
            intent["clarification_required"] = False
            filled.append("clarification_required")
    return filled


def _argument_lists(plan: Any) -> list[str]:
    """Wrap a lone non-empty string argument value in its one-element list.

    The contract maps each argument name to a list of exact strings; a single
    string has exactly one list form. The value itself is unchanged and is
    still checked against the user's message. Any other non-list value is left
    for the contract to reject.
    """
    wrapped: list[str] = []
    delegations = plan.get("delegations") if isinstance(plan, dict) else None
    for delegation in delegations if isinstance(delegations, list) else ():
        intent = delegation.get("intent") if isinstance(delegation, dict) else None
        if not isinstance(intent, dict):
            continue
        for field in ("allowed_arguments", "forbidden_arguments"):
            arguments = intent.get(field)
            if not isinstance(arguments, dict):
                continue
            for name, value in arguments.items():
                if isinstance(value, str) and value.strip():
                    arguments[name] = [value]
                    wrapped.append("argument_list")
    return wrapped


def parse_routing_plan(response: str) -> tuple[Any, tuple[str, ...]]:
    """Parse a Hub routing response; return the plan and the format repairs made.

    Raises the original JSONDecodeError for anything the narrow repairs do not
    cover, so the trusted one-repair path and its feedback stay unchanged.
    """
    repairs: list[str] = []
    try:
        plan = json.loads(response)
    except json.JSONDecodeError as original:
        reordered = _closing_order(response)
        if reordered is None:
            raise
        try:
            plan = json.loads(reordered)
        except json.JSONDecodeError:
            raise original from None
        repairs.append("closing_order")
    repairs.extend(_intent_constants(plan))
    repairs.extend(_argument_lists(plan))
    return plan, tuple(dict.fromkeys(repairs))
