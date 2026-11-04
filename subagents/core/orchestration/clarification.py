"""A reply to Janus's own clarification completes the request it asked about.

Janus binds an authority-bearing value (an account name, a project key) only
when the user wrote it in the request being answered. When Janus answers with a
clarification ("Please provide the required family name") and the user replies
with the missing detail ("Doe"), the earlier values ("John") are not in the
reply, so the request is grounded in the chain of the user's own messages
instead: the consecutive requests Janus answered with a saved clarification,
ending with the current reply.

Only user-authored text joins the chain. An assistant answer never does, and a
turn counts as a clarification only by its saved result record (status
``clarification_required``), never by its wording. Any other answer ends the
chain. The chain is bounded, and a change still needs approval of its exact
values.
"""
from __future__ import annotations

from typing import Any

# The current reply plus at most this many earlier requests in total.
CHAIN_LIMIT = 4


def _asked_for_clarification(turn: dict[str, Any]) -> bool:
    """Whether an assistant turn's saved result record is a clarification, and only that."""
    evidence = turn.get("result_evidence")
    if not isinstance(evidence, dict) or "approval_execution" in evidence:
        return False
    results = evidence.get("results")
    return (
        isinstance(results, list) and bool(results)
        and all(isinstance(item, dict) and item.get("status") == "clarification_required" for item in results)
    )


def clarification_chain(user_request: str, context: list[dict[str, Any]]) -> list[str]:
    """The user's own requests this reply completes, oldest first, ending with it.

    ``context`` is the normalized prior conversation (oldest first). Without an
    open clarification the current request stands alone.
    """
    chain = [user_request]
    index = len(context) - 1
    while len(chain) < CHAIN_LIMIT and index >= 1:
        answer, request = context[index], context[index - 1]
        if (answer.get("role") != "assistant" or not _asked_for_clarification(answer)
                or request.get("role") != "user"):
            break
        chain.insert(0, request["content"])
        index -= 2
    return chain


def grounding_request(user_request: str, context: list[dict[str, Any]]) -> str:
    """The text a reply's values are grounded in: the chain, one request per line."""
    return "\n".join(clarification_chain(user_request, context))
