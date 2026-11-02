"""The requesting owner's reference passages for a knowledge search.

Runs on the host after the knowledge provider answers, inside the request's
trusted owner context; the model never names the owner. Passages are quoted
reference text: they never grant permission or change routing. Their citations
are recorded with the answer like any other passage a model was given.
"""
from __future__ import annotations

from typing import Any

MAX_OWNER_PASSAGES = 3


def owner_knowledge_search(result: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """``result`` with the owner's matching reference passages, when there are any.

    Reference datasets are knowledge, not runbooks, so a runbook-only search
    adds nothing. Outside an owner's request nothing is added.
    """
    from services.extensions.context import current_owner, disclose_references

    owner = current_owner()
    query = arguments.get("query")
    if owner is None or not isinstance(query, str) or not query.strip() or arguments.get("kind") == "runbook":
        return result
    from services.extensions.service import extension_store

    hits = extension_store().search_reference(owner, query, limit=MAX_OWNER_PASSAGES)
    if not hits:
        return result
    disclose_references(hits)
    passages = [{"source": hit["source"], "text": hit["content"], "trust": "untrusted_reference"} for hit in hits]
    count = result.get("count") if isinstance(result.get("count"), int) else len(result.get("hits") or [])
    return {**result, "reference_passages": passages, "count": count + len(passages)}
