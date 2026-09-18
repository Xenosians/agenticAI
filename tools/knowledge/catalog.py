from typing import (
    Any,
)


def format_knowledge_search_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    hits = (
        result.get(
            "hits"
        )
    )

    if not isinstance(
        hits,
        list,
    ):

        return (
            "The knowledge search completed, but "
            "no readable result list was returned."
        )

    if not hits:

        return (
            "No knowledge documents or runbooks "
            "matched the search."
        )

    lines = []

    for hit in hits:

        if not isinstance(
            hit,
            dict,
        ):

            continue

        document_id = (
            hit.get(
                "document_id",
                "Unknown",
            )
        )

        kind = (
            hit.get(
                "kind",
                "knowledge",
            )
        )

        title = (
            hit.get(
                "title",
                "Untitled",
            )
        )

        summary = (
            hit.get(
                "summary",
                "",
            )
        )

        lines.append(
            f"- {document_id} "
            f"[{kind}] — {title}: "
            f"{summary}"
        )

    return (
        "Knowledge search results:\n"
        + "\n".join(
            lines
        )
    )


def format_knowledge_document_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    document = (
        result.get(
            "document"
        )
    )

    if not isinstance(
        document,
        dict,
    ):

        return (
            "The knowledge lookup completed, but "
            "no readable document was returned."
        )

    document_id = (
        document.get(
            "document_id",
            "Unknown",
        )
    )

    title = (
        document.get(
            "title",
            "Untitled",
        )
    )

    content = (
        document.get(
            "content",
            "",
        )
    )

    return (
        f"{document_id} — {title}\n\n"
        f"{content}"
    )


KNOWLEDGE_TOOLS = {
    "knowledge_search": {
        "description": (
            "Search trusted ITSM knowledge and runbook metadata "
            "using a bounded natural-language query. This is a "
            "read-only retrieval capability."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments":
            [],

        "parameters": {
            "query": {
                "type":
                    "str",

                "description": (
                    "Natural-language search text describing "
                    "the knowledge needed."
                ),
            },

            "kind": {
                "type":
                    "str",

                "description": (
                    "Optional result type filter: "
                    "'knowledge' or 'runbook'."
                ),
            },

            "limit": {
                "type":
                    "int",

                "description": (
                    "Maximum number of results "
                    "to return from 1 to 25."
                ),
            },
        },

        "result_formatter":
            format_knowledge_search_result,
    },

    "knowledge_get": {
        "description": (
            "Retrieve exactly one trusted knowledge article by "
            "its explicit document identifier. Use only for "
            "knowledge articles, not runbooks."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "document_id",
        ],

        "parameters": {
            "document_id": {
                "type":
                    "str",

                "description": (
                    "Exact knowledge document identifier "
                    "supplied by the user."
                ),
            },
        },

        "result_formatter":
            format_knowledge_document_result,
    },

    "runbook_get": {
        "description": (
            "Retrieve exactly one trusted operational runbook by "
            "its explicit runbook identifier. This capability "
            "only reads the runbook and does not execute its steps."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "runbook_id",
        ],

        "parameters": {
            "runbook_id": {
                "type":
                    "str",

                "description": (
                    "Exact runbook identifier supplied by the user."
                ),
            },
        },

        "result_formatter":
            format_knowledge_document_result,
    },
}
