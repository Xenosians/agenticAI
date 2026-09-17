from typing import (
    Any,
)

from tools.ticketing.presentation import (
    build_ticket_comments_card,
    build_ticket_get_card,
    build_ticket_history_card,
    build_ticket_search_card,
    format_ticket_comments_result,
    format_ticket_get_result,
    format_ticket_history_result,
    format_ticket_search_result,
)


def format_ticket_add_comment_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    ticket_key = (
        arguments.get(
            "ticket_key"
        )
    )

    if (
        isinstance(
            ticket_key,
            str,
        )
        and ticket_key.strip()
    ):

        return (
            "Adding a comment to "
            f"{ticket_key} requires approval."
        )

    return (
        "Adding the ticket comment "
        "requires approval."
    )


TICKETING_TOOLS = {
    # ============================================================
    # CURRENT TICKET STATE
    # ============================================================

    "ticket_get": {
        "description": (
            "Retrieve the current metadata and state of exactly "
            "one ticket, such as summary, status, type, priority, "
            "assignee, reporter, project, and timestamps. "
            "This capability does not retrieve ticket comments "
            "or ticket change history."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "ticket_key",
        ],

        "parameters": {
            "ticket_key": {
                "type":
                    "str",

                "description": (
                    "Exact ticket identifier supplied "
                    "by the user."
                ),
            },
        },

        "result_formatter": (
            format_ticket_get_result
        ),

        "presentation_builder": (
            build_ticket_get_card
        ),
    },

    # ============================================================
    # SEARCH
    # ============================================================

    "ticket_search": {
        "description": (
            "Search for a bounded collection of tickets using "
            "structured filters such as project, status, priority, "
            "or text. Use this capability when the user wants a "
            "set of matching tickets rather than one exact ticket."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "project_key",
        ],

        "parameters": {
            "text": {
                "type":
                    "str",

                "description": (
                    "Optional text to search for "
                    "inside ticket content."
                ),
            },

            "project_key": {
                "type":
                    "str",

                "description": (
                    "Optional exact project key "
                    "supplied by the user."
                ),
            },

            "status": {
                "type":
                    "str",

                "description": (
                    "Optional ticket status filter."
                ),
            },

            "priority": {
                "type":
                    "str",

                "description": (
                    "Optional ticket priority filter."
                ),
            },

            "limit": {
                "type":
                    "int",

                "description": (
                    "Maximum number of tickets "
                    "to return from 1 to 25."
                ),
            },
        },

        "result_formatter": (
            format_ticket_search_result
        ),

        "presentation_builder": (
            build_ticket_search_card
        ),
    },

    # ============================================================
    # CHANGE HISTORY
    # ============================================================

    "ticket_history": {
        "description": (
            "Retrieve the recorded change history for exactly "
            "one ticket, including changed fields, previous values, "
            "new values, authors, and timestamps. "
            "Use this capability when the requested information "
            "concerns how a ticket changed over time rather than "
            "only its current state."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "ticket_key",
        ],

        "parameters": {
            "ticket_key": {
                "type":
                    "str",

                "description": (
                    "Exact ticket identifier supplied "
                    "by the user."
                ),
            },

            "limit": {
                "type":
                    "int",

                "description": (
                    "Maximum number of history entries "
                    "to return from 1 to 50."
                ),
            },
        },

        "result_formatter": (
            format_ticket_history_result
        ),

        "presentation_builder": (
            build_ticket_history_card
        ),
    },

    # ============================================================
    # READ COMMENTS
    # ============================================================

    "ticket_comments": {
        "description": (
            "Retrieve the comments or discussion entries attached "
            "to exactly one ticket, including comment text, author, "
            "and timestamps. Use this capability only for reading "
            "existing comments."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "ticket_key",
        ],

        "parameters": {
            "ticket_key": {
                "type":
                    "str",

                "description": (
                    "Exact ticket identifier supplied "
                    "by the user."
                ),
            },

            "limit": {
                "type":
                    "int",

                "description": (
                    "Maximum number of comments "
                    "to return from 1 to 50."
                ),
            },
        },

        "result_formatter": (
            format_ticket_comments_result
        ),

        "presentation_builder": (
            build_ticket_comments_card
        ),
    },

    # ============================================================
    # ADD COMMENT
    #
    # This capability changes external provider state.
    # ToolGateway approval remains mandatory.
    # ============================================================

    "ticket_add_comment": {
        "description": (
            "Add exactly one user-supplied comment or note to "
            "exactly one existing ticket. This changes ticket "
            "provider state. Use only when the user explicitly "
            "asks to add, post, append, or record a comment or "
            "note. Do not use this capability merely to read "
            "existing comments."
        ),

        "risk":
            "low",

        "requires_approval":
            True,

        "grounded_arguments": [
            "ticket_key",
            "comment",
        ],

        "parameters": {
            "ticket_key": {
                "type":
                    "str",

                "description": (
                    "Exact ticket identifier supplied "
                    "by the user."
                ),
            },

            "comment": {
                "type":
                    "str",

                "description": (
                    "Exact comment text requested by the user. "
                    "Do not summarize, rewrite, embellish, or "
                    "invent ticket comment content."
                ),
            },
        },

        "approval_formatter": (
            format_ticket_add_comment_approval
        ),
    },
}
