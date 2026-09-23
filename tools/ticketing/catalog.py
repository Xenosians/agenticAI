from typing import (
    Any,
)

from tools.ticketing.policy import (
    evaluate_ticket_create_policy,
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


def _ticket_key_approval(
    arguments: dict[
        str,
        Any,
    ],
    *,
    action: str,
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
            f"{action} {ticket_key} "
            "requires approval."
        )

    return (
        f"{action} the ticket "
        "requires approval."
    )


def format_ticket_add_comment_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    return (
        _ticket_key_approval(
            arguments,
            action=(
                "Adding a comment to"
            ),
        )
    )


def format_ticket_assign_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    return (
        _ticket_key_approval(
            arguments,
            action=(
                "Assigning"
            ),
        )
    )


def format_ticket_transition_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    return (
        _ticket_key_approval(
            arguments,
            action=(
                "Changing the status of"
            ),
        )
    )


def format_ticket_create_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    project_key = (
        arguments.get(
            "project_key"
        )
    )

    if (
        isinstance(
            project_key,
            str,
        )
        and project_key.strip()
    ):

        return (
            "Creating a ticket in project "
            f"{project_key} requires approval."
        )

    return (
        "Creating the ticket requires approval."
    )


TICKETING_TOOLS = {
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

    "ticket_search": {
        "description": (
            "Search for a bounded collection of tickets or issues "
            "using structured filters such as project, status, "
            "priority, or text. The returned and primary resources "
            "are tickets/issues. A project key is only a scope or "
            "filter on the ticket collection; it does not turn this "
            "into a project-list operation. Use this when the user "
            "wants multiple matching tickets rather than one exact "
            "ticket."
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
                    "Optional text to search inside ticket content."
                ),
            },

            "project_key": {
                "type":
                    "str",

                "description": (
                    "Optional exact project key supplied "
                    "by the user."
                ),
            },

            "status": {
                "type":
                    "str",

                "description":
                    "Optional ticket status filter.",
            },

            "priority": {
                "type":
                    "str",

                "description":
                    "Optional ticket priority filter.",
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

    "ticket_history": {
        "description": (
            "Retrieve the recorded change history for exactly "
            "one ticket, including changed fields, previous values, "
            "new values, authors, and timestamps."
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

    "ticket_comments": {
        "description": (
            "Retrieve comments or discussion entries attached "
            "to exactly one ticket. Use only for reading existing "
            "comments."
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

    "ticket_add_comment": {
        "description": (
            "Add exactly one user-supplied comment or note to "
            "exactly one existing ticket. This changes provider "
            "state. Do not use merely to read comments."
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
                    "Exact comment text supplied by the user. "
                    "Do not summarize, rewrite, or invent it."
                ),
            },
        },

        "approval_formatter": (
            format_ticket_add_comment_approval
        ),
    },

    "ticket_create": {
        "description": (
            "Create exactly one new ticket in an explicitly "
            "specified project using the exact user-supplied "
            "summary. An optional ticket type may be supplied "
            "only when the user explicitly specifies it."
        ),

        "risk":
            "medium",

        "requires_approval":
            True,

        "policy_resolver":
            evaluate_ticket_create_policy,

        "trusted_policy_arguments": [
            "expected_project_id",
            "expected_project_key",
            "expected_project_name",
            "expected_ticket_type_id",
            "expected_ticket_type_name",
        ],

        "grounded_arguments": [
            "project_key",
            "summary",
            "ticket_type",
        ],

        "parameters": {
            "project_key": {
                "type":
                    "str",

                "description": (
                    "Exact project key supplied by the user."
                ),
            },

            "summary": {
                "type":
                    "str",

                "description": (
                    "Exact requested ticket summary/title. "
                    "Do not rewrite or invent it."
                ),
            },

            "ticket_type": {
                "type":
                    "str",

                "description": (
                    "Optional ticket type explicitly supplied "
                    "by the user."
                ),
            },
        },

        "approval_formatter": (
            format_ticket_create_approval
        ),
    },

    "ticket_assign": {
        "description": (
            "Assign exactly one existing ticket to exactly one "
            "explicitly supplied assignee identifier."
        ),

        "risk":
            "low",

        "requires_approval":
            True,

        "grounded_arguments": [
            "ticket_key",
            "assignee",
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

            "assignee": {
                "type":
                    "str",

                "description": (
                    "Exact assignee identifier supplied by "
                    "the user. For Jira deployments this must "
                    "resolve as a Jira account identifier."
                ),
            },
        },

        "approval_formatter": (
            format_ticket_assign_approval
        ),
    },

    "ticket_transition": {
        "description": (
            "Change exactly one existing ticket to an explicitly "
            "requested workflow status. The trusted provider "
            "adapter resolves the provider-native transition."
        ),

        "risk":
            "low",

        "requires_approval":
            True,

        "grounded_arguments": [
            "ticket_key",
            "status",
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

            "status": {
                "type":
                    "str",

                "description": (
                    "Exact requested destination status supplied "
                    "by the user."
                ),
            },
        },

        "approval_formatter": (
            format_ticket_transition_approval
        ),
    },
}
