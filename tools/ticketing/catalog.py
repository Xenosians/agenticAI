from typing import (
    Any,
)

from tools.ticketing.policy import (
    evaluate_ticket_add_comment_policy,
    evaluate_ticket_assign_policy,
    evaluate_ticket_create_policy,
    evaluate_ticket_link_policy,
    evaluate_ticket_transition_policy,
    evaluate_ticket_update_policy,
    evaluate_ticket_worklog_add_policy,
)

from tools.ticketing.presentation import (
    build_ticket_assigned_card,
    build_ticket_comments_card,
    build_ticket_get_card,
    build_ticket_history_card,
    build_ticket_links_card,
    build_ticket_search_card,
    format_ticket_assigned_result,
    format_ticket_comments_result,
    format_ticket_get_result,
    format_ticket_history_result,
    format_ticket_links_result,
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


def _change_line(
    field: str,
    old: Any,
    new: Any,
) -> str:
    return f"{field}: {old if old not in (None, '') else '(none)'} → {new}"


def format_ticket_update_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:
    """List each requested change as old → new, using the frozen snapshot."""

    ticket_key = arguments.get("ticket_key") or "the ticket"
    changes = []

    # Without a provider snapshot (non-Jira providers) the old value is
    # unknown, so only the requested value is shown.
    priority = arguments.get("priority")
    if priority is not None:
        changes.append(
            _change_line(
                "priority",
                arguments.get("expected_priority_name"),
                arguments.get("expected_target_priority_name") or priority,
            )
            if "expected_priority_name" in arguments
            else f"priority → {priority}"
        )

    labels = arguments.get("expected_labels")
    add_labels = arguments.get("add_labels") or []
    remove_labels = arguments.get("remove_labels") or []
    if add_labels or remove_labels:
        if isinstance(labels, list):
            final = [label for label in labels if label not in remove_labels]
            final.extend(label for label in add_labels if label not in final)
            changes.append(
                _change_line(
                    "labels",
                    ", ".join(labels),
                    ", ".join(sorted(final)) or "(none)",
                )
            )
        else:
            if add_labels:
                changes.append("labels: add " + ", ".join(add_labels))
            if remove_labels:
                changes.append("labels: remove " + ", ".join(remove_labels))

    due_date = arguments.get("due_date")
    if due_date is not None:
        changes.append(
            _change_line(
                "due date",
                arguments.get("expected_due_date"),
                due_date,
            )
            if "expected_due_date" in arguments
            else f"due date → {due_date}"
        )

    if not changes:
        return f"Updating {ticket_key} requires approval."

    return (
        f"Updating {ticket_key} requires approval: "
        + "; ".join(changes)
        + "."
    )


def format_ticket_link_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:
    ticket_key = arguments.get("ticket_key")
    other_key = arguments.get("other_ticket_key")
    phrase = (
        arguments.get("expected_link_phrase")
        or arguments.get("relationship")
    )

    if not all(isinstance(value, str) and value.strip() for value in (ticket_key, other_key, phrase)):
        return "Linking the tickets requires approval."

    sentence = f"{ticket_key} {phrase} {other_key}"

    if arguments.get("expected_link_exists") is True:
        return (
            f"Linking requires approval: {sentence}. "
            "This link already exists, so nothing will change."
        )

    return f"Linking requires approval: {sentence}."


def format_ticket_worklog_add_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:
    ticket_key = arguments.get("ticket_key") or "the ticket"
    time_spent = arguments.get("time_spent") or "time"
    comment = arguments.get("comment")

    message = f"Logging {time_spent} of work on {ticket_key} requires approval"

    if isinstance(comment, str) and comment.strip():
        message += f", with the comment: {comment}"

    return message + "."


def format_ticket_mutation_changes_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    message = result.get("message") or result.get("error") or "Ticket update finished."
    changes = result.get("changes")

    lines = [
        _change_line(
            change.get("field"),
            change.get("from_value"),
            change.get("to_value") if change.get("to_value") is not None else "(none)",
        )
        for change in (changes if isinstance(changes, list) else [])
        if isinstance(change, dict) and isinstance(change.get("field"), str)
    ]

    if not lines:
        return message

    return message + "\n" + "\n".join(lines)


TICKET_KEY_PARAMETER = {
    "type":
        "str",

    "description": (
        "Exact ticket identifier supplied "
        "by the user."
    ),
}


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

        "required_arguments": [
            "ticket_key",
        ],

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
            "Search or list a bounded collection of TICKETS / ISSUES. "
            "Use this when the user asks to find, list, search, or show "
            "multiple tickets/issues, including requests such as tickets "
            "inside one explicitly named project. Structured filters may "
            "include project, status, priority, or text. The returned and "
            "PRIMARY resources are tickets/issues. A project key is only a "
            "scope or filter on the ticket collection; it does not turn this "
            "into a project-list operation. Do NOT reinterpret a ticket search "
            "as project lookup or project listing merely because the project "
            "key is explicit."
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

        "required_arguments": [
            "ticket_key",
        ],

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

        "required_arguments": [
            "ticket_key",
        ],

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
            "medium",

        "requires_approval":
            True,

        "policy_resolver":
            evaluate_ticket_add_comment_policy,

        "trusted_policy_arguments": [
            "expected_issue_id",
            "expected_issue_key",
            "expected_issue_summary",
            "expected_project_id",
            "expected_project_key",
            "expected_project_name",
        ],

        "required_arguments": [
            "ticket_key",
            "comment",
        ],

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
            "CREATE exactly one new TICKET / ISSUE inside an explicitly "
            "specified existing project. Preserve an exact user-supplied "
            "summary; otherwise compose its derived summary from the "
            "requested outcome and explicitly supplied verified workflow "
            "content, if present. If the user says create a Task, Bug, Story, issue, "
            "or ticket in project X, the PRIMARY resource being created "
            "is the ticket/issue and project X is its containing scope. "
            "Do NOT reinterpret this as project creation or project lookup. "
            "An optional ticket type may be supplied only when the user "
            "explicitly specifies it."
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

        "required_arguments": [
            "project_key",
            "summary",
        ],

        "grounded_arguments": [
            "project_key",
            "ticket_type",
        ],

        "derived_arguments": [
            "summary",
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
                    "Preserve an exact user-supplied summary. Otherwise "
                    "compose concise content from the requested outcome; "
                    "exact facts explicitly supplied in runtime VERIFIED "
                    "WORKFLOW CONTENT may be included as content only. "
                    "Do not invent facts, scope, participants, goals, "
                    "identifiers, dates, or provider state."
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
            "medium",

        "requires_approval":
            True,

        "policy_resolver":
            evaluate_ticket_assign_policy,

        "trusted_policy_arguments": [
            "expected_issue_id",
            "expected_issue_key",
            "expected_issue_summary",
            "expected_project_id",
            "expected_project_key",
            "expected_project_name",
            "expected_previous_assignee_present",
            "expected_previous_assignee_account_id",
            "expected_previous_assignee_display_name",
            "expected_assignee_account_id",
            "expected_assignee_display_name",
        ],

        "required_arguments": [
            "ticket_key",
            "assignee",
        ],

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
            "medium",

        "requires_approval":
            True,

        "policy_resolver":
            evaluate_ticket_transition_policy,

        "trusted_policy_arguments": [
            "expected_issue_id",
            "expected_issue_key",
            "expected_issue_summary",
            "expected_project_id",
            "expected_project_key",
            "expected_project_name",
            "expected_source_status_id",
            "expected_source_status_name",
            "expected_transition_noop",
            "expected_transition_id",
            "expected_transition_name",
            "expected_target_status_id",
            "expected_target_status_name",
        ],

        "required_arguments": [
            "ticket_key",
            "status",
        ],

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

    "ticket_update": {
        "description": (
            "Change the priority, labels and/or due date of exactly one "
            "existing ticket, using only values the user supplied: an exact "
            "priority name, exact labels to add or remove, and an exact "
            "YYYY-MM-DD due date. At least one change is required. This does "
            "NOT change status (use the transition capability), assignee, "
            "summary, or comments, and it never replaces the whole label set."
        ),

        "risk":
            "medium",

        "requires_approval":
            True,

        "policy_resolver":
            evaluate_ticket_update_policy,

        "trusted_policy_arguments": [
            "expected_issue_id",
            "expected_issue_key",
            "expected_issue_summary",
            "expected_project_id",
            "expected_project_key",
            "expected_project_name",
            "expected_priority_id",
            "expected_priority_name",
            "expected_labels",
            "expected_due_date",
            "expected_target_priority_id",
            "expected_target_priority_name",
        ],

        "required_arguments": [
            "ticket_key",
        ],

        "grounded_arguments": [
            "ticket_key",
            "priority",
            "add_labels",
            "remove_labels",
            "due_date",
        ],

        "parameters": {
            "ticket_key":
                TICKET_KEY_PARAMETER,

            "priority": {
                "type":
                    "str",

                "description": (
                    "Optional exact priority name supplied by the user, "
                    "such as High."
                ),
            },

            "add_labels": {
                "type":
                    "list[str]",

                "description": (
                    "Optional exact labels the user asked to add. Do not "
                    "invent, rename or normalize labels."
                ),
            },

            "remove_labels": {
                "type":
                    "list[str]",

                "description": (
                    "Optional exact labels the user asked to remove."
                ),
            },

            "due_date": {
                "type":
                    "str",

                "description": (
                    "Optional exact due date in YYYY-MM-DD form exactly as "
                    "the user wrote it. Never convert a relative phrase such "
                    "as next Friday into a date."
                ),
            },
        },

        "result_formatter": (
            format_ticket_mutation_changes_result
        ),

        "approval_formatter": (
            format_ticket_update_approval
        ),
    },

    "ticket_link": {
        "description": (
            "Create one link between two existing tickets, such as one "
            "ticket blocks, is blocked by, duplicates, causes, clones or "
            "relates to another. Uses the exact relationship words the user "
            "supplied; the trusted provider resolves the link type and "
            "direction. This does NOT read existing links and does NOT "
            "remove links."
        ),

        "risk":
            "medium",

        "requires_approval":
            True,

        "policy_resolver":
            evaluate_ticket_link_policy,

        "trusted_policy_arguments": [
            "expected_issue_id",
            "expected_issue_key",
            "expected_issue_summary",
            "expected_project_id",
            "expected_project_key",
            "expected_project_name",
            "expected_other_issue_id",
            "expected_other_issue_key",
            "expected_other_issue_summary",
            "expected_link_type_id",
            "expected_link_type_name",
            "expected_link_direction",
            "expected_link_phrase",
            "expected_link_exists",
        ],

        "required_arguments": [
            "ticket_key",
            "other_ticket_key",
            "relationship",
        ],

        "grounded_arguments": [
            "ticket_key",
            "other_ticket_key",
            "relationship",
        ],

        "parameters": {
            "ticket_key": {
                "type":
                    "str",

                "description": (
                    "Exact identifier of the ticket that is the subject of "
                    "the relationship sentence, as supplied by the user."
                ),
            },

            "other_ticket_key": {
                "type":
                    "str",

                "description": (
                    "Exact identifier of the other ticket, as supplied by "
                    "the user."
                ),
            },

            "relationship": {
                "type":
                    "str",

                "description": (
                    "Exact relationship words supplied by the user so that "
                    "'<ticket_key> <relationship> <other_ticket_key>' states "
                    "the request, for example blocks, is blocked by, "
                    "duplicates or relates to."
                ),
            },
        },

        "result_formatter": (
            format_ticket_mutation_changes_result
        ),

        "approval_formatter": (
            format_ticket_link_approval
        ),
    },

    "ticket_links": {
        "description": (
            "List the issues linked to exactly one ticket, each with the "
            "relationship from that ticket's perspective, the other ticket's "
            "key, summary and status. Read-only; this does NOT create or "
            "remove links."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "required_arguments": [
            "ticket_key",
        ],

        "grounded_arguments": [
            "ticket_key",
        ],

        "parameters": {
            "ticket_key":
                TICKET_KEY_PARAMETER,
        },

        "result_formatter": (
            format_ticket_links_result
        ),

        "presentation_builder": (
            build_ticket_links_card
        ),
    },

    "ticket_assigned": {
        "description": (
            "List tickets assigned to one explicitly identified user: a "
            "provider account ID, or an exact email address or display name "
            "that trusted lookup resolves to exactly one user. Optional exact "
            "status filter and limit. Read-only. A self-reference such as me "
            "is not accepted, and this does NOT assign tickets."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "required_arguments": [
            "assignee",
        ],

        "grounded_arguments": [
            "assignee",
            "status",
        ],

        "parameters": {
            "assignee": {
                "type":
                    "str",

                "description": (
                    "Exact account ID, email address or display name of the "
                    "user, as supplied by the user."
                ),
            },

            "status": {
                "type":
                    "str",

                "description": (
                    "Optional exact ticket status filter supplied by the user."
                ),
            },

            "limit": {
                "type":
                    "int",

                "description": (
                    "Maximum number of tickets to return from 1 to 25."
                ),
            },
        },

        "result_formatter": (
            format_ticket_assigned_result
        ),

        "presentation_builder": (
            build_ticket_assigned_card
        ),
    },

    "ticket_worklog_add": {
        "description": (
            "Log time spent on exactly one existing ticket as one worklog "
            "entry, with an exact Jira duration such as 1h 30m and an "
            "optional exact comment. This does NOT add a ticket comment, "
            "change the remaining estimate, or edit existing worklogs."
        ),

        "risk":
            "medium",

        "requires_approval":
            True,

        "policy_resolver":
            evaluate_ticket_worklog_add_policy,

        "trusted_policy_arguments": [
            "expected_issue_id",
            "expected_issue_key",
            "expected_issue_summary",
            "expected_project_id",
            "expected_project_key",
            "expected_project_name",
            "expected_time_spent_seconds",
        ],

        "required_arguments": [
            "ticket_key",
            "time_spent",
        ],

        "grounded_arguments": [
            "ticket_key",
            "time_spent",
            "comment",
        ],

        "parameters": {
            "ticket_key":
                TICKET_KEY_PARAMETER,

            "time_spent": {
                "type":
                    "str",

                "description": (
                    "Exact duration supplied by the user in Jira form, such "
                    "as 15m, 2h or 1h 30m."
                ),
            },

            "comment": {
                "type":
                    "str",

                "description": (
                    "Optional exact worklog comment supplied by the user. Do "
                    "not summarize, rewrite, or invent it."
                ),
            },
        },

        "result_formatter": (
            format_ticket_mutation_changes_result
        ),

        "approval_formatter": (
            format_ticket_worklog_add_approval
        ),
    },
}
