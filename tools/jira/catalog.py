from typing import (
    Any,
)

from services.jira import (
    jira_project_template_argument_values,
)

from tools.jira.policy import (
    evaluate_jira_project_archive_policy,
    evaluate_jira_project_delete_policy,
    evaluate_jira_project_create_policy,
    evaluate_jira_project_update_policy,
)


def format_jira_project_list_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    count = (
        result.get(
            "count",
            0,
        )
    )

    total = (
        result.get(
            "total"
        )
    )

    query = (
        result.get(
            "query"
        )
    )

    scope = (
        f" matching {query!r}"
        if (
            isinstance(
                query,
                str,
            )
            and query
        )
        else ""
    )

    total_text = (
        f" of {total} visible project(s)"
        if isinstance(
            total,
            int,
        )
        else " project(s)"
    )

    suffix = (
        " Additional projects exist beyond this bounded result."
        if result.get(
            "truncated"
        )
        else ""
    )

    return (
        f"Found {count}{total_text}{scope}."
        f"{suffix}"
    )


def format_jira_project_get_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    project = (
        result.get(
            "project"
        )
    )

    if not isinstance(
        project,
        dict,
    ):
        return (
            "Jira project metadata was retrieved."
        )

    key = (
        project.get(
            "key"
        )
        or "unknown"
    )

    name = (
        project.get(
            "name"
        )
        or "unknown"
    )

    project_type = (
        project.get(
            "project_type_key"
        )
    )

    if project_type:
        return (
            f"Jira project {key} is {name!r} "
            f"with project type {project_type!r}."
        )

    return (
        f"Jira project {key} is {name!r}."
    )


def format_jira_project_create_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    project_key = (
        arguments.get(
            "project_key"
        )
        or "unknown"
    )

    project_name = (
        arguments.get(
            "project_name"
        )
        or "unknown"
    )

    template = (
        arguments.get(
            "template"
        )
        or "unknown"
    )

    return (
        f"Creating Jira project {project_key} "
        f"named {project_name!r} using template "
        f"{template!r} requires approval."
    )


def format_jira_project_create_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    project_key = (
        result.get(
            "project_key"
        )
        or "unknown"
    )

    project_name = (
        result.get(
            "project_name"
        )
        or "unknown"
    )

    project_id = (
        result.get(
            "project_id"
        )
    )

    if (
        result.get(
            "verification_ok"
        )
        is not True
    ):

        return (
            f"Jira project creation for {project_key} "
            "did not finish with verified final state."
        )

    suffix = (
        f" Project ID: {project_id}."
        if project_id
        else ""
    )

    return (
        f"Created and verified Jira project "
        f"{project_key} ({project_name})."
        f"{suffix}"
    )




JIRA_PROJECT_UPDATE_TRUSTED_ARGUMENTS = [
    "expected_project_id",
    "expected_project_key",
    "expected_project_name",
]


def format_jira_project_update_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    project_key = (
        arguments.get(
            "expected_project_key"
        )
        or arguments.get(
            "project_id_or_key"
        )
        or "project"
    )

    old_name = (
        arguments.get(
            "expected_project_name"
        )
        or "unknown"
    )

    new_name = (
        arguments.get(
            "new_name"
        )
        or "unknown"
    )

    return (
        f"Renaming Jira project {project_key} "
        f"from {old_name!r} to {new_name!r} "
        "requires approval."
    )


def format_jira_project_update_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    key = (
        result.get(
            "project_key"
        )
        or "project"
    )

    old_name = (
        result.get(
            "previous_project_name"
        )
        or "unknown"
    )

    new_name = (
        result.get(
            "new_project_name"
        )
        or "unknown"
    )

    if (
        result.get(
            "verification_ok"
        )
        is not True
    ):
        return (
            f"Jira project rename for {key} "
            f"from {old_name!r} to {new_name!r} "
            "did not finish with a verified final state."
        )

    return (
        f"Renamed and verified Jira project {key} "
        f"from {old_name!r} to {new_name!r}."
    )

JIRA_PROJECT_CREATE_TRUSTED_ARGUMENTS = [
    "expected_project_type_key",
    "expected_project_template_key",
    "expected_lead_account_id",
]


JIRA_TOOLS = {
    "jira_project_list": {
        "description": (
            "List a bounded collection of Jira PROJECT records "
            "visible to the configured Jira account. The projects "
            "themselves are the primary resources returned by this "
            "capability. An optional exact text query may narrow the "
            "provider-side project search. A project named only as "
            "the scope or filter for another resource does not make "
            "that request a project-list operation. Do not use this "
            "capability to search tickets or issues inside a project. "
            "This is read-only."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "query",
        ],

        "parameters": {
            "query": {
                "type":
                    "str",

                "description": (
                    "Optional exact project-search text explicitly "
                    "supplied by the user. Do not invent, rewrite, "
                    "expand, or normalize it."
                ),
            },

            "limit": {
                "type":
                    "int",

                "description": (
                    "Optional maximum number of projects to return "
                    "from 1 to 50."
                ),
            },
        },

        "result_formatter":
            format_jira_project_list_result,
    },

    "jira_project_get": {
        "description": (
            "Retrieve read-only metadata for exactly one Jira PROJECT "
            "using the exact project ID or project key supplied by "
            "the user. This retrieves the project itself; it does not "
            "retrieve a ticket or issue contained by that project."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "project_id_or_key",
        ],

        "parameters": {
            "project_id_or_key": {
                "type":
                    "str",

                "description": (
                    "Exact Jira project ID or project key supplied "
                    "by the user. Do not invent, rewrite, uppercase, "
                    "normalize, or otherwise alter it."
                ),
            },
        },

        "result_formatter":
            format_jira_project_get_result,
    },

    "jira_project_create": {
        "description": (
            "Create exactly one Jira project using an exact project "
            "key, exact project name, and one explicitly requested "
            "trusted template alias. Trusted policy resolves the "
            "authenticated project lead and provider-native project "
            "type/template identifiers. This is a governed mutation."
        ),

        "risk":
            "medium",

        "requires_approval":
            True,

        "policy_owns_preconditions":
            True,

        "policy_resolver":
            evaluate_jira_project_create_policy,

        "trusted_policy_arguments":
            JIRA_PROJECT_CREATE_TRUSTED_ARGUMENTS,

        "grounded_arguments": [
            "project_key",
            "project_name",
            "template",
        ],

        "parameters": {
            "project_key": {
                "type":
                    "str",

                "description": (
                    "Exact Jira project key explicitly supplied by "
                    "the user. Do not uppercase, rewrite, generate, "
                    "or replace it."
                ),
            },

            "project_name": {
                "type":
                    "str",

                "description": (
                    "Exact Jira project name explicitly supplied by "
                    "the user. Do not rewrite or generate it."
                ),
            },

            "template": {
                "type":
                    "str",

                "description": (
                    "Exact trusted project-template alias explicitly "
                    "supplied by the user."
                ),
            },
        },

        "argument_values_resolver":
            jira_project_template_argument_values,

        "approval_formatter":
            format_jira_project_create_approval,

        "result_formatter":
            format_jira_project_create_result,
    },
}

JIRA_TOOLS[
    "jira_project_update"
] = {
    "description": (
        "Rename exactly one existing Jira project using the exact "
        "project identifier/key and exact new project name supplied "
        "by the user. This capability changes ONLY the project name. "
        "It does not change project keys, schemes, lead, category, "
        "template, type, permissions, or any other project setting."
    ),

    "risk":
        "medium",

    "requires_approval":
        True,

    "policy_owns_preconditions":
        True,

    "policy_resolver":
        evaluate_jira_project_update_policy,

    "trusted_policy_arguments":
        JIRA_PROJECT_UPDATE_TRUSTED_ARGUMENTS,

    "grounded_arguments": [
        "project_id_or_key",
        "new_name",
    ],

    "parameters": {
        "project_id_or_key": {
            "type":
                "str",

            "description": (
                "Exact Jira project ID or key explicitly supplied "
                "by the user. Do not rewrite or normalize it."
            ),
        },

        "new_name": {
            "type":
                "str",

            "description": (
                "Exact new Jira project name explicitly supplied "
                "by the user. Do not rewrite, summarize, generate, "
                "or normalize it."
            ),
        },
    },

    "approval_formatter":
        format_jira_project_update_approval,

    "result_formatter":
        format_jira_project_update_result,
}

JIRA_PROJECT_ARCHIVE_TRUSTED_ARGUMENTS = [
    "expected_project_id",
    "expected_project_key",
    "expected_project_name",
]


def format_jira_project_archive_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    project_key = (
        arguments.get(
            "expected_project_key"
        )
        or arguments.get(
            "project_id_or_key"
        )
        or "project"
    )

    project_name = (
        arguments.get(
            "expected_project_name"
        )
        or "unknown"
    )

    return (
        f"Archiving Jira project {project_key} "
        f"({project_name!r}) requires HIGH-risk approval."
    )


def format_jira_project_archive_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    project_key = (
        result.get(
            "project_key"
        )
        or "project"
    )

    project_name = (
        result.get(
            "project_name"
        )
        or "unknown"
    )

    if (
        result.get(
            "verification_ok"
        )
        is not True
        or result.get(
            "archived"
        )
        is not True
    ):
        return (
            f"Jira project archival for {project_key} "
            f"({project_name!r}) did not finish with "
            "verified archived state."
        )

    return (
        f"Archived and verified Jira project "
        f"{project_key} ({project_name!r})."
    )


JIRA_TOOLS[
    "jira_project_archive"
] = {
    "description": (
        "Archive exactly one existing LIVE Jira project identified "
        "by the exact project ID or key explicitly supplied by the "
        "user. Trusted policy freezes the immutable project ID, key, "
        "name, and live-state precondition before approval. This is "
        "a HIGH-risk administrative mutation. It does not delete, "
        "restore, rename, or otherwise edit the project."
    ),

    "risk":
        "high",

    "requires_approval":
        True,

    "policy_owns_preconditions":
        True,

    "policy_resolver":
        evaluate_jira_project_archive_policy,

    "trusted_policy_arguments":
        JIRA_PROJECT_ARCHIVE_TRUSTED_ARGUMENTS,

    "grounded_arguments": [
        "project_id_or_key",
    ],

    "parameters": {
        "project_id_or_key": {
            "type":
                "str",

            "description": (
                "Exact Jira project ID or key explicitly supplied "
                "by the user. Do not rewrite, infer, broaden, or "
                "normalize it."
            ),
        },
    },

    "approval_formatter":
        format_jira_project_archive_approval,

    "result_formatter":
        format_jira_project_archive_result,
}



# ============================================================
# GOVERNED JIRA PROJECT DELETE
# ============================================================

JIRA_PROJECT_DELETE_TRUSTED_ARGUMENTS = [
    "expected_project_id",
    "expected_project_key",
    "expected_project_name",
]


def format_jira_project_delete_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    project_key = (
        arguments.get(
            "expected_project_key"
        )
        or arguments.get(
            "project_id_or_key"
        )
        or "project"
    )

    project_name = (
        arguments.get(
            "expected_project_name"
        )
        or "unknown"
    )

    return (
        f"Deleting Jira project {project_key} "
        f"({project_name!r}) requires HIGH-risk approval. "
        "Trusted execution keeps Jira undo enabled."
    )


def format_jira_project_delete_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    project_key = (
        result.get(
            "project_key"
        )
        or "project"
    )

    project_name = (
        result.get(
            "project_name"
        )
        or "unknown"
    )

    if (
        result.get(
            "verification_ok"
        )
        is not True

        or result.get(
            "deleted"
        )
        is not True
    ):
        return (
            f"Jira project deletion for {project_key} "
            f"({project_name!r}) did not finish with "
            "verified deleted state."
        )

    return (
        f"Deleted and verified Jira project "
        f"{project_key} ({project_name!r}) "
        "with provider undo enabled."
    )


JIRA_TOOLS[
    "jira_project_delete"
] = {
    "description": (
        "Delete exactly one existing LIVE Jira project identified "
        "by the exact project ID or key explicitly supplied by the "
        "user. Trusted policy freezes the immutable project ID, key, "
        "name, and live-state precondition before approval. Trusted "
        "execution always enables Jira undo; the model cannot request "
        "permanent deletion. Archived projects are rejected. This is "
        "a HIGH-risk administrative mutation."
    ),

    "risk":
        "high",

    "requires_approval":
        True,

    "policy_owns_preconditions":
        True,

    "policy_resolver":
        evaluate_jira_project_delete_policy,

    "trusted_policy_arguments":
        JIRA_PROJECT_DELETE_TRUSTED_ARGUMENTS,

    "grounded_arguments": [
        "project_id_or_key",
    ],

    "parameters": {
        "project_id_or_key": {
            "type":
                "str",

            "description": (
                "Exact Jira project ID or key explicitly supplied "
                "by the user. Do not invent, rewrite, normalize, "
                "restore, archive, or otherwise alter it."
            ),
        },
    },

    "approval_formatter":
        format_jira_project_delete_approval,

    "result_formatter":
        format_jira_project_delete_result,
}
