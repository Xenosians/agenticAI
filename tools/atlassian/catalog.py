from typing import Any


def _credential_status(result: dict[str, Any]) -> str:
    admin = "configured" if result.get("admin_api_key_configured") else "not configured"
    jira = "configured" if result.get("jira_basic_configured") else "not configured"
    default_org = "configured" if result.get("default_org_configured") else "not configured"
    return (
        "Atlassian credential status: "
        f"admin API key {admin}; Jira API-token auth {jira}; "
        f"default organization {default_org}. Secret values are not exposed."
    )


def _org_list(result: dict[str, Any]) -> str:
    suffix = " Additional organizations exist." if result.get("truncated") else ""
    return f"Found {result.get('count', 0)} Atlassian organization(s).{suffix}"


def _org_get(result: dict[str, Any]) -> str:
    org = result.get("organization")
    if not isinstance(org, dict):
        return "Atlassian organization lookup completed."
    return f"Atlassian organization: {org.get('name') or 'unnamed'} ({org.get('id')})."


def _workspace_list(result: dict[str, Any]) -> str:
    suffix = " Additional workspaces exist." if result.get("truncated") else ""
    org_id = result.get("organization_id")
    scope = f" in organization {org_id}" if org_id else ""
    return f"Found {result.get('count', 0)} Atlassian workspace(s){scope}.{suffix}"


def _token_metadata(
    result: dict[
        str,
        Any,
    ],
) -> str:

    organization_id = (
        result.get(
            "organization_id"
        )
    )

    scope = (
        f" in organization {organization_id}"
        if organization_id
        else ""
    )

    suffix = (
        " Additional token metadata records exist."
        if result.get(
            "truncated"
        )
        else ""
    )

    return (
        f"Found {result.get('count', 0)} "
        f"API token metadata record(s){scope}. "
        "Token secret values are not exposed."
        f"{suffix}"
    )


ATLASSIAN_TOOLS = {
    "atlassian_credential_status": {
        "description": (
            "Inspect whether trusted Atlassian/Jira credentials and a default "
            "organization are configured. This never returns API keys, API token "
            "secrets, Authorization headers, passwords, or refresh tokens."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": [],
        "parameters": {},
        "result_formatter": _credential_status,
    },
    "atlassian_org_list": {
        "description": (
            "List Atlassian organizations visible to the trusted organization-admin "
            "API credential. This is read-only."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": [],
        "parameters": {
            "limit": {
                "type": "int",
                "description": "Optional maximum number of organizations from 1 to 50.",
            }
        },
        "result_formatter": _org_list,
    },
    "atlassian_org_get": {
        "description": (
            "Retrieve one Atlassian organization by exact ID. If no ID is supplied, "
            "trusted configuration may provide the default organization."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": ["org_id"],
        "parameters": {
            "org_id": {
                "type": "str",
                "description": (
                    "Optional exact organization ID explicitly supplied by the user. "
                    "Omit it to use trusted default organization configuration."
                ),
            }
        },
        "result_formatter": _org_get,
    },
    "atlassian_workspace_list": {
        "description": (
            "List or search existing Atlassian workspaces/product instances inside one "
            "organization. This is discovery only and does not create workspaces."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": ["org_id", "name"],
        "parameters": {
            "org_id": {
                "type": "str",
                "description": (
                    "Optional exact organization ID supplied by the user. Omit it to "
                    "use trusted default organization configuration."
                ),
            },
            "name": {
                "type": "str",
                "description": "Optional exact workspace/site name to search.",
            },
            "limit": {
                "type": "int",
                "description": "Optional maximum number of workspaces from 1 to 50.",
            },
        },
        "result_formatter": _workspace_list,
    },
    "atlassian_api_token_metadata": {
        "description": (
            "List metadata for API tokens owned by exactly one Atlassian account when "
            "the trusted admin credential has permission. Returns token IDs, labels, "
            "timestamps, and expiry metadata only; it can never recover token secrets."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": ["account_id"],
        "parameters": {
            "account_id": {
                "type": "str",
                "description": "Exact Atlassian account ID explicitly supplied by the user.",
            },
            "limit": {
                "type": "int",
                "description": "Optional maximum token metadata records from 1 to 50.",
            },
        },
        "result_formatter": _token_metadata,
    },
}
