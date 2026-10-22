"""Directory reads: account search, group membership and access explanations."""
from typing import Any

from services.directory.base import (
    MAX_ACCOUNT_SEARCH_RESULTS,
    MAX_GROUP_MEMBERS,
)


# Bound the text handed back to the model; the full result stays structured.
MAX_LISTED_NAMES = 25


def _failure(result: dict[str, Any], subject: str) -> str:
    return f"The directory {subject} did not succeed: {result.get('error') or 'unknown error'}"


def _names(items: list[dict[str, Any]], render) -> str:
    listed = [render(item) for item in items[:MAX_LISTED_NAMES]]
    hidden = len(items) - len(listed)
    return "; ".join(listed) + (f"; and {hidden} more" if hidden > 0 else "")


def _account(item: dict[str, Any]) -> str:
    state = ("enabled" if item.get("enabled") else "disabled") + (", locked" if item.get("locked") else "")
    details = ", ".join(value for value in (item.get("display_name"), item.get("email")) if value)
    return f"{item.get('username')} ({details + '; ' if details else ''}{state})"


def format_account_search_result(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return _failure(result, "account search")
    accounts = result.get("accounts") or []
    query = result.get("query")
    if not accounts:
        return f"No directory accounts match '{query}'."
    more = " The result was capped; refine the search text to see the rest." if result.get("truncated") else ""
    return f"{len(accounts)} directory account(s) match '{query}': {_names(accounts, _account)}.{more}"


def _group(item: dict[str, Any]) -> str:
    resources = item.get("resources") or []
    mapped = f", grants {', '.join(resources)}" if resources else ""
    return f"{item.get('name')} ({item.get('membership')}{mapped})"


def format_user_groups_result(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return _failure(result, "group lookup")
    groups = result.get("groups") or []
    user_id = result.get("user_id")
    if not groups:
        return f"{user_id} is not a member of any directory group."
    more = " The list was capped." if result.get("truncated") else ""
    return f"{user_id} is a member of {len(groups)} group(s): {_names(groups, _group)}.{more}"


def _member(item: dict[str, Any]) -> str:
    kind = item.get("type")
    label = f"{item.get('name')} ({kind}, {item.get('membership')})"
    return label if kind else f"{item.get('name')} ({item.get('membership')})"


def format_group_members_result(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return _failure(result, "member lookup")
    members = result.get("members") or []
    resources = result.get("resources") or []
    name = result.get("group")
    label = f"{name} (grants {', '.join(resources)})" if resources else name
    if not members:
        return f"The group {label} has no members."
    more = f" Only the first {len(members)} members are listed." if result.get("truncated") else ""
    return f"The group {label} has {len(members)} member(s): {_names(members, _member)}.{more}"


def format_access_explain_result(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return _failure(result, "access explanation")
    user_id, resource = result.get("user_id"), result.get("resource")
    reasons = result.get("reasons") or []
    group = result.get("group")
    if result.get("has_access") is True:
        membership = (result.get("checks") or {}).get("membership")
        return (f"{user_id} has access to {resource}: the account is enabled and unlocked "
                f"and is a {membership} member of {group}. The directory does not block it; "
                "if it still fails, the cause is outside the directory.")
    lines = []
    for reason in reasons:
        fix = reason.get("suggested_tool")
        lines.append(reason.get("message", "") + (f" Suggested fix (needs approval): {fix}." if fix else ""))
    verdict = ("does not have access to" if result.get("has_access") is False
               else "has undetermined access to")
    return f"{user_id} {verdict} {resource}. " + " ".join(lines) + " Nothing was changed."


_USER_ID = {
    "type": "str",
    "description": "Exact directory username (sAMAccountName) supplied by the user.",
}


DIRECTORY_TOOLS: dict = {
    "account_search": {
        "description": (
            # Router packets keep the first 200 characters, so the boundary comes first.
            # Avoid the word "text": it is the workspace search's routing evidence.
            "Read only, changes nothing: find directory accounts whose username, display name or "
            "email contains one exact fragment the user supplied; at most "
            f"{MAX_ACCOUNT_SEARCH_RESULTS} matches with display name, email, enabled and locked."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": ["query"],
        "parameters": {
            "query": {
                "type": "str",
                "description": (
                    "Exact search text supplied by the user (2 to 64 characters), such as part of a "
                    "username, display name or email. Do not invent, expand or rewrite it."
                ),
            },
        },
        "condition_fields": ["ok", "count"],
        "result_formatter": format_account_search_result,
    },
    "user_groups": {
        "description": (
            "Read only, changes nothing: show every directory group one exact user is a member of, "
            "including nested and primary-group membership, marking groups that grant a configured "
            "logical resource. Not a group's member list."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": ["user_id"],
        "parameters": {"user_id": _USER_ID},
        "condition_fields": ["ok", "count"],
        "result_formatter": format_user_groups_result,
    },
    "group_members": {
        "description": (
            "Read only, changes nothing: show who is in one directory group (users, nested groups, "
            "computers), named by a configured logical resource such as vpn or an exact group name; "
            f"at most {MAX_GROUP_MEMBERS} members, paged."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": ["group"],
        "parameters": {
            "group": {
                "type": "str",
                "description": (
                    "Exact logical resource name or exact directory group name supplied by the user. "
                    "Never invent a group name or distinguished name."
                ),
            },
        },
        "condition_fields": ["ok", "count"],
        "result_formatter": format_group_members_result,
    },
    "access_explain": {
        "description": (
            "Read only, never executes a fix: explain why one exact user can or cannot use one "
            "logical resource (account exists, enabled, unlocked, resource group mapped, direct or "
            "nested membership). Each blocking reason names the existing fix capability as a "
            "suggestion only."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": ["user_id", "resource"],
        "parameters": {
            "user_id": _USER_ID,
            "resource": {
                "type": "str",
                "description": "Exact logical resource name or identifier supplied by the user, such as vpn.",
            },
        },
        "condition_fields": ["ok", "has_access"],
        "result_formatter": format_access_explain_result,
    },
}
