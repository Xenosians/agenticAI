from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from .jira import JiraTicketService
from .jira_issue_policy import (
    MAX_PROVIDER_ID_LENGTH,
    MAX_PROVIDER_NAME_LENGTH,
    MAX_PROVIDER_TEXT_LENGTH,
    base_snapshot_arguments,
    exact_user_string,
    jira_mutation_result,
    jira_rejection_result,
    provider_string,
    read_jira_issue_snapshot,
    require_issue_permission,
    revalidate_base_snapshot,
    trusted_get_json,
    validate_issue_key,
)
from .validation import (
    MAX_RELATIONSHIP_LENGTH,
    resolve_link_relationship,
)


OPERATION = "link_tickets"


def _link_types(service: JiraTicketService) -> list[dict[str, str]]:
    payload = trusted_get_json(
        service,
        "/rest/api/3/issueLinkType",
        operation="issue link type lookup",
    )

    raw_types = (
        payload.get("issueLinkTypes")
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(raw_types, list):
        raise RuntimeError("Jira returned invalid issue link types.")

    link_types = []
    for item in raw_types:
        if not isinstance(item, dict):
            continue
        try:
            link_types.append(
                {
                    name: provider_string(
                        item.get(name),
                        field_name=f"link type {name}",
                        max_length=(
                            MAX_PROVIDER_ID_LENGTH
                            if name == "id"
                            else MAX_PROVIDER_NAME_LENGTH
                        ),
                    )
                    for name in ("id", "name", "inward", "outward")
                }
            )
        except RuntimeError:
            continue

    return link_types


def _matching_link_ids(
    service: JiraTicketService,
    *,
    issue_key: str,
    other_key: str,
    link_type_id: str,
    direction: str,
    symmetric: bool,
    operation: str,
) -> list[str]:
    """IDs of links on ``issue_key`` that already state the requested sentence."""

    payload = trusted_get_json(
        service,
        f"/rest/api/3/issue/{quote(issue_key, safe='')}",
        operation=operation,
        params={"fields": "issuelinks"},
    )

    fields = payload.get("fields") if isinstance(payload, dict) else None
    links = fields.get("issuelinks") if isinstance(fields, dict) else None
    if not isinstance(links, list):
        raise RuntimeError("Jira returned invalid issue link metadata.")

    directions = ("outward", "inward") if symmetric else (direction,)
    matches = []

    for link in links:
        if not isinstance(link, dict):
            continue
        link_type = link.get("type")
        if not isinstance(link_type, dict) or str(link_type.get("id")) != link_type_id:
            continue
        for side in directions:
            other = link.get(f"{side}Issue")
            if isinstance(other, dict) and other.get("key") == other_key:
                matches.append(str(link.get("id")))

    return matches


def _other_issue_arguments(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_other_issue_id": snapshot["issue_id"],
        "expected_other_issue_key": snapshot["issue_key"],
        "expected_other_issue_summary": snapshot["issue_summary"],
    }


def prepare_jira_ticket_link(
    service: JiraTicketService,
    *,
    ticket_key: str,
    other_ticket_key: str,
    relationship: str,
) -> dict[str, Any]:
    """Read-only preparation: both issues, link type and direction, existing link."""

    try:
        requested_key = validate_issue_key(ticket_key)
        other_key = validate_issue_key(other_ticket_key)
        requested_relationship = exact_user_string(
            relationship,
            field_name="relationship",
            max_length=MAX_RELATIONSHIP_LENGTH,
        )

        if requested_key.casefold() == other_key.casefold():
            raise ValueError("A Jira issue cannot be linked to itself.")

        resolved = resolve_link_relationship(
            requested_relationship,
            _link_types(service),
        )

        snapshot = read_jira_issue_snapshot(
            service,
            requested_key,
            operation="link issue snapshot lookup",
        )
        other = read_jira_issue_snapshot(
            service,
            other_key,
            operation="linked issue snapshot lookup",
        )

        if snapshot["issue_key"] != requested_key or other["issue_key"] != other_key:
            raise ValueError(
                "Jira resolved an issue reference to a different issue key. "
                "Trusted policy will not rewrite it."
            )

        require_issue_permission(
            service,
            requested_key,
            "LINK_ISSUES",
            operation="link permission check",
        )

        existing = _matching_link_ids(
            service,
            issue_key=requested_key,
            other_key=other_key,
            link_type_id=resolved["link_type_id"],
            direction=resolved["direction"],
            symmetric=resolved["symmetric"],
            operation="existing link lookup",
        )

        return {
            "ok": True,
            "status": "ready",
            "risk": "medium",
            "requires_approval": True,
            "execution_arguments": {
                "ticket_key": requested_key,
                "other_ticket_key": other_key,
                "relationship": requested_relationship,
                **base_snapshot_arguments(snapshot),
                **_other_issue_arguments(other),
                "expected_link_type_id": resolved["link_type_id"],
                "expected_link_type_name": resolved["link_type_name"],
                "expected_link_direction": resolved["direction"],
                "expected_link_phrase": resolved["phrase"],
                "expected_link_exists": bool(existing),
            },
            "error": None,
        }

    except (PermissionError, LookupError, ValueError) as exc:
        return {
            "ok": False,
            "status": "denied",
            "risk": "medium",
            "requires_approval": True,
            "error": str(exc),
        }

    except (RuntimeError, httpx.HTTPError) as exc:
        return {
            "ok": False,
            "status": "error",
            "risk": "medium",
            "requires_approval": True,
            "error": str(exc),
        }


def execute_jira_ticket_link(
    service: JiraTicketService,
    *,
    ticket_key: str,
    other_ticket_key: str,
    relationship: str,
    expected_issue_id: str | None = None,
    expected_issue_key: str | None = None,
    expected_issue_summary: str | None = None,
    expected_project_id: str | None = None,
    expected_project_key: str | None = None,
    expected_project_name: str | None = None,
    expected_other_issue_id: str | None = None,
    expected_other_issue_key: str | None = None,
    expected_other_issue_summary: str | None = None,
    expected_link_type_id: str | None = None,
    expected_link_type_name: str | None = None,
    expected_link_direction: str | None = None,
    expected_link_phrase: str | None = None,
    expected_link_exists: bool | None = None,
) -> dict[str, Any]:
    """Create one approved link once; an existing identical link is a no-op."""

    try:
        requested_key = validate_issue_key(ticket_key)
        other_key = validate_issue_key(other_ticket_key)

        trusted = (
            expected_issue_id,
            expected_issue_key,
            expected_issue_summary,
            expected_project_id,
            expected_project_key,
            expected_project_name,
            expected_other_issue_id,
            expected_other_issue_key,
            expected_other_issue_summary,
            expected_link_type_id,
            expected_link_type_name,
            expected_link_direction,
            expected_link_phrase,
            expected_link_exists,
        )
        if any(value is None for value in trusted) or not isinstance(expected_link_exists, bool):
            raise ValueError(
                "Approved Jira link requires a complete trusted provider snapshot."
            )

        if expected_issue_key != requested_key or expected_other_issue_key != other_key:
            raise ValueError(
                "Approved Jira link snapshot does not match the persisted ticket keys."
            )

        resolved = resolve_link_relationship(
            relationship,
            _link_types(service),
        )

        if (
            resolved["link_type_id"] != expected_link_type_id
            or resolved["link_type_name"] != expected_link_type_name
            or resolved["direction"] != expected_link_direction
            or resolved["phrase"] != expected_link_phrase
        ):
            raise PermissionError(
                "The Jira link type changed after approval. The approved link "
                "will not be created."
            )

        revalidate_base_snapshot(
            service,
            operation="approved link issue snapshot revalidation",
            expected_issue_id=expected_issue_id,
            expected_issue_key=expected_issue_key,
            expected_issue_summary=expected_issue_summary,
            expected_project_id=expected_project_id,
            expected_project_key=expected_project_key,
            expected_project_name=expected_project_name,
        )

        other = read_jira_issue_snapshot(
            service,
            other_key,
            operation="approved linked issue snapshot revalidation",
        )
        if (
            other["issue_id"] != exact_user_string(
                expected_other_issue_id,
                field_name="expected_other_issue_id",
                max_length=MAX_PROVIDER_ID_LENGTH,
            )
            or other["issue_summary"] != exact_user_string(
                expected_other_issue_summary,
                field_name="expected_other_issue_summary",
                max_length=MAX_PROVIDER_TEXT_LENGTH,
            )
        ):
            raise PermissionError(
                "The linked Jira issue changed after approval. The approved "
                "link will not be created."
            )

        existing = _matching_link_ids(
            service,
            issue_key=requested_key,
            other_key=other_key,
            link_type_id=expected_link_type_id,
            direction=expected_link_direction,
            symmetric=resolved["symmetric"],
            operation="approved link existence check",
        )

        # The user approved a no-op; creating the link now would be a change
        # they never saw.
        if expected_link_exists and not existing:
            raise PermissionError(
                "The existing link was removed after approval. The approval "
                "covered no change, so no link will be created."
            )

    except (PermissionError, LookupError, ValueError) as exc:
        return jira_mutation_result(
            operation=OPERATION,
            ticket_key=ticket_key,
            status="denied",
            error=str(exc),
        )

    except (RuntimeError, httpx.HTTPError) as exc:
        return jira_mutation_result(
            operation=OPERATION,
            ticket_key=ticket_key,
            status="error",
            error=str(exc),
        )

    sentence = f"{requested_key} {expected_link_phrase} {other_key}"

    if existing:
        return jira_mutation_result(
            operation=OPERATION,
            ticket_key=requested_key,
            ok=True,
            status="success",
            verification_ok=True,
            link_id=existing[0],
            new_value=sentence,
            message=f"The link already exists ({sentence}); no duplicate was created.",
        )

    # Jira states a created link as "<inwardIssue> <outward phrase>
    # <outwardIssue>", the reverse of what the field names suggest. The live
    # read-back below is what proves the direction.
    if expected_link_direction == "outward":
        inward_key, outward_key = requested_key, other_key
    else:
        inward_key, outward_key = other_key, requested_key

    try:
        response = service.client.post(
            "/rest/api/3/issueLink",
            json={
                "type": {"id": expected_link_type_id},
                "inwardIssue": {"key": inward_key},
                "outwardIssue": {"key": outward_key},
            },
        )
    except (httpx.TimeoutException, httpx.TransportError):
        return jira_mutation_result(
            operation=OPERATION,
            ticket_key=requested_key,
            status="outcome_unknown",
            mutation_performed=None,
            new_value=sentence,
            error=(
                "The Jira link request encountered an ambiguous transport "
                "failure. Automatic retry is disabled."
            ),
        )

    rejection = jira_rejection_result(
        response,
        operation=OPERATION,
        ticket_key=requested_key,
        action="issue link",
        new_value=sentence,
    )
    if rejection is not None:
        return rejection

    try:
        created = _matching_link_ids(
            service,
            issue_key=requested_key,
            other_key=other_key,
            link_type_id=expected_link_type_id,
            direction=expected_link_direction,
            symmetric=resolved["symmetric"],
            operation="link read-back verification",
        )
    except Exception:
        created = []

    if len(created) != 1:
        return jira_mutation_result(
            operation=OPERATION,
            ticket_key=requested_key,
            status="outcome_unknown",
            mutation_performed=True,
            new_value=sentence,
            error=(
                "Jira accepted the link request, but trusted read-back could not "
                f"confirm exactly one link stating '{sentence}'."
            ),
        )

    return jira_mutation_result(
        operation=OPERATION,
        ticket_key=requested_key,
        ok=True,
        status="success",
        changed=True,
        mutation_performed=True,
        verification_ok=True,
        link_id=created[0],
        new_value=sentence,
        message=f"Link created and verified: {sentence}.",
    )
