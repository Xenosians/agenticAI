"""Provider-neutral validation of user-supplied ticket values.

Both the mock and the Jira providers use these rules so that trusted
policy, approval and execution agree on what a request means.
"""

from __future__ import annotations

import re

from datetime import date
from typing import Any


MAX_USER_REFERENCE_LENGTH = 200
MAX_PRIORITY_LENGTH = 100
MAX_LABEL_LENGTH = 255
MAX_LABELS_PER_UPDATE = 20
MAX_RELATIONSHIP_LENGTH = 100
MAX_WORKLOG_COMMENT_LENGTH = 5000

# A request about "me" must never silently become the integration's own
# service account, which is the identity the provider authenticates as.
SELF_REFERENCES = frozenset(
    {
        "me",
        "myself",
        "i",
        "self",
        "mine",
        "my account",
        "current user",
        "the current user",
    }
)

DUE_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

DURATION_TOKEN_PATTERN = re.compile(r"^(\d{1,5})([wdhm])$")
DURATION_UNITS = ("w", "d", "h", "m")


def explicit_user_reference(
    value: object,
) -> tuple[
    str | None,
    str | None,
]:
    """Validate one explicit user reference; self-references are refused."""

    if not isinstance(value, str):
        return None, "assignee must be a string."

    normalized = value.strip()

    if not normalized:
        return None, "assignee must not be empty."

    if len(normalized) > MAX_USER_REFERENCE_LENGTH:
        return None, (
            "assignee exceeds the maximum length of "
            f"{MAX_USER_REFERENCE_LENGTH} characters."
        )

    if normalized.casefold() in SELF_REFERENCES:
        return None, (
            "Identify the user explicitly by account ID, exact email address "
            "or exact display name. A self-reference such as 'me' is never "
            "mapped to the integration service account."
        )

    return normalized, None


def validate_due_date(
    value: object,
) -> str:
    if not isinstance(value, str) or DUE_DATE_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "due_date must be an exact calendar date in YYYY-MM-DD form; "
            "relative phrases such as 'next Friday' are not accepted."
        )

    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            "due_date is not a valid calendar date."
        ) from exc

    return value


def validate_labels(
    value: object,
    *,
    field_name: str,
) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(
            f"{field_name} must be a list of exact labels."
        )

    if len(value) > MAX_LABELS_PER_UPDATE:
        raise ValueError(
            f"{field_name} accepts at most {MAX_LABELS_PER_UPDATE} labels."
        )

    labels: list[str] = []

    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or len(item) > MAX_LABEL_LENGTH
            or any(char.isspace() for char in item)
        ):
            raise ValueError(
                f"{field_name} entries must be non-empty labels without "
                f"spaces of at most {MAX_LABEL_LENGTH} characters."
            )

        if item in labels:
            raise ValueError(
                f"{field_name} contains the label '{item}' more than once."
            )

        labels.append(item)

    return labels


def validate_update_request(
    *,
    priority: object = None,
    add_labels: object = None,
    remove_labels: object = None,
    due_date: object = None,
) -> dict[str, Any]:
    """Validate the requested field changes; an empty update is refused."""

    requested_priority = None

    if priority is not None:
        if (
            not isinstance(priority, str)
            or not priority.strip()
            or priority != priority.strip()
            or len(priority) > MAX_PRIORITY_LENGTH
        ):
            raise ValueError(
                "priority must be an exact priority name without leading "
                f"or trailing whitespace, at most {MAX_PRIORITY_LENGTH} "
                "characters."
            )
        requested_priority = priority

    additions = validate_labels(add_labels, field_name="add_labels")
    removals = validate_labels(remove_labels, field_name="remove_labels")

    overlap = sorted(set(additions) & set(removals))
    if overlap:
        raise ValueError(
            "A label cannot be both added and removed in one update: "
            + ", ".join(overlap)
        )

    requested_due_date = (
        None
        if due_date is None
        else validate_due_date(due_date)
    )

    if (
        requested_priority is None
        and not additions
        and not removals
        and requested_due_date is None
    ):
        raise ValueError(
            "A ticket update needs at least one change: priority, "
            "add_labels, remove_labels or due_date."
        )

    return {
        "priority": requested_priority,
        "add_labels": additions,
        "remove_labels": removals,
        "due_date": requested_due_date,
    }


def updated_labels(
    current: list[str],
    *,
    add_labels: list[str],
    remove_labels: list[str],
) -> list[str]:
    result = [label for label in current if label not in remove_labels]
    result.extend(label for label in add_labels if label not in result)
    return sorted(result)


def parse_duration(
    value: object,
) -> dict[str, int]:
    """Parse an exact Jira duration such as '1h 30m' into unit amounts."""

    error = (
        "time_spent must be an exact Jira duration such as '45m', '2h' or "
        "'1h 30m' (units w, d, h, m in that order, separated by single "
        "spaces); relative or free-form phrases are not accepted."
    )

    if not isinstance(value, str) or not value:
        raise ValueError(error)

    amounts: dict[str, int] = {}
    last_unit_index = -1

    for token in value.split(" "):
        match = DURATION_TOKEN_PATTERN.fullmatch(token)
        if match is None:
            raise ValueError(error)

        unit_index = DURATION_UNITS.index(match.group(2))
        if unit_index <= last_unit_index:
            raise ValueError(error)

        last_unit_index = unit_index
        amounts[match.group(2)] = int(match.group(1))

    if not any(amounts.values()):
        raise ValueError(
            "time_spent must be greater than zero."
        )

    return amounts


def duration_seconds(
    amounts: dict[str, int],
    *,
    hours_per_day: float,
    days_per_week: float,
) -> int:
    """Seconds Jira records for a duration under the site's working time."""

    hour = 3600.0
    day = hours_per_day * hour
    week = days_per_week * day

    return int(
        round(
            amounts.get("w", 0) * week
            + amounts.get("d", 0) * day
            + amounts.get("h", 0) * hour
            + amounts.get("m", 0) * 60
        )
    )


def _phrase(value: str) -> str:
    return " ".join(value.casefold().split())


def resolve_link_relationship(
    relationship: object,
    link_types: list[dict[str, str]],
) -> dict[str, Any]:
    """
    Resolve the user's relationship words to one link type and direction.

    ``link_types`` items carry ``id``, ``name``, ``inward`` and ``outward``.
    ``direction`` is ``outward`` when the sentence
    "<ticket> <outward phrase> <other ticket>" states the request, and
    ``inward`` for "<ticket> <inward phrase> <other ticket>". Symmetric
    types always resolve to ``outward``.
    """

    if (
        not isinstance(relationship, str)
        or not relationship.strip()
        or len(relationship) > MAX_RELATIONSHIP_LENGTH
    ):
        raise ValueError(
            "relationship must be the relationship words supplied by the "
            f"user, at most {MAX_RELATIONSHIP_LENGTH} characters."
        )

    requested = _phrase(relationship)
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    options: list[str] = []

    for link_type in link_types:
        inward = _phrase(link_type["inward"])
        outward = _phrase(link_type["outward"])
        symmetric = inward == outward

        for phrase in (link_type["outward"], link_type["inward"]):
            if phrase not in options:
                options.append(phrase)

        directions = []
        if requested == outward:
            directions.append("outward")
        if requested == inward and not symmetric:
            directions.append("inward")
        if (
            not directions
            and symmetric
            and requested == _phrase(link_type["name"])
        ):
            directions.append("outward")

        for direction in directions:
            candidates[(link_type["id"], direction)] = {
                "link_type_id": link_type["id"],
                "link_type_name": link_type["name"],
                "link_type_inward": link_type["inward"],
                "link_type_outward": link_type["outward"],
                "direction": direction,
                "symmetric": symmetric,
                "phrase": link_type[direction],
            }

    valid = ", ".join(options) if options else "none configured"

    if not candidates:
        raise ValueError(
            f"Unknown relationship '{relationship.strip()}'. "
            f"Valid relationships: {valid}."
        )

    if len(candidates) > 1:
        names = ", ".join(
            sorted({item["link_type_name"] for item in candidates.values()})
        )
        raise ValueError(
            f"The relationship '{relationship.strip()}' is ambiguous across "
            f"link types {names}. Valid relationships: {valid}."
        )

    return next(iter(candidates.values()))
