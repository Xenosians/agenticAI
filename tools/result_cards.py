from __future__ import annotations

from typing import Any


RESULT_CARD_SCHEMA = (
    "result-card.v1"
)

VALID_SECTION_KINDS = {
    "text",
    "preformatted",
    "list",
}


def result_field(
    label: str,
    value: Any,
) -> dict[
    str,
    str,
] | None:
    """
    Build one safe display field.

    Presentation values are intentionally converted to text.
    The authoritative machine-readable value remains available
    separately in AgentResult.tool_result.
    """

    if not isinstance(
        label,
        str,
    ):
        return None

    label = (
        label.strip()
    )

    if not label:
        return None

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        rendered_value = (
            "yes"
            if value
            else "no"
        )

    else:
        rendered_value = (
            str(
                value
            ).strip()
        )

    if not rendered_value:
        return None

    return {
        "label":
            label,

        "value":
            rendered_value,
    }


def text_section(
    *,
    title: str,
    content: str,
    kind: str = "text",
) -> dict[
    str,
    Any,
] | None:
    """
    Build a text/preformatted result-card section.
    """

    if kind not in {
        "text",
        "preformatted",
    }:
        return None

    if not isinstance(
        title,
        str,
    ):
        return None

    if not isinstance(
        content,
        str,
    ):
        return None

    title = (
        title.strip()
    )

    content = (
        content.strip()
    )

    if (
        not title
        or not content
    ):
        return None

    return {
        "kind":
            kind,

        "title":
            title,

        "content":
            content,
    }


def list_section(
    *,
    title: str,
    items: list[str],
) -> dict[
    str,
    Any,
] | None:
    """
    Build a list-style result-card section.
    """

    if not isinstance(
        title,
        str,
    ):
        return None

    title = (
        title.strip()
    )

    if not title:
        return None

    if not isinstance(
        items,
        list,
    ):
        return None

    normalized_items = []

    for item in items:
        if not isinstance(
            item,
            str,
        ):
            continue

        item = (
            item.strip()
        )

        if item:
            normalized_items.append(
                item
            )

    if not normalized_items:
        return None

    return {
        "kind":
            "list",

        "title":
            title,

        "content":
            normalized_items,
    }


def build_result_card(
    *,
    kind: str,
    title: str,
    status: str,
    fields: list[
        dict[
            str,
            str,
        ]
        | None
    ] | None = None,
    sections: list[
        dict[
            str,
            Any,
        ]
        | None
    ] | None = None,
) -> dict[
    str,
    Any,
]:
    """
    Construct one canonical result-card.v1 object.
    """

    card = {
        "schema":
            RESULT_CARD_SCHEMA,

        "kind":
            kind,

        "title":
            title,

        "status":
            status,

        "fields": [
            field

            for field
            in (
                fields
                or []
            )

            if field is not None
        ],

        "sections": [
            section

            for section
            in (
                sections
                or []
            )

            if section is not None
        ],
    }

    validated = (
        validate_result_card(
            card
        )
    )

    if validated is None:
        raise ValueError(
            "Invalid result-card presentation."
        )

    return validated


def validate_result_card(
    card: Any,
) -> dict[
    str,
    Any,
] | None:
    """
    Validate and normalize the presentation object crossing the
    AI -> Phoenix -> frontend boundary.

    Unknown presentation keys are deliberately discarded.
    """

    if not isinstance(
        card,
        dict,
    ):
        return None

    if (
        card.get(
            "schema"
        )
        != RESULT_CARD_SCHEMA
    ):
        return None

    kind = (
        card.get(
            "kind"
        )
    )

    title = (
        card.get(
            "title"
        )
    )

    status = (
        card.get(
            "status"
        )
    )

    if not all(
        isinstance(
            value,
            str,
        )
        and bool(
            value.strip()
        )

        for value
        in (
            kind,
            title,
            status,
        )
    ):
        return None

    raw_fields = (
        card.get(
            "fields",
            [],
        )
    )

    if not isinstance(
        raw_fields,
        list,
    ):
        return None

    fields = []

    for field in raw_fields:
        if not isinstance(
            field,
            dict,
        ):
            return None

        label = (
            field.get(
                "label"
            )
        )

        value = (
            field.get(
                "value"
            )
        )

        if (
            not isinstance(
                label,
                str,
            )
            or not label.strip()
            or not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            return None

        fields.append(
            {
                "label":
                    label.strip(),

                "value":
                    value.strip(),
            }
        )

    raw_sections = (
        card.get(
            "sections",
            [],
        )
    )

    if not isinstance(
        raw_sections,
        list,
    ):
        return None

    sections = []

    for section in raw_sections:
        if not isinstance(
            section,
            dict,
        ):
            return None

        section_kind = (
            section.get(
                "kind"
            )
        )

        section_title = (
            section.get(
                "title"
            )
        )

        content = (
            section.get(
                "content"
            )
        )

        if (
            section_kind
            not in VALID_SECTION_KINDS
        ):
            return None

        if (
            not isinstance(
                section_title,
                str,
            )
            or not section_title.strip()
        ):
            return None

        if (
            section_kind
            == "list"
        ):
            if not isinstance(
                content,
                list,
            ):
                return None

            normalized_content = []

            for item in content:
                if (
                    not isinstance(
                        item,
                        str,
                    )
                    or not item.strip()
                ):
                    return None

                normalized_content.append(
                    item.strip()
                )

        else:
            if (
                not isinstance(
                    content,
                    str,
                )
                or not content.strip()
            ):
                return None

            normalized_content = (
                content.strip()
            )

        sections.append(
            {
                "kind":
                    section_kind,

                "title":
                    section_title.strip(),

                "content":
                    normalized_content,
            }
        )

    return {
        "schema":
            RESULT_CARD_SCHEMA,

        "kind":
            kind.strip(),

        "title":
            title.strip(),

        "status":
            status.strip(),

        "fields":
            fields,

        "sections":
            sections,
    }