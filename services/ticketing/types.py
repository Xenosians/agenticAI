from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class TicketRecord(
    BaseModel
):
    provider: str

    key: str
    summary: str
    status: str

    ticket_type: (
        str | None
    ) = None

    priority: (
        str | None
    ) = None

    assignee: (
        str | None
    ) = None

    reporter: (
        str | None
    ) = None

    project_key: (
        str | None
    ) = None

    project_name: (
        str | None
    ) = None

    created_at: (
        str | None
    ) = None

    updated_at: (
        str | None
    ) = None


class TicketLookupResult(
    BaseModel
):
    ok: bool
    status: str

    ticket: (
        TicketRecord | None
    ) = None

    error: (
        str | None
    ) = None


class TicketSearchQuery(
    BaseModel
):
    """
    Provider-neutral bounded ticket search.

    Provider-native query languages such as Jira JQL remain
    behind trusted provider implementations.
    """

    text: (
        str | None
    ) = Field(
        default=None,
        max_length=200,
    )

    project_key: (
        str | None
    ) = Field(
        default=None,
        max_length=100,
    )

    status: (
        str | None
    ) = Field(
        default=None,
        max_length=100,
    )

    priority: (
        str | None
    ) = Field(
        default=None,
        max_length=100,
    )

    limit: int = Field(
        default=10,
        ge=1,
        le=25,
    )

    @field_validator(
        "text",
        "project_key",
        "status",
        "priority",
        mode="before",
    )
    @classmethod
    def normalize_optional_string(
        cls,
        value,
    ):
        if value is None:
            return None

        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                "Ticket search filters "
                "must be strings."
            )

        normalized = (
            value.strip()
        )

        if not normalized:
            return None

        return normalized


class TicketSearchResult(
    BaseModel
):
    ok: bool
    status: str

    tickets: list[
        TicketRecord
    ] = Field(
        default_factory=list
    )

    count: int = Field(
        default=0,
        ge=0,
    )

    truncated: bool = False

    error: (
        str | None
    ) = None


class TicketFieldChange(
    BaseModel
):
    field: str

    from_value: (
        str | None
    ) = None

    to_value: (
        str | None
    ) = None


class TicketHistoryEntry(
    BaseModel
):
    id: (
        str | None
    ) = None

    author: (
        str | None
    ) = None

    created_at: (
        str | None
    ) = None

    changes: list[
        TicketFieldChange
    ] = Field(
        default_factory=list
    )


class TicketHistoryResult(
    BaseModel
):
    ok: bool
    status: str

    provider: (
        str | None
    ) = None

    ticket_key: (
        str | None
    ) = None

    history: list[
        TicketHistoryEntry
    ] = Field(
        default_factory=list
    )

    count: int = Field(
        default=0,
        ge=0,
    )

    truncated: bool = False

    error: (
        str | None
    ) = None


class TicketComment(
    BaseModel
):
    id: str

    author: (
        str | None
    ) = None

    body: str

    created_at: (
        str | None
    ) = None

    updated_at: (
        str | None
    ) = None


class TicketCommentsResult(
    BaseModel
):
    ok: bool
    status: str

    provider: (
        str | None
    ) = None

    ticket_key: (
        str | None
    ) = None

    comments: list[
        TicketComment
    ] = Field(
        default_factory=list
    )

    count: int = Field(
        default=0,
        ge=0,
    )

    truncated: bool = False

    error: (
        str | None
    ) = None