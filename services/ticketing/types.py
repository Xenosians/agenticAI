from pydantic import (
    BaseModel,
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
