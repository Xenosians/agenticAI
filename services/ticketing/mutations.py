from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from datetime import (
    datetime,
    timezone,
)

from urllib.parse import (
    quote,
)

import httpx

from pydantic import (
    BaseModel,
)

from .base import (
    TicketService,
)

from .jira import (
    JiraTicketService,
)

from .mock import (
    MockTicketService,
)

from .types import (
    TicketComment,
)


MAX_TICKET_COMMENT_LENGTH = (
    5000
)


class TicketMutationResult(
    BaseModel
):
    """
    Provider-neutral ticket mutation result.

    This object describes provider execution only.

    Authorization and approval are owned by ToolGateway and the
    approval subsystem before these mutation methods are invoked.
    """

    ok: bool

    status: str

    provider: (
        str | None
    ) = None

    ticket_key: (
        str | None
    ) = None

    operation: (
        str | None
    ) = None

    changed: bool = False

    comment_id: (
        str | None
    ) = None

    message: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


class TicketMutationService(
    ABC
):
    """
    Provider-neutral command boundary for ticket mutations.

    Read/query operations remain on TicketService.

    Keeping command operations separate makes it explicit which
    provider calls can modify external ITSM state.
    """

    @abstractmethod
    def add_comment(
        self,
        ticket_key: str,
        comment: str,
    ) -> TicketMutationResult:
        raise NotImplementedError


def normalize_ticket_comment(
    comment: str,
) -> tuple[
    str | None,
    str | None,
]:
    if not isinstance(
        comment,
        str,
    ):

        return (
            None,
            "comment must be a string.",
        )

    normalized = (
        comment.strip()
    )

    if not normalized:

        return (
            None,
            "comment must not be empty.",
        )

    if (
        len(
            normalized
        )
        > MAX_TICKET_COMMENT_LENGTH
    ):

        return (
            None,
            (
                "comment exceeds the maximum "
                f"length of {MAX_TICKET_COMMENT_LENGTH} "
                "characters."
            ),
        )

    return (
        normalized,
        None,
    )


def utc_timestamp(
) -> str:

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


class MockTicketMutationService(
    TicketMutationService
):
    """
    Mutation adapter sharing the exact in-memory state owned by the
    corresponding MockTicketService.

    This matters because a mutation followed by a read must observe
    the changed state.
    """

    def __init__(
        self,
        ticket_service: MockTicketService,
    ) -> None:

        if not isinstance(
            ticket_service,
            MockTicketService,
        ):

            raise TypeError(
                "MockTicketMutationService requires "
                "MockTicketService."
            )

        self.ticket_service = (
            ticket_service
        )

    def add_comment(
        self,
        ticket_key: str,
        comment: str,
    ) -> TicketMutationResult:

        (
            normalized_comment,
            comment_error,
        ) = (
            normalize_ticket_comment(
                comment
            )
        )

        if normalized_comment is None:

            return (
                TicketMutationResult(
                    ok=False,

                    status="denied",

                    provider="mock",

                    operation=(
                        "add_comment"
                    ),

                    error=(
                        comment_error
                    ),
                )
            )

        lookup = (
            self.ticket_service
            .get_ticket(
                ticket_key
            )
        )

        if (
            not lookup.ok
            or lookup.ticket
            is None
        ):

            return (
                TicketMutationResult(
                    ok=False,

                    status=(
                        lookup.status
                    ),

                    provider="mock",

                    ticket_key=(
                        (
                            lookup.ticket.key
                        )
                        if lookup.ticket
                        is not None
                        else None
                    ),

                    operation=(
                        "add_comment"
                    ),

                    error=(
                        lookup.error
                    ),
                )
            )

        normalized_key = (
            lookup.ticket.key
        )

        comments = (
            self.ticket_service
            .comments
            .setdefault(
                normalized_key,
                [],
            )
        )

        comment_id = (
            f"mock-comment-"
            f"{len(comments) + 1}"
        )

        timestamp = (
            utc_timestamp()
        )

        comments.append(
            TicketComment(
                id=(
                    comment_id
                ),

                author=(
                    "Agentic ITSM"
                ),

                body=(
                    normalized_comment
                ),

                created_at=(
                    timestamp
                ),

                updated_at=(
                    timestamp
                ),
            )
        )

        lookup.ticket.updated_at = (
            timestamp
        )

        return (
            TicketMutationResult(
                ok=True,

                status="success",

                provider="mock",

                ticket_key=(
                    normalized_key
                ),

                operation=(
                    "add_comment"
                ),

                changed=True,

                comment_id=(
                    comment_id
                ),

                message=(
                    "Ticket comment added."
                ),

                error=None,
            )
        )


class JiraTicketMutationService(
    TicketMutationService
):
    """
    Jira command adapter.

    It intentionally reuses the authenticated httpx client owned by
    JiraTicketService.

    This avoids:
        - duplicate credentials
        - duplicate configuration
        - duplicate HTTP pools
        - a second provider lifecycle

    JiraTicketService remains the query/read provider.
    """

    def __init__(
        self,
        ticket_service: JiraTicketService,
    ) -> None:

        if not isinstance(
            ticket_service,
            JiraTicketService,
        ):

            raise TypeError(
                "JiraTicketMutationService requires "
                "JiraTicketService."
            )

        self.ticket_service = (
            ticket_service
        )

    @staticmethod
    def _comment_adf(
        comment: str,
    ) -> dict:

        paragraphs = []

        for line in (
            comment.splitlines()
            or [
                comment,
            ]
        ):

            content = []

            if line:

                content.append(
                    {
                        "type":
                            "text",

                        "text":
                            line,
                    }
                )

            paragraphs.append(
                {
                    "type":
                        "paragraph",

                    "content":
                        content,
                }
            )

        return {
            "type":
                "doc",

            "version":
                1,

            "content":
                paragraphs,
        }

    def add_comment(
        self,
        ticket_key: str,
        comment: str,
    ) -> TicketMutationResult:

        (
            normalized_key,
            key_error,
        ) = (
            self.ticket_service
            ._normalize_ticket_key(
                ticket_key
            )
        )

        if normalized_key is None:

            return (
                TicketMutationResult(
                    ok=False,

                    status="denied",

                    provider="jira",

                    operation=(
                        "add_comment"
                    ),

                    error=(
                        key_error
                    ),
                )
            )

        (
            normalized_comment,
            comment_error,
        ) = (
            normalize_ticket_comment(
                comment
            )
        )

        if normalized_comment is None:

            return (
                TicketMutationResult(
                    ok=False,

                    status="denied",

                    provider="jira",

                    ticket_key=(
                        normalized_key
                    ),

                    operation=(
                        "add_comment"
                    ),

                    error=(
                        comment_error
                    ),
                )
            )

        encoded_key = (
            quote(
                normalized_key,
                safe="",
            )
        )

        request_body = {
            "body":
                self._comment_adf(
                    normalized_comment
                ),
        }

        try:

            response = (
                self.ticket_service
                .client
                .post(
                    (
                        "/rest/api/3/"
                        f"issue/{encoded_key}/comment"
                    ),

                    json=(
                        request_body
                    ),
                )
            )

        except httpx.HTTPError:

            # A transport failure can occur after a remote server
            # receives the request.
            #
            # Do not claim that no mutation occurred.
            return (
                TicketMutationResult(
                    ok=False,

                    status="unknown",

                    provider="jira",

                    ticket_key=(
                        normalized_key
                    ),

                    operation=(
                        "add_comment"
                    ),

                    error=(
                        "The Jira comment request failed and "
                        "the remote mutation state is unknown."
                    ),
                )
            )

        if (
            response.status_code
            == 404
        ):

            return (
                TicketMutationResult(
                    ok=False,

                    status="not_found",

                    provider="jira",

                    ticket_key=(
                        normalized_key
                    ),

                    operation=(
                        "add_comment"
                    ),

                    error=(
                        f"Ticket '{normalized_key}' "
                        "was not found."
                    ),
                )
            )

        if (
            response.status_code
            in {
                401,
                403,
            }
        ):

            return (
                TicketMutationResult(
                    ok=False,

                    status="error",

                    provider="jira",

                    ticket_key=(
                        normalized_key
                    ),

                    operation=(
                        "add_comment"
                    ),

                    error=(
                        "Jira authentication or "
                        "authorization failed."
                    ),
                )
            )

        if (
            response.status_code
            == 400
        ):

            return (
                TicketMutationResult(
                    ok=False,

                    status="denied",

                    provider="jira",

                    ticket_key=(
                        normalized_key
                    ),

                    operation=(
                        "add_comment"
                    ),

                    error=(
                        "Jira rejected the supplied "
                        "ticket comment."
                    ),
                )
            )

        if not (
            200
            <= response.status_code
            < 300
        ):

            status = (
                "unknown"
                if response.status_code
                >= 500
                else "error"
            )

            return (
                TicketMutationResult(
                    ok=False,

                    status=(
                        status
                    ),

                    provider="jira",

                    ticket_key=(
                        normalized_key
                    ),

                    operation=(
                        "add_comment"
                    ),

                    error=(
                        "Jira returned HTTP "
                        f"{response.status_code} "
                        "while adding the comment."
                    ),
                )
            )

        # --------------------------------------------------------
        # SUCCESS
        #
        # The remote mutation is already authoritative at this
        # point. Failure to decode optional response metadata must
        # not turn a successful mutation into a retryable failure.
        # --------------------------------------------------------

        comment_id = None

        try:

            payload = (
                response.json()
            )

        except ValueError:

            payload = None

        if isinstance(
            payload,
            dict,
        ):

            comment_id = (
                self.ticket_service
                ._string_field(
                    payload.get(
                        "id"
                    )
                )
            )

        return (
            TicketMutationResult(
                ok=True,

                status="success",

                provider="jira",

                ticket_key=(
                    normalized_key
                ),

                operation=(
                    "add_comment"
                ),

                changed=True,

                comment_id=(
                    comment_id
                ),

                message=(
                    "Ticket comment added."
                ),

                error=None,
            )
        )


def build_ticket_mutation_service(
    ticket_service: TicketService,
) -> TicketMutationService:
    """
    Build the command provider from the already-selected query
    provider.

    Provider selection remains single-source-of-truth.

    We intentionally do NOT read configuration again here.
    """

    if isinstance(
        ticket_service,
        MockTicketService,
    ):

        return (
            MockTicketMutationService(
                ticket_service
            )
        )

    if isinstance(
        ticket_service,
        JiraTicketService,
    ):

        return (
            JiraTicketMutationService(
                ticket_service
            )
        )

    raise RuntimeError(
        "Unsupported TicketService implementation for "
        "ticket mutations: "
        f"{type(ticket_service).__name__}"
    )
