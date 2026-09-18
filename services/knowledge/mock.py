import re

from .base import (
    KnowledgeService,
)

from .types import (
    KnowledgeDocument,
    KnowledgeLookupResult,
    KnowledgeSearchHit,
    KnowledgeSearchQuery,
    KnowledgeSearchResult,
)


class MockKnowledgeService(
    KnowledgeService
):

    def __init__(
        self,
    ) -> None:

        self.documents = {
            "KB-001": (
                KnowledgeDocument(
                    provider="mock",

                    document_id="KB-001",

                    kind="knowledge",

                    title=(
                        "VPN access troubleshooting"
                    ),

                    summary=(
                        "Common checks for users who "
                        "cannot connect to the corporate VPN."
                    ),

                    content=(
                        "VPN troubleshooting guidance:\n"
                        "1. Confirm that the user's account "
                        "is enabled and not locked.\n"
                        "2. Confirm that the user has the "
                        "configured VPN access entitlement.\n"
                        "3. Confirm that the endpoint has "
                        "network connectivity.\n"
                        "4. Confirm that the VPN client is "
                        "configured for the expected gateway.\n"
                        "5. Escalate authentication or gateway "
                        "failures when the preceding checks pass."
                    ),

                    tags=[
                        "vpn",
                        "access",
                        "network",
                        "troubleshooting",
                    ],
                )
            ),

            "KB-002": (
                KnowledgeDocument(
                    provider="mock",

                    document_id="KB-002",

                    kind="knowledge",

                    title=(
                        "Active Directory account lockout"
                    ),

                    summary=(
                        "Explanation of account lockout state "
                        "and how it differs from account disablement."
                    ),

                    content=(
                        "An account lockout and an account "
                        "disablement are different states. "
                        "A locked account may remain enabled. "
                        "Unlocking clears lockout state, while "
                        "enabling or disabling changes account "
                        "lifecycle state."
                    ),

                    tags=[
                        "active-directory",
                        "account",
                        "lockout",
                        "identity",
                    ],
                )
            ),

            "KB-003": (
                KnowledgeDocument(
                    provider="mock",

                    document_id="KB-003",

                    kind="knowledge",

                    title=(
                        "Managed device assignment"
                    ),

                    summary=(
                        "How managed asset ownership is represented "
                        "inside the IT asset inventory."
                    ),

                    content=(
                        "Managed devices may be assigned to one "
                        "logical owner. Available devices have no "
                        "owner. Assignment changes inventory "
                        "ownership and should be treated as a "
                        "governed mutation."
                    ),

                    tags=[
                        "asset",
                        "device",
                        "inventory",
                        "assignment",
                    ],
                )
            ),

            "RB-001": (
                KnowledgeDocument(
                    provider="mock",

                    document_id="RB-001",

                    kind="runbook",

                    title=(
                        "VPN access restoration triage"
                    ),

                    summary=(
                        "Read-only triage sequence for a user "
                        "reporting unavailable VPN access."
                    ),

                    content=(
                        "VPN access restoration triage:\n"
                        "1. Check the user's account status.\n"
                        "2. Check the user's VPN access state.\n"
                        "3. Review relevant ticket context.\n"
                        "4. Confirm that no explicit restriction "
                        "or exclusion is present.\n"
                        "5. If a state-changing action is required, "
                        "request it through the governed capability "
                        "and approval workflow."
                    ),

                    tags=[
                        "vpn",
                        "access",
                        "triage",
                        "runbook",
                    ],
                )
            ),

            "RB-002": (
                KnowledgeDocument(
                    provider="mock",

                    document_id="RB-002",

                    kind="runbook",

                    title=(
                        "Account lockout triage"
                    ),

                    summary=(
                        "Triage sequence for an account that may "
                        "be locked or disabled."
                    ),

                    content=(
                        "Account lockout triage:\n"
                        "1. Retrieve account status.\n"
                        "2. Determine whether the account is "
                        "locked, disabled, or both.\n"
                        "3. Do not treat unlocking and enabling "
                        "as equivalent operations.\n"
                        "4. If the user requested a mutation, "
                        "submit the exact governed operation "
                        "for approval."
                    ),

                    tags=[
                        "account",
                        "lockout",
                        "identity",
                        "runbook",
                    ],
                )
            ),
        }

    @staticmethod
    def _normalize_id(
        value: str,
        *,
        field_name: str,
    ) -> tuple[
        str | None,
        str | None,
    ]:

        if not isinstance(
            value,
            str,
        ):

            return (
                None,
                f"{field_name} must be a string.",
            )

        normalized = (
            value
            .strip()
            .upper()
        )

        if not normalized:

            return (
                None,
                f"{field_name} must not be empty.",
            )

        return (
            normalized,
            None,
        )

    @staticmethod
    def _tokens(
        value: str,
    ) -> set[str]:

        return {
            token

            for token
            in re.findall(
                r"[A-Za-z0-9_-]+",
                value.casefold(),
            )

            if token
        }

    @classmethod
    def _score(
        cls,
        document: KnowledgeDocument,
        query: str,
    ) -> int:

        query_tokens = (
            cls._tokens(
                query
            )
        )

        if not query_tokens:

            return 0

        title_tokens = (
            cls._tokens(
                document.title
            )
        )

        summary_tokens = (
            cls._tokens(
                document.summary
            )
        )

        content_tokens = (
            cls._tokens(
                document.content
            )
        )

        tag_tokens = {
            tag.casefold()

            for tag
            in document.tags
        }

        score = 0

        score += (
            4
            * len(
                query_tokens
                & title_tokens
            )
        )

        score += (
            3
            * len(
                query_tokens
                & tag_tokens
            )
        )

        score += (
            2
            * len(
                query_tokens
                & summary_tokens
            )
        )

        score += len(
            query_tokens
            & content_tokens
        )

        return score

    def search(
        self,
        query: KnowledgeSearchQuery,
    ) -> KnowledgeSearchResult:

        if not isinstance(
            query,
            KnowledgeSearchQuery,
        ):

            return (
                KnowledgeSearchResult(
                    ok=False,

                    status="denied",

                    error=(
                        "Invalid knowledge search query."
                    ),
                )
            )

        scored = []

        for document in (
            self.documents.values()
        ):

            if (
                query.kind
                is not None
                and document.kind
                != query.kind
            ):

                continue

            score = (
                self._score(
                    document,
                    query.query,
                )
            )

            if score <= 0:

                continue

            scored.append(
                (
                    score,
                    document,
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].document_id,
            )
        )

        truncated = (
            len(
                scored
            )
            > query.limit
        )

        selected = (
            scored[
                :query.limit
            ]
        )

        hits = [
            KnowledgeSearchHit(
                provider=(
                    document.provider
                ),

                document_id=(
                    document.document_id
                ),

                kind=(
                    document.kind
                ),

                title=(
                    document.title
                ),

                summary=(
                    document.summary
                ),

                tags=list(
                    document.tags
                ),
            )

            for (
                _,
                document,
            )
            in selected
        ]

        return (
            KnowledgeSearchResult(
                ok=True,

                status="success",

                hits=hits,

                count=len(
                    hits
                ),

                truncated=truncated,

                error=None,
            )
        )

    def get_knowledge(
        self,
        document_id: str,
    ) -> KnowledgeLookupResult:

        (
            normalized_id,
            error,
        ) = (
            self._normalize_id(
                document_id,
                field_name=(
                    "document_id"
                ),
            )
        )

        if normalized_id is None:

            return (
                KnowledgeLookupResult(
                    ok=False,

                    status="denied",

                    error=error,
                )
            )

        document = (
            self.documents.get(
                normalized_id
            )
        )

        if (
            document is None
            or document.kind
            != "knowledge"
        ):

            return (
                KnowledgeLookupResult(
                    ok=False,

                    status="not_found",

                    error=(
                        f"Knowledge document "
                        f"'{normalized_id}' was not found."
                    ),
                )
            )

        return (
            KnowledgeLookupResult(
                ok=True,

                status="success",

                document=document,

                error=None,
            )
        )

    def get_runbook(
        self,
        runbook_id: str,
    ) -> KnowledgeLookupResult:

        (
            normalized_id,
            error,
        ) = (
            self._normalize_id(
                runbook_id,
                field_name=(
                    "runbook_id"
                ),
            )
        )

        if normalized_id is None:

            return (
                KnowledgeLookupResult(
                    ok=False,

                    status="denied",

                    error=error,
                )
            )

        document = (
            self.documents.get(
                normalized_id
            )
        )

        if (
            document is None
            or document.kind
            != "runbook"
        ):

            return (
                KnowledgeLookupResult(
                    ok=False,

                    status="not_found",

                    error=(
                        f"Runbook '{normalized_id}' "
                        "was not found."
                    ),
                )
            )

        return (
            KnowledgeLookupResult(
                ok=True,

                status="success",

                document=document,

                error=None,
            )
        )
