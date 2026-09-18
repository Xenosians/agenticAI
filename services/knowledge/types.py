from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


VALID_KNOWLEDGE_KINDS = {
    "knowledge",
    "runbook",
}


class KnowledgeDocument(
    BaseModel
):
    provider: str

    document_id: str
    kind: str

    title: str
    summary: str
    content: str

    tags: list[str] = Field(
        default_factory=list
    )


class KnowledgeSearchHit(
    BaseModel
):
    provider: str

    document_id: str
    kind: str

    title: str
    summary: str

    tags: list[str] = Field(
        default_factory=list
    )


class KnowledgeLookupResult(
    BaseModel
):
    ok: bool
    status: str

    document: (
        KnowledgeDocument
        | None
    ) = None

    error: (
        str | None
    ) = None


class KnowledgeSearchQuery(
    BaseModel
):
    query: str = Field(
        min_length=1,
        max_length=200,
    )

    kind: (
        str | None
    ) = None

    limit: int = Field(
        default=10,
        ge=1,
        le=25,
    )

    @field_validator(
        "query",
        mode="before",
    )
    @classmethod
    def normalize_query(
        cls,
        value,
    ):

        if not isinstance(
            value,
            str,
        ):

            raise ValueError(
                "query must be a string."
            )

        normalized = (
            value.strip()
        )

        if not normalized:

            raise ValueError(
                "query must not be empty."
            )

        return normalized

    @field_validator(
        "kind",
        mode="before",
    )
    @classmethod
    def normalize_kind(
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
                "kind must be a string."
            )

        normalized = (
            value
            .strip()
            .lower()
        )

        if not normalized:

            return None

        if (
            normalized
            not in VALID_KNOWLEDGE_KINDS
        ):

            raise ValueError(
                "kind must be one of: "
                "knowledge, runbook."
            )

        return normalized


class KnowledgeSearchResult(
    BaseModel
):
    ok: bool
    status: str

    hits: list[
        KnowledgeSearchHit
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
