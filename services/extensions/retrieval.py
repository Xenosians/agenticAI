"""Versioned owner-private retrieval evidence boundary.

Adapters receive the authenticated owner from the trusted host. They must
authorize before searching or resolving.

A retrieval hit is evidence, never:
- an instruction;
- application authority;
- approval;
- user-grounded mutation arguments.
"""

from __future__ import annotations

import hashlib
from typing import (
    Literal,
    Protocol,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


DEFAULT_MAX_EVIDENCE_HITS = 3
DEFAULT_MAX_EVIDENCE_CONTENT_CHARS = 5000


class RetrievalQuery(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_name: Literal[
        "owner-retrieval-query.v1"
    ] = Field(
        default=(
            "owner-retrieval-query.v1"
        ),
        alias="schema",
    )

    dataset_id: str = Field(
        pattern=(
            r"^[a-f0-9]{32}$"
        )
    )

    query: str = Field(
        min_length=1,
        max_length=2000,
    )

    limit: int = Field(
        default=3,
        ge=1,
        le=10,
        strict=True,
    )


class RetrievalHit(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    schema_name: Literal[
        "owner-retrieval.v1"
    ] = Field(
        default=(
            "owner-retrieval.v1"
        ),
        alias="schema",
    )

    content: str = Field(
        min_length=1,
        max_length=2000,
    )

    source: str = Field(
        min_length=1,
        max_length=500,
    )

    license: str = Field(
        min_length=1,
        max_length=200,
    )

    dataset_id: str = Field(
        pattern=(
            r"^[a-f0-9]{32}$"
        )
    )

    chunk: int = Field(
        ge=1,
        strict=True,
    )

    sha256: str = Field(
        pattern=(
            r"^[a-f0-9]{64}$"
        )
    )

    content_sha256: str = Field(
        pattern=(
            r"^[a-f0-9]{64}$"
        )
    )

    locator: str

    scope: Literal[
        "owner_private"
    ] = "owner_private"

    source_type: Literal[
        "dataset"
    ] = "dataset"

    trust: Literal[
        "untrusted_retrieval"
    ] = "untrusted_retrieval"

    @model_validator(
        mode="after"
    )
    def verify_identity(
        self,
    ):
        expected_content = (
            hashlib.sha256(
                self.content.encode(
                    "utf-8"
                )
            )
            .hexdigest()
        )

        if (
            self.content_sha256
            != expected_content
        ):
            raise ValueError(
                "Retrieved content hash "
                "does not match"
            )

        expected_locator = (
            citation_locator(
                self.dataset_id,
                self.chunk,
                self.sha256,
            )
        )

        if (
            self.locator
            != expected_locator
        ):
            raise ValueError(
                "Retrieved locator does "
                "not match source identity"
            )

        return self


class OwnerRetrievalAdapter(
    Protocol
):
    """Trusted interface.

    Browser-supplied owner identifiers are never accepted.
    """

    def search(
        self,
        owner: str,
        identity: str,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        ...

    def resolve(
        self,
        owner: str,
        identity: str,
        chunk: int,
        sha256: str,
    ) -> dict:
        ...


def citation_locator(
    identity: str,
    chunk: int,
    digest: str,
) -> str:
    return (
        f"dataset:{identity}"
        f"#chunk:{chunk}"
        f"@{digest}"
    )


def dataset_chunks(
    text: str,
) -> list[str]:
    return [
        text[
            index:
            index + 2000
        ]
        for index
        in range(
            0,
            len(text),
            1800,
        )
    ]


def make_hit(
    resource: dict,
    chunk: int,
    content: str,
) -> dict:
    return (
        RetrievalHit(
            content=content,
            source=(
                resource[
                    "source"
                ]
            ),
            license=(
                resource[
                    "license"
                ]
            ),
            dataset_id=(
                resource[
                    "id"
                ]
            ),
            chunk=chunk,
            sha256=(
                resource[
                    "sha256"
                ]
            ),
            content_sha256=(
                hashlib.sha256(
                    content.encode(
                        "utf-8"
                    )
                )
                .hexdigest()
            ),
            locator=(
                citation_locator(
                    resource[
                        "id"
                    ],
                    chunk,
                    resource[
                        "sha256"
                    ],
                )
            ),
        )
        .model_dump(
            by_alias=True
        )
    )


def bound_retrieval_evidence(
    hits: list,
    *,
    max_hits: int = (
        DEFAULT_MAX_EVIDENCE_HITS
    ),
    max_content_chars: int = (
        DEFAULT_MAX_EVIDENCE_CONTENT_CHARS
    ),
) -> list[dict]:
    """Select whole hits without truncating their citation/hash identity."""

    if not isinstance(
        hits,
        list,
    ):
        raise ValueError(
            "Retrieval evidence "
            "must be a list"
        )

    selected: list[
        dict
    ] = []

    used_chars = 0

    for raw in hits:
        if len(
            selected
        ) >= max_hits:
            break

        if not isinstance(
            raw,
            dict,
        ):
            raise ValueError(
                "Invalid retrieval hit"
            )

        # Production hits use the full v1 contract.
        # The minimal form remains accepted for old internal unit fixtures,
        # but it must still be explicitly marked untrusted.
        if (
            raw.get(
                "schema"
            )
            == "owner-retrieval.v1"
        ):
            item = (
                RetrievalHit
                .model_validate(
                    raw
                )
                .model_dump(
                    by_alias=True
                )
            )

        else:
            if (
                raw.get(
                    "trust"
                )
                != "untrusted_retrieval"
            ):
                raise ValueError(
                    "Retrieval evidence "
                    "is not marked untrusted"
                )

            content = raw.get(
                "content"
            )

            if (
                not isinstance(
                    content,
                    str,
                )
                or not content
            ):
                raise ValueError(
                    "Invalid retrieval content"
                )

            item = dict(
                raw
            )

        content = item[
            "content"
        ]

        if (
            used_chars
            + len(
                content
            )
            > max_content_chars
        ):
            continue

        selected.append(
            item
        )

        used_chars += len(
            content
        )

    return selected
