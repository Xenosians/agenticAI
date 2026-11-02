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
import re
from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


DEFAULT_MAX_EVIDENCE_HITS = 3
DEFAULT_MAX_EVIDENCE_CONTENT_CHARS = 5000


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


# How a dataset's text is cut into citable chunks. A dataset records the
# chunking it was admitted with, so its citations always reopen the same text.
WINDOW_CHUNKING = "window-2000.v1"
PASSAGE_CHUNKING = "passage-1200.v1"
PASSAGE_CHARS = 1200
# A heading is a short line that does not end a sentence: a title, a question
# or a lead-in ending in a colon. It starts a passage and stays with the text
# after it. List items continue the passage they are in.
HEADING_CHARS = 80
BODY_LINE_ENDINGS = (".", "!", ";", ",")
_LIST_ITEM = re.compile(r"(?:[-*+\u2022]|\d{1,3}[.)]|[a-z][.)])\s")
_SENTENCE_BREAK = re.compile(r"[.!?](?=\s)")


def dataset_chunks(
    text: str,
    chunking: str = WINDOW_CHUNKING,
) -> list[str]:
    if chunking == WINDOW_CHUNKING:
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
    if chunking == PASSAGE_CHUNKING:
        return _passages(text)
    raise ValueError("Unknown dataset chunking")


def _passages(text: str) -> list[str]:
    """Passages that follow the text's own structure, each an exact span of it.

    Blank lines end a passage. A heading line (short, without a closing full
    stop: a title, a question or a lead-in ending in a colon) starts a new
    passage and stays with the text after it. Lines are packed up to
    PASSAGE_CHARS; a longer line is split at sentence ends, and a sentence
    longer than a passage at the limit, so no text is lost.
    """
    passages: list[str] = []
    start = end = None
    has_body = False

    def close():
        nonlocal start, end, has_body
        if start is not None:
            passages.append(text[start:end])
        start = end = None
        has_body = False

    position = 0
    for line in text.split("\n"):
        line_start, line_end = position, position + len(line)
        position = line_end + 1
        stripped = line.strip()
        if not stripped:
            if has_body:
                close()
            continue
        line_start += len(line) - len(line.lstrip())
        line_end -= len(line) - len(line.rstrip())
        heading = stripped.startswith("#") or (
            len(stripped) <= HEADING_CHARS and not stripped.endswith(BODY_LINE_ENDINGS)
            and not _LIST_ITEM.match(stripped))
        if heading and has_body:
            close()
        if start is not None and line_end - start > PASSAGE_CHARS:
            close()
        if line_end - line_start > PASSAGE_CHARS:
            close()
            for piece_start, piece_end in _bounded_spans(text, line_start, line_end):
                passages.append(text[piece_start:piece_end])
            continue
        if start is None:
            start = line_start
        end = line_end
        has_body = has_body or not heading
    close()
    return passages


_PAGE_TITLE = re.compile(r"\A#{1,6}[ 	]+(.+)")


def passage_titles(chunks: list[str]) -> list[str]:
    """The title of the page each passage belongs to: the nearest "# Title" line at or before it.

    Crawled pages and Markdown files open with such a line; a passage cut
    from the middle of a page then still carries the page's subject.
    """
    titles, current = [], ""
    for chunk in chunks:
        match = _PAGE_TITLE.match(chunk)
        if match:
            current = match.group(1).strip()
        titles.append(current)
    return titles


def _bounded_spans(text: str, start: int, end: int):
    """Cut one long line into spans of at most PASSAGE_CHARS, at sentence ends where possible."""
    while start < end:
        limit = min(start + PASSAGE_CHARS, end)
        cut = limit
        if limit < end:
            ends = [m.end() for m in _SENTENCE_BREAK.finditer(text, start, limit)]
            if ends:
                cut = ends[-1]
        piece_end = cut
        while piece_end > start and text[piece_end - 1].isspace():
            piece_end -= 1
        yield start, piece_end
        start = cut
        while start < end and text[start].isspace():
            start += 1


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
