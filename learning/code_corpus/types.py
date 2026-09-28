from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SourceKind = Literal[
    "general",
    "project",
]

RevisionKind = Literal[
    "git",
    "content_snapshot",
]


class SourceSnapshot(BaseModel):
    """
    Immutable identity of the source tree being indexed.

    Git repositories use the current commit plus dirty-state evidence.
    Non-Git sources receive a deterministic content snapshot hash.

    A dirty project checkout remains indexable for inspection, but is not
    training-eligible by default.
    """

    model_config = ConfigDict(extra="forbid")

    source_kind: SourceKind
    logical_repository: str
    root: str

    revision_kind: RevisionKind
    revision_id: str

    git_commit_sha: str | None = None
    git_dirty: bool = False
    git_changed_path_count: int = 0

    training_eligible: bool = False


class CodeCorpusChunk(BaseModel):
    """
    One semantically split code/config/prompt record.

    Splitting occurs before model tokenization:

        repository
            -> file
            -> symbol
            -> logical/size part when required
            -> tokenizer later
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="semantic-code-chunk.v1",
        alias="schema",
    )

    chunk_id: str

    source_kind: SourceKind
    logical_repository: str

    revision_kind: RevisionKind
    revision_id: str
    git_commit_sha: str | None = None
    git_dirty: bool = False

    relative_path: str
    language: str

    symbol: str
    symbol_kind: str
    parent_symbol: str | None = None

    semantic_depth: int = Field(ge=0)
    part_index: int = Field(default=1, ge=1)
    part_count: int = Field(default=1, ge=1)

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    content: str
    content_sha256: str

    audiences: list[str] = Field(default_factory=list)
    curriculum_hints: list[str] = Field(default_factory=list)

    training_eligible: bool = False


class CodeCorpusManifest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="semantic-code-corpus-manifest.v1",
        alias="schema",
    )

    corpus_id: str
    created_at: str

    source: SourceSnapshot

    chunk_count: int
    content_sha256: str

    language_counts: dict[str, int] = Field(default_factory=dict)
    audience_counts: dict[str, int] = Field(default_factory=dict)
    symbol_kind_counts: dict[str, int] = Field(default_factory=dict)

    training_eligible_chunk_count: int = 0
    excluded_file_count: int = 0
