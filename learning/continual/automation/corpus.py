from __future__ import annotations

import hashlib
import json
import math

from pathlib import Path
from typing import Any

from learning.paths import (
    REPOSITORY_ROOT,
    RUNTIME_LEARNING_ROOT,
)
from learning.continual.storage import (
    canonical_json,
    sha256_file,
)
from learning.continual.automation.config import (
    ContinualAutomationSettings,
    load_declared_corpus_sources,
)
from learning.continual.automation.store import (
    ContinualAutomationStore,
)
from learning.continual.automation.types import (
    CorpusPage,
    CorpusSource,
)


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _resolve(
    value: str,
) -> Path:
    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():
        path = (
            REPOSITORY_ROOT
            / path
        )

    return path.resolve()


def discover_semantic_code_sources(
    *,
    root: Path | None = None,
) -> list[CorpusSource]:
    """
    Discover only the latest immutable corpus directories already produced
    by the governed semantic code indexer.

    No web scraping occurs here.
    """
    if root is None:
        root = (
            RUNTIME_LEARNING_ROOT
            / "code-corpus"
        )

    root = (
        root
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        return []

    # Lazy import keeps ordinary text/book paging independent from
    # the semantic-code subsystem when no code corpus is configured.
    from learning.code_corpus.types import (
        CodeCorpusManifest,
    )

    latest_by_repository: dict[
        str,
        tuple[str, Path, CodeCorpusManifest],
    ] = {}

    for manifest_path in root.glob(
        "**/manifest.json"
    ):
        try:
            manifest = (
                CodeCorpusManifest
                .model_validate_json(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )
            )
        except Exception:
            continue

        source = manifest.source

        if not (
            source.training_eligible
            and manifest.training_eligible_chunk_count
            > 0
        ):
            continue

        chunks_path = (
            manifest_path.parent
            / "chunks.jsonl"
        )

        if not chunks_path.is_file():
            continue

        repository = (
            source.logical_repository
        )

        rank = (
            manifest.created_at,
            manifest_path.parent,
            manifest,
        )

        current = latest_by_repository.get(
            repository
        )

        if (
            current is None
            or rank[0]
            > current[0]
        ):
            latest_by_repository[
                repository
            ] = rank

    sources = []

    for (
        repository,
        (
            _created_at,
            directory,
            manifest,
        ),
    ) in sorted(
        latest_by_repository.items()
    ):
        target_role = (
            "hub"
            if manifest.source.source_kind
            == "general"
            else "developer"
        )

        sources.append(
            CorpusSource(
                source_id=(
                    "semantic-code:"
                    + repository
                ),
                kind=(
                    "semantic_code_jsonl"
                ),
                path=str(
                    directory
                    / "chunks.jsonl"
                ),
                target_role=(
                    target_role
                ),
                audience=[
                    target_role
                ],
                trusted=True,
                training_eligible=True,
                records_per_page=1,
            )
        )

    return sources


def register_sources(
    *,
    store: ContinualAutomationStore,
    settings: ContinualAutomationSettings,
) -> list[CorpusSource]:
    declared = (
        load_declared_corpus_sources(
            settings
        )
    )

    discovered = (
        discover_semantic_code_sources()
    )

    by_id = {
        source.source_id:
            source
        for source in [
            *discovered,
            *declared,
        ]
    }

    result = []

    for source in sorted(
        by_id.values(),
        key=lambda item:
            item.source_id,
    ):
        path = _resolve(
            source.path
        )

        if not path.is_file():
            continue

        source_sha = sha256_file(
            path
        )

        if not source_sha:
            continue

        total_units = None

        if source.kind == "semantic_code_jsonl":
            total_units = sum(
                1
                for line in path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )

        elif source.kind == "text":
            total_units = math.ceil(
                path.stat().st_size
                / source.page_chars
            )

        store.upsert_corpus_source(
            source,
            source_sha256=(
                source_sha
            ),
            total_units=(
                total_units
            ),
        )

        result.append(
            source
        )

    return result


def _page_id(
    *,
    source_id: str,
    source_sha256: str,
    page_index: int,
    content: str,
) -> str:
    payload = {
        "source_id":
            source_id,

        "source_sha256":
            source_sha256,

        "page_index":
            page_index,

        "content_sha256":
            _sha256_text(
                content
            ),
    }

    return (
        "corpus-page-"
        + _sha256_text(
            canonical_json(
                payload
            )
        )[:24]
    )


def _code_pages(
    *,
    row: dict[str, Any],
    limit: int,
) -> tuple[
    list[CorpusPage],
    int,
]:
    """
    Stream JSONL instead of loading the full corpus into memory.

    cursor is the next physical JSONL line index to inspect.
    """
    path = _resolve(
        row[
            "path"
        ]
    )

    cursor = int(
        row[
            "cursor"
        ]
    )

    source_sha = str(
        row[
            "source_sha256"
        ]
    )

    pages = []
    next_cursor = cursor

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for index, line in enumerate(
            handle
        ):
            if index < cursor:
                continue

            next_cursor = (
                index
                + 1
            )

            if not line.strip():
                continue

            raw = json.loads(
                line
            )

            content = raw.get(
                "content"
            )

            if not (
                isinstance(
                    content,
                    str,
                )
                and content.strip()
            ):
                continue

            if raw.get(
                "training_eligible"
            ) is not True:
                continue

            title = (
                f"{raw.get('logical_repository', 'code')}:"
                f"{raw.get('relative_path', '')}:"
                f"{raw.get('symbol', '')}"
            )

            pages.append(
                CorpusPage(
                    page_id=_page_id(
                        source_id=(
                            row[
                                "source_id"
                            ]
                        ),
                        source_sha256=(
                            source_sha
                        ),
                        page_index=(
                            index
                        ),
                        content=(
                            content
                        ),
                    ),
                    source_id=(
                        row[
                            "source_id"
                        ]
                    ),
                    source_sha256=(
                        source_sha
                    ),
                    page_index=(
                        index
                    ),
                    cursor_start=(
                        index
                    ),
                    cursor_end=(
                        index
                        + 1
                    ),
                    target_role=(
                        row[
                            "target_role"
                        ]
                    ),
                    title=title,
                    content=(
                        content
                    ),
                    metadata={
                        "logical_repository":
                            raw.get(
                                "logical_repository"
                            ),

                        "relative_path":
                            raw.get(
                                "relative_path"
                            ),

                        "language":
                            raw.get(
                                "language"
                            ),

                        "symbol":
                            raw.get(
                                "symbol"
                            ),

                        "symbol_kind":
                            raw.get(
                                "symbol_kind"
                            ),

                        "curriculum_hints":
                            raw.get(
                                "curriculum_hints",
                                [],
                            ),
                    },
                    training_eligible=True,
                )
            )

            if len(
                pages
            ) >= limit:
                break

    return (
        pages,
        next_cursor,
    )

def _text_pages(
    *,
    row: dict[str, Any],
    limit: int,
) -> tuple[
    list[CorpusPage],
    int,
]:
    """
    Stream book/reference material by byte windows.

    cursor is a byte offset, so even multi-gigabyte sources are not loaded
    into memory. The page boundary is approximate; UTF-8 partial bytes are
    ignored at the edge and the next page continues from the next byte.
    """
    path = _resolve(
        row[
            "path"
        ]
    )

    page_bytes = int(
        row[
            "page_chars"
        ]
    )

    cursor = int(
        row[
            "cursor"
        ]
    )

    source_sha = str(
        row[
            "source_sha256"
        ]
    )

    result = []
    next_cursor = cursor
    file_size = (
        path.stat().st_size
    )

    with path.open(
        "rb"
    ) as handle:
        handle.seek(
            cursor
        )

        while (
            handle.tell()
            < file_size
            and len(
                result
            )
            < limit
        ):
            start = (
                handle.tell()
            )

            blob = handle.read(
                page_bytes
            )

            end = (
                handle.tell()
            )

            next_cursor = end

            if not blob:
                break

            content = (
                blob.decode(
                    "utf-8",
                    errors="ignore",
                )
                .strip()
            )

            if not content:
                continue

            page_index = (
                start
                // page_bytes
            )

            result.append(
                CorpusPage(
                    page_id=_page_id(
                        source_id=(
                            row[
                                "source_id"
                            ]
                        ),
                        source_sha256=(
                            source_sha
                        ),
                        page_index=(
                            page_index
                        ),
                        content=(
                            content
                        ),
                    ),
                    source_id=(
                        row[
                            "source_id"
                        ]
                    ),
                    source_sha256=(
                        source_sha
                    ),
                    page_index=(
                        page_index
                    ),
                    cursor_start=(
                        start
                    ),
                    cursor_end=(
                        end
                    ),
                    target_role=(
                        row[
                            "target_role"
                        ]
                    ),
                    title=(
                        f"{path.name} page "
                        f"{page_index + 1}"
                    ),
                    content=(
                        content
                    ),
                    metadata={
                        "path":
                            str(
                                path
                            ),
                    },
                    training_eligible=True,
                )
            )

    return (
        result,
        next_cursor,
    )

def next_progressive_pages(
    *,
    store: ContinualAutomationStore,
    limit: int,
) -> list[CorpusPage]:
    """
    Read only a small advancing window. The default controller asks for
    ten pages/chunks per cycle, matching the "read a book a few pages at
    a time" behavior.
    """
    if limit <= 0:
        return []

    rows = (
        store
        .corpus_source_rows()
    )

    if not rows:
        return []

    pages: list[
        CorpusPage
    ] = []

    source_index = 0

    while (
        len(
            pages
        )
        < limit
        and rows
    ):
        row = rows[
            source_index
            % len(
                rows
            )
        ]

        remaining = (
            limit
            - len(
                pages
            )
        )

        if row[
            "kind"
        ] == "semantic_code_jsonl":
            selected, cursor = (
                _code_pages(
                    row=row,
                    limit=1,
                )
            )

        else:
            selected, cursor = (
                _text_pages(
                    row=row,
                    limit=1,
                )
            )

        if selected:
            page = selected[
                0
            ]

            store.remember_page(
                page
            )

            pages.append(
                page
            )

            store.set_corpus_cursor(
                source_id=(
                    row[
                        "source_id"
                    ]
                ),
                cursor=(
                    cursor
                ),
            )

            row[
                "cursor"
            ] = cursor

        else:
            # Exhausted source.
            rows.remove(
                row
            )

            if not rows:
                break

            source_index -= 1

        source_index += 1

    return pages


def study_split(
    content: str,
    *,
    prefix_fraction: float = 0.65,
) -> tuple[
    str,
    str,
]:
    """
    Deterministic prefix -> continuation curriculum.

    This avoids pretending a source chunk is an instruction-following
    answer while still allowing the causal model to study trusted text/code.
    """
    normalized = (
        content.strip()
    )

    if len(
        normalized
    ) < 80:
        midpoint = max(
            1,
            len(
                normalized
            )
            // 2,
        )

    else:
        midpoint = max(
            40,
            int(
                len(
                    normalized
                )
                * prefix_fraction
            ),
        )

    midpoint = min(
        midpoint,
        len(
            normalized
        )
        - 1,
    )

    return (
        normalized[
            :midpoint
        ],
        normalized[
            midpoint:
        ],
    )
