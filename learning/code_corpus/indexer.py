from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from learning.code_corpus.audiences import (
    classify_chunk_audiences,
    curriculum_hints_for_chunk,
)
from learning.code_corpus.chunking import (
    RawSemanticChunk,
    split_semantic_source,
)
from learning.code_corpus.policy import (
    detect_language,
    normalize_relative_path,
    should_include_file,
)
from learning.code_corpus.snapshot import (
    resolve_source_snapshot,
)
from learning.code_corpus.types import (
    CodeCorpusChunk,
    CodeCorpusManifest,
)
from learning.continual.storage import (
    canonical_json,
    immutable_write_json,
    immutable_write_jsonl,
)
from learning.paths import (
    RUNTIME_LEARNING_ROOT,
)


DEFAULT_CODE_CORPUS_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "code-corpus"
)


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _chunk_id(
    *,
    revision_id: str,
    relative_path: str,
    raw: RawSemanticChunk,
    content_sha256: str,
) -> str:
    payload = {
        "revision_id": revision_id,
        "relative_path": relative_path,
        "symbol": raw.symbol,
        "symbol_kind": raw.symbol_kind,
        "start_line": raw.start_line,
        "end_line": raw.end_line,
        "content_sha256": content_sha256,
    }

    digest = _sha256_text(
        canonical_json(
            payload
        )
    )

    return (
        "code-chunk-"
        + digest[
            :24
        ]
    )


def build_code_corpus(
    *,
    root: Path,
    logical_repository: str,
    source_kind: str,
    output_root: Path = DEFAULT_CODE_CORPUS_ROOT,
    max_chars: int = 8000,
    overlap_lines: int = 8,
    allow_dirty: bool = False,
    allow_unversioned: bool = False,
) -> CodeCorpusManifest:
    normalized_repository = (
        logical_repository
        .strip()
    )

    if not normalized_repository:
        raise ValueError(
            "logical_repository must not be empty"
        )

    if source_kind not in {
        "general",
        "project",
    }:
        raise ValueError(
            "source_kind must be 'general' or 'project'"
        )

    resolved_root = (
        root
        .expanduser()
        .resolve()
    )

    snapshot = resolve_source_snapshot(
        root=resolved_root,
        source_kind=source_kind,
        logical_repository=normalized_repository,
        allow_dirty=allow_dirty,
        allow_unversioned=allow_unversioned,
    )

    chunks: list[
        CodeCorpusChunk
    ] = []

    excluded_file_count = 0

    for path in sorted(
        candidate
        for candidate in resolved_root.rglob("*")
        if candidate.is_file()
    ):
        relative = path.relative_to(
            resolved_root
        )

        if not should_include_file(
            relative_path=relative,
            source_kind=source_kind,
        ):
            excluded_file_count += 1
            continue

        language = detect_language(
            relative
        )

        if language is None:
            excluded_file_count += 1
            continue

        try:
            text = path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            excluded_file_count += 1
            continue

        relative_text = normalize_relative_path(
            relative
        )

        audiences = classify_chunk_audiences(
            source_kind=source_kind,
            relative_path=relative_text,
        )

        raw_chunks = split_semantic_source(
            text=text,
            language=language,
            max_chars=max_chars,
            overlap_lines=overlap_lines,
        )

        part_totals: Counter[
            tuple[
                str,
                str,
            ]
        ] = Counter()

        for raw in raw_chunks:
            root_symbol = (
                raw.parent_symbol
                or raw.symbol.split(
                    "#part",
                    1,
                )[0]
            )

            part_totals[
                (
                    root_symbol,
                    raw.symbol_kind.replace(
                        "_part",
                        "",
                    ),
                )
            ] += 1

        part_seen: Counter[
            tuple[
                str,
                str,
            ]
        ] = Counter()

        for raw in raw_chunks:
            content_sha256 = _sha256_text(
                raw.content
            )

            root_symbol = (
                raw.parent_symbol
                or raw.symbol.split(
                    "#part",
                    1,
                )[0]
            )

            base_kind = raw.symbol_kind.replace(
                "_part",
                "",
            )

            key = (
                root_symbol,
                base_kind,
            )

            part_seen[
                key
            ] += 1

            training_eligible = (
                snapshot.training_eligible
            )

            chunks.append(
                CodeCorpusChunk(
                    chunk_id=_chunk_id(
                        revision_id=snapshot.revision_id,
                        relative_path=relative_text,
                        raw=raw,
                        content_sha256=content_sha256,
                    ),
                    source_kind=source_kind,
                    logical_repository=normalized_repository,
                    revision_kind=snapshot.revision_kind,
                    revision_id=snapshot.revision_id,
                    git_commit_sha=snapshot.git_commit_sha,
                    git_dirty=snapshot.git_dirty,
                    relative_path=relative_text,
                    language=language,
                    symbol=raw.symbol,
                    symbol_kind=raw.symbol_kind,
                    parent_symbol=raw.parent_symbol,
                    semantic_depth=raw.semantic_depth,
                    part_index=part_seen[key],
                    part_count=part_totals[key],
                    start_line=raw.start_line,
                    end_line=raw.end_line,
                    content=raw.content,
                    content_sha256=content_sha256,
                    audiences=audiences,
                    curriculum_hints=curriculum_hints_for_chunk(
                        audiences=audiences,
                        symbol_kind=raw.symbol_kind,
                    ),
                    training_eligible=training_eligible,
                )
            )

    if not chunks:
        raise ValueError(
            "No supported semantic code-corpus chunks were produced."
        )

    # Deterministic ordering is part of corpus identity.
    chunks.sort(
        key=lambda item: (
            item.relative_path,
            item.start_line,
            item.semantic_depth,
            item.symbol,
        )
    )

    serialized = "".join(
        json.dumps(
            item.model_dump(
                mode="json",
                by_alias=True,
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
        + "\n"
        for item in chunks
    )

    content_sha256 = hashlib.sha256(
        serialized.encode(
            "utf-8"
        )
    ).hexdigest()

    identity = {
        "source_kind": source_kind,
        "logical_repository": normalized_repository,
        "revision_kind": snapshot.revision_kind,
        "revision_id": snapshot.revision_id,
        "content_sha256": content_sha256,
        "max_chars": max_chars,
        "overlap_lines": overlap_lines,
    }

    corpus_id = (
        "code-corpus-"
        + _sha256_text(
            canonical_json(
                identity
            )
        )[
            :24
        ]
    )

    target = (
        output_root
        .expanduser()
        .resolve()
        / normalized_repository
        / corpus_id
    )

    manifest_path = (
        target
        / "manifest.json"
    )

    if manifest_path.is_file():
        return CodeCorpusManifest.model_validate_json(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

    language_counts = Counter(
        item.language
        for item in chunks
    )

    symbol_kind_counts = Counter(
        item.symbol_kind
        for item in chunks
    )

    audience_counts: Counter[
        str
    ] = Counter()

    for item in chunks:
        audience_counts.update(
            item.audiences
        )

    manifest = CodeCorpusManifest(
        corpus_id=corpus_id,
        created_at=_utc_now(),
        source=snapshot,
        chunk_count=len(
            chunks
        ),
        content_sha256=content_sha256,
        language_counts=dict(
            language_counts
        ),
        audience_counts=dict(
            audience_counts
        ),
        symbol_kind_counts=dict(
            symbol_kind_counts
        ),
        training_eligible_chunk_count=sum(
            1
            for item in chunks
            if item.training_eligible
        ),
        excluded_file_count=excluded_file_count,
    )

    target.mkdir(
        parents=True,
        exist_ok=False,
    )

    immutable_write_jsonl(
        target
        / "chunks.jsonl",
        chunks,
    )

    immutable_write_json(
        manifest_path,
        manifest,
    )

    return manifest
