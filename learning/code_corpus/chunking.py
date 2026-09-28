from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(
    frozen=True
)
class RawSemanticChunk:
    symbol: str
    symbol_kind: str
    parent_symbol: str | None

    semantic_depth: int

    start_line: int
    end_line: int

    content: str


def _line_slice(
    lines: list[str],
    start_line: int,
    end_line: int,
) -> str:
    return "".join(
        lines[
            start_line - 1:
            end_line
        ]
    )


def _split_oversized(
    *,
    chunk: RawSemanticChunk,
    max_chars: int,
    overlap_lines: int,
) -> list[RawSemanticChunk]:
    if len(chunk.content) <= max_chars:
        return [
            chunk,
        ]

    lines = chunk.content.splitlines(
        keepends=True
    )

    if not lines:
        return [
            chunk,
        ]

    parts: list[RawSemanticChunk] = []
    local_start = 0
    part_number = 1

    while local_start < len(lines):
        size = 0
        local_end = local_start

        while local_end < len(lines):
            candidate = len(
                lines[
                    local_end
                ]
            )

            if (
                local_end > local_start
                and size + candidate > max_chars
            ):
                break

            size += candidate
            local_end += 1

        if local_end <= local_start:
            local_end = local_start + 1

        absolute_start = (
            chunk.start_line
            + local_start
        )

        absolute_end = (
            chunk.start_line
            + local_end
            - 1
        )

        parts.append(
            RawSemanticChunk(
                symbol=(
                    f"{chunk.symbol}#part{part_number}"
                ),
                symbol_kind=(
                    f"{chunk.symbol_kind}_part"
                ),
                parent_symbol=(
                    chunk.symbol
                ),
                semantic_depth=(
                    chunk.semantic_depth
                    + 1
                ),
                start_line=absolute_start,
                end_line=absolute_end,
                content="".join(
                    lines[
                        local_start:
                        local_end
                    ]
                ),
            )
        )

        if local_end >= len(lines):
            break

        next_start = max(
            local_start + 1,
            local_end - max(
                0,
                overlap_lines,
            ),
        )

        local_start = next_start
        part_number += 1

    return parts


def _gap_chunks(
    *,
    lines: list[str],
    occupied: list[tuple[int, int]],
) -> list[RawSemanticChunk]:
    chunks: list[RawSemanticChunk] = []
    cursor = 1
    segment_number = 1

    for start, end in sorted(
        occupied
    ):
        if cursor < start:
            content = _line_slice(
                lines,
                cursor,
                start - 1,
            )

            if content.strip():
                chunks.append(
                    RawSemanticChunk(
                        symbol=(
                            f"<module-segment-{segment_number}>"
                        ),
                        symbol_kind="module_segment",
                        parent_symbol=None,
                        semantic_depth=0,
                        start_line=cursor,
                        end_line=start - 1,
                        content=content,
                    )
                )
                segment_number += 1

        cursor = max(
            cursor,
            end + 1,
        )

    if cursor <= len(lines):
        content = _line_slice(
            lines,
            cursor,
            len(lines),
        )

        if content.strip():
            chunks.append(
                RawSemanticChunk(
                    symbol=(
                        f"<module-segment-{segment_number}>"
                    ),
                    symbol_kind="module_segment",
                    parent_symbol=None,
                    semantic_depth=0,
                    start_line=cursor,
                    end_line=len(lines),
                    content=content,
                )
            )

    return chunks


def _python_chunks(
    text: str,
) -> list[RawSemanticChunk]:
    lines = text.splitlines(
        keepends=True
    )

    try:
        tree = ast.parse(
            text
        )
    except SyntaxError:
        return [
            RawSemanticChunk(
                symbol="<python-module>",
                symbol_kind="module",
                parent_symbol=None,
                semantic_depth=0,
                start_line=1,
                end_line=max(
                    1,
                    len(lines),
                ),
                content=text,
            )
        ]

    chunks: list[RawSemanticChunk] = []
    occupied: list[tuple[int, int]] = []

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue

        start_line = int(
            getattr(
                node,
                "lineno",
                1,
            )
        )

        end_line = int(
            getattr(
                node,
                "end_lineno",
                start_line,
            )
        )

        occupied.append(
            (
                start_line,
                end_line,
            )
        )

        if isinstance(
            node,
            ast.ClassDef,
        ):
            class_content = _line_slice(
                lines,
                start_line,
                end_line,
            )

            # Small classes remain one coherent semantic chapter.
            chunks.append(
                RawSemanticChunk(
                    symbol=node.name,
                    symbol_kind="class",
                    parent_symbol=None,
                    semantic_depth=1,
                    start_line=start_line,
                    end_line=end_line,
                    content=class_content,
                )
            )

            # Methods are emitted too. Downstream deduplication may choose
            # class-level or method-level granularity according to curriculum.
            for child in node.body:
                if not isinstance(
                    child,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):
                    continue

                child_start = int(
                    getattr(
                        child,
                        "lineno",
                        start_line,
                    )
                )

                child_end = int(
                    getattr(
                        child,
                        "end_lineno",
                        child_start,
                    )
                )

                chunks.append(
                    RawSemanticChunk(
                        symbol=(
                            f"{node.name}.{child.name}"
                        ),
                        symbol_kind="method",
                        parent_symbol=node.name,
                        semantic_depth=2,
                        start_line=child_start,
                        end_line=child_end,
                        content=_line_slice(
                            lines,
                            child_start,
                            child_end,
                        ),
                    )
                )

            continue

        kind = (
            "async_function"
            if isinstance(
                node,
                ast.AsyncFunctionDef,
            )
            else "function"
        )

        if node.name.startswith(
            "test_"
        ):
            kind = "test_function"

        chunks.append(
            RawSemanticChunk(
                symbol=node.name,
                symbol_kind=kind,
                parent_symbol=None,
                semantic_depth=1,
                start_line=start_line,
                end_line=end_line,
                content=_line_slice(
                    lines,
                    start_line,
                    end_line,
                ),
            )
        )

    chunks.extend(
        _gap_chunks(
            lines=lines,
            occupied=occupied,
        )
    )

    if not chunks and text.strip():
        chunks.append(
            RawSemanticChunk(
                symbol="<python-module>",
                symbol_kind="module",
                parent_symbol=None,
                semantic_depth=0,
                start_line=1,
                end_line=max(
                    1,
                    len(lines),
                ),
                content=text,
            )
        )

    return sorted(
        chunks,
        key=lambda item: (
            item.start_line,
            item.semantic_depth,
            item.symbol,
        ),
    )


_BOUNDARY_PATTERNS = {
    "elixir": re.compile(
        r"^\s*(defmodule|defprotocol|defimpl|defmacro|defmacrop|defp|def)\s+([A-Za-z0-9_!?\.]+)",
        re.MULTILINE,
    ),
    "nim": re.compile(
        r"^\s*(proc|func|method|iterator|template|macro|type)\s+([A-Za-z0-9_`*]+)",
        re.MULTILINE,
    ),
    "javascript": re.compile(
        r"^\s*(?:export\s+)?(?:(async)\s+)?(function|class)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r"^\s*(?:export\s+)?(?:(async)\s+)?(function|class|interface|type)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
        re.MULTILINE,
    ),
}


def _line_number(
    text: str,
    offset: int,
) -> int:
    return (
        text.count(
            "\n",
            0,
            offset,
        )
        + 1
    )


def _regex_semantic_chunks(
    *,
    text: str,
    language: str,
) -> list[RawSemanticChunk]:
    pattern = _BOUNDARY_PATTERNS[
        language
    ]

    matches = list(
        pattern.finditer(
            text
        )
    )

    if not matches:
        lines = text.splitlines(
            keepends=True
        )

        return [
            RawSemanticChunk(
                symbol=f"<{language}-module>",
                symbol_kind="module",
                parent_symbol=None,
                semantic_depth=0,
                start_line=1,
                end_line=max(
                    1,
                    len(lines),
                ),
                content=text,
            )
        ]

    chunks: list[RawSemanticChunk] = []

    if matches[0].start() > 0:
        prefix = text[
            :matches[0].start()
        ]

        if prefix.strip():
            chunks.append(
                RawSemanticChunk(
                    symbol="<module-preamble>",
                    symbol_kind="module_segment",
                    parent_symbol=None,
                    semantic_depth=0,
                    start_line=1,
                    end_line=max(
                        1,
                        _line_number(
                            text,
                            matches[0].start(),
                        )
                        - 1,
                    ),
                    content=prefix,
                )
            )

    for index, match in enumerate(
        matches
    ):
        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        raw = match.groups()

        if language in {
            "javascript",
            "typescript",
        }:
            kind = raw[-2]
            symbol = raw[-1]
        else:
            kind = raw[0]
            symbol = raw[1]

        start_line = _line_number(
            text,
            start,
        )

        end_line = max(
            start_line,
            _line_number(
                text,
                max(
                    start,
                    end - 1,
                ),
            ),
        )

        chunks.append(
            RawSemanticChunk(
                symbol=symbol,
                symbol_kind=kind,
                parent_symbol=None,
                semantic_depth=1,
                start_line=start_line,
                end_line=end_line,
                content=text[
                    start:end
                ],
            )
        )

    return chunks


def _markdown_chunks(
    text: str,
) -> list[RawSemanticChunk]:
    lines = text.splitlines(
        keepends=True
    )

    heading_lines: list[
        tuple[
            int,
            str,
            int,
        ]
    ] = []

    for index, line in enumerate(
        lines,
        start=1,
    ):
        match = re.match(
            r"^(#{1,6})\s+(.+?)\s*$",
            line,
        )

        if match:
            heading_lines.append(
                (
                    index,
                    match.group(2),
                    len(
                        match.group(1)
                    ),
                )
            )

    if not heading_lines:
        return [
            RawSemanticChunk(
                symbol="<markdown-document>",
                symbol_kind="document",
                parent_symbol=None,
                semantic_depth=0,
                start_line=1,
                end_line=max(
                    1,
                    len(lines),
                ),
                content=text,
            )
        ]

    chunks: list[RawSemanticChunk] = []

    if heading_lines[0][0] > 1:
        chunks.append(
            RawSemanticChunk(
                symbol="<document-preamble>",
                symbol_kind="document_segment",
                parent_symbol=None,
                semantic_depth=0,
                start_line=1,
                end_line=(
                    heading_lines[0][0]
                    - 1
                ),
                content=_line_slice(
                    lines,
                    1,
                    heading_lines[0][0] - 1,
                ),
            )
        )

    for index, (
        start_line,
        title,
        depth,
    ) in enumerate(
        heading_lines
    ):
        end_line = (
            heading_lines[
                index + 1
            ][0]
            - 1
            if index + 1 < len(
                heading_lines
            )
            else len(lines)
        )

        chunks.append(
            RawSemanticChunk(
                symbol=title,
                symbol_kind="section",
                parent_symbol=None,
                semantic_depth=depth,
                start_line=start_line,
                end_line=end_line,
                content=_line_slice(
                    lines,
                    start_line,
                    end_line,
                ),
            )
        )

    return chunks


def split_semantic_source(
    *,
    text: str,
    language: str,
    max_chars: int = 8000,
    overlap_lines: int = 8,
) -> list[RawSemanticChunk]:
    """
    Split source semantically first, then use bounded line windows only as a
    last resort for oversized semantic units.
    """

    if max_chars < 256:
        raise ValueError(
            "max_chars must be at least 256"
        )

    if overlap_lines < 0:
        raise ValueError(
            "overlap_lines must be non-negative"
        )

    if not text.strip():
        return []

    if language == "python":
        semantic = _python_chunks(
            text
        )

    elif language in _BOUNDARY_PATTERNS:
        semantic = _regex_semantic_chunks(
            text=text,
            language=language,
        )

    elif language == "markdown":
        semantic = _markdown_chunks(
            text
        )

    else:
        lines = text.splitlines(
            keepends=True
        )

        semantic = [
            RawSemanticChunk(
                symbol=f"<{language}-document>",
                symbol_kind="document",
                parent_symbol=None,
                semantic_depth=0,
                start_line=1,
                end_line=max(
                    1,
                    len(lines),
                ),
                content=text,
            )
        ]

    result: list[RawSemanticChunk] = []

    for chunk in semantic:
        result.extend(
            _split_oversized(
                chunk=chunk,
                max_chars=max_chars,
                overlap_lines=overlap_lines,
            )
        )

    return result
