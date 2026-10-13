"""Stateless FTS ranking over an already-authorized set of records.

This module performs no source discovery, admission or access decisions. Callers
must apply those checks before supplying records. Each call owns an independent
in-memory index; private records cannot persist into another caller's search.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import re
import sqlite3


@dataclass(frozen=True)
class FtsRecord:
    identifier: str
    content: str
    kind: str = field(default="", kw_only=True)
    title: str = field(default="", kw_only=True)
    summary: str = field(default="", kw_only=True)
    tags: tuple[str, ...] = field(default=(), kw_only=True)


@dataclass(frozen=True)
class FtsMatches:
    identifiers: tuple[str, ...] = ()
    truncated: bool = False


def search_fts(
    records: Iterable[FtsRecord],
    query: str,
    *,
    limit: int,
    kind: str | None = None,
    max_query_tokens: int | None = None,
) -> FtsMatches:
    """Rank supplied records, returning only identifiers and truncation status."""
    if type(limit) is not int or limit < 1:
        raise ValueError("Search limit must be a positive integer")
    if max_query_tokens is not None and (
        type(max_query_tokens) is not int or max_query_tokens < 1
    ):
        raise ValueError("Query token limit must be a positive integer")
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    if max_query_tokens is not None:
        tokens = tokens[:max_query_tokens]
    if not tokens:
        return FtsMatches()
    expression = " OR ".join('"' + token + '"' for token in tokens)

    with sqlite3.connect(":memory:") as db:
        db.execute(
            "CREATE VIRTUAL TABLE docs USING "
            "fts5(id UNINDEXED, kind UNINDEXED, title, summary, content, tags)"
        )
        seen = set()
        for record in records:
            if not record.identifier or record.identifier in seen:
                raise ValueError("Search records require unique nonempty identifiers")
            seen.add(record.identifier)
            db.execute(
                "INSERT INTO docs VALUES(?,?,?,?,?,?)",
                (record.identifier, record.kind, record.title, record.summary,
                 record.content, " ".join(record.tags)),
            )
        rows = db.execute(
            "SELECT id FROM docs WHERE docs MATCH ? "
            "AND (? IS NULL OR kind = ?) ORDER BY bm25(docs), id LIMIT ?",
            (expression, kind, kind, limit + 1),
        ).fetchall()
    return FtsMatches(tuple(row[0] for row in rows[:limit]), len(rows) > limit)
