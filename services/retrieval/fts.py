"""FTS ranking over an already-authorized set of records.

This module performs no source discovery, admission or access decisions. Callers
must apply those checks before supplying records or a table.

``search_fts`` ranks capability records for routing. ``search_passages`` finds
passages that answer a question in a private in-memory index built for that
call; ``match_passages`` does the same in a caller's own persistent passage
table. Both match word forms (porter stemming), ignore grammatical words and
return nothing rather than unrelated text.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
import math
import re
import sqlite3

# Grammatical words only (articles, pronouns, auxiliaries, prepositions and
# conjunctions): they say how a question is asked, never what it is about.
PASSAGE_STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most my
myself no nor not now of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your yours
yourself yourselves may might must shall cannot don doesn didn isn aren wasn weren won wouldn shouldn
couldn haven hasn hadn ll re ve much many
""".split())

# A word written with hyphens or slashes ("Wi-Fi", "sign-in") is also indexed
# joined ("wifi", "signin"), so either spelling finds it.
_COMPOUND = re.compile(r"[^\W_]+(?:[-/][^\W_]+)+")
_WORD = re.compile(r"[^\W_]+(?:[-/][^\W_]+)*")


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


def query_terms(query: str, *, max_terms: int | None = None) -> list[str]:
    """The words of a question that say what it is about, in order, once each."""
    terms: list[str] = []
    for match in _WORD.finditer(query.casefold()):
        word = re.sub(r"[-/]", "", match.group(0))
        if len(word) < 2 or word in PASSAGE_STOP_WORDS or word in terms:
            continue
        terms.append(word)
    return terms[:max_terms] if max_terms is not None else terms


def _joined_compounds(*texts: str) -> str:
    return " ".join(re.sub(r"[-/]", "", match.group(0)) for text in texts for match in _COMPOUND.finditer(text))


PASSAGE_COLUMNS = "title, summary, content, tags, joined"


def passage_row(record: FtsRecord) -> tuple[str, str, str, str, str]:
    """A record's indexed columns, in PASSAGE_COLUMNS order."""
    tags = " ".join(record.tags)
    return (record.title, record.summary, record.content, tags,
            _joined_compounds(record.title, record.summary, record.content, tags))


def passage_table_sql(table: str, *extra_columns: str) -> str:
    """The FTS5 table every passage search reads: stemmed words, titles and joined compounds."""
    columns = ", ".join((*extra_columns, PASSAGE_COLUMNS))
    return f"CREATE VIRTUAL TABLE IF NOT EXISTS \"{table}\" USING fts5({columns}, tokenize='porter unicode61')"


def search_passages(
    records: Iterable[FtsRecord],
    query: str,
    *,
    limit: int,
    kind: str | None = None,
    max_query_tokens: int | None = 32,
) -> FtsMatches:
    """Passages among ``records`` that answer ``query``, best first, or none.

    Builds a private in-memory index for this call; see ``match_passages``.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("Search limit must be a positive integer")
    with sqlite3.connect(":memory:") as db:
        db.execute(passage_table_sql("docs", "id UNINDEXED", "kind UNINDEXED"))
        seen = set()
        for record in records:
            if not record.identifier or record.identifier in seen:
                raise ValueError("Search records require unique nonempty identifiers")
            seen.add(record.identifier)
            db.execute("INSERT INTO docs VALUES(?,?,?,?,?,?,?)",
                       (record.identifier, record.kind, *passage_row(record)))
        scope, parameters = ("kind = ?", (kind,)) if kind is not None else ("1", ())
        identifiers = match_passages(db, "docs", "id", query, limit=limit, scope=scope,
                                     parameters=parameters, max_query_tokens=max_query_tokens)
    return identifiers


def match_passages(
    db: sqlite3.Connection,
    table: str,
    key: str,
    query: str,
    *,
    limit: int,
    scope: str = "1",
    parameters: tuple = (),
    max_query_tokens: int | None = 32,
) -> FtsMatches:
    """Rows of a passage table that answer ``query``, best first, or none.

    ``table`` is a trusted table built with ``passage_table_sql``; ``key`` is the
    column naming each row, and ``scope`` a trusted SQL condition (with
    ``parameters``) limiting which rows may match.

    Words match their other forms ("freezes" finds "freeze"), and a hyphenated
    word matches its joined spelling. A passage qualifies only when its own text
    contains enough of the question's words (every word of a one- or two-word
    question, otherwise at least two and a third of them) and no other passage
    contains more of them, so one shared word never makes an unrelated passage
    evidence. A page title counts towards a passage's words but cannot qualify
    it alone. Ranking among qualified passages: BM25, titles weighted highest.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("Search limit must be a positive integer")
    terms = query_terms(query, max_terms=max_query_tokens)
    if not terms:
        return FtsMatches()
    needed = len(terms) if len(terms) <= 2 else max(2, math.ceil(len(terms) / 3))
    select = f'SELECT {key} FROM "{table}" WHERE "{table}" MATCH ? AND ({scope})'
    covered: Counter = Counter()  # How many of the question's words each passage contains.
    for term in terms:
        covered.update(identifier for (identifier,) in db.execute(select, ('"' + term + '"', *parameters)))
    if not covered:
        return FtsMatches()
    floor = max(needed, max(covered.values()))
    candidates = {identifier for identifier, words in covered.items() if words >= floor}
    any_term = " OR ".join('"' + term + '"' for term in terms)
    # A title names a whole page; only words in the passage's own text show
    # that this passage, not just its page, is about the question.
    qualified = {identifier for (identifier,) in db.execute(
        select, ("{summary content tags joined} : (" + any_term + ")", *parameters)) if identifier in candidates}
    if not qualified:
        return FtsMatches()
    # Titles and summaries describe a passage more strongly than its body. The
    # weights follow PASSAGE_COLUMNS; columns before them carry no text.
    weights = ", ".join(["0"] * _leading_columns(db, table) + ["3.0", "2.0", "1.0", "2.0", "1.0"])
    scores = {identifier: score for identifier, score in db.execute(
        f'SELECT {key}, bm25("{table}", {weights}) FROM "{table}" WHERE "{table}" MATCH ? AND ({scope})',
        (any_term, *parameters)) if identifier in qualified}
    ranked = sorted(qualified, key=lambda identifier: (scores[identifier], identifier))
    return FtsMatches(tuple(ranked[:limit]), len(ranked) > limit)


def _leading_columns(db: sqlite3.Connection, table: str) -> int:
    """How many columns precede the passage columns in ``table``."""
    names = [row[1] for row in db.execute(f'PRAGMA table_info("{table}")')]
    return names.index("title")
