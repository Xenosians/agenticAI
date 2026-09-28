import json

from pathlib import Path

from learning.continual.automation.corpus import (
    next_progressive_pages,
    study_split,
)
from learning.continual.automation.store import (
    ContinualAutomationStore,
)
from learning.continual.automation.types import (
    CorpusSource,
)


def test_study_split_is_progressive_continuation():
    prefix, chosen = study_split(
        "abcdefghijklmnopqrstuvwxyz"
        * 10
    )

    assert prefix
    assert chosen
    assert prefix + chosen == (
        "abcdefghijklmnopqrstuvwxyz"
        * 10
    )


def test_text_corpus_advances_one_page_at_a_time(
    tmp_path: Path,
):
    corpus = tmp_path / "book.txt"

    corpus.write_text(
        "A" * 5000,
        encoding="utf-8",
    )

    store = ContinualAutomationStore(
        tmp_path
        / "state.sqlite3"
    )

    store.initialize()

    source = CorpusSource(
        source_id="book",
        kind="text",
        path=str(
            corpus
        ),
        target_role="shared",
        trusted=True,
        training_eligible=True,
        page_chars=1000,
    )

    import hashlib

    sha = hashlib.sha256(
        corpus.read_bytes()
    ).hexdigest()

    store.upsert_corpus_source(
        source,
        source_sha256=sha,
        total_units=5,
    )

    first = next_progressive_pages(
        store=store,
        limit=2,
    )

    second = next_progressive_pages(
        store=store,
        limit=2,
    )

    assert [
        page.page_index
        for page in first
    ] == [
        0,
        1,
    ]

    assert [
        page.page_index
        for page in second
    ] == [
        2,
        3,
    ]
