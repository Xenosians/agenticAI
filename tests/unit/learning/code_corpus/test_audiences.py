from learning.code_corpus.audiences import (
    classify_chunk_audiences,
    curriculum_hints_for_chunk,
)


def test_general_code_goes_to_hub_not_project_specialists():
    audiences = classify_chunk_audiences(
        source_kind="general",
        relative_path="python/examples/parser.py",
    )

    assert audiences == [
        "hub",
    ]


def test_project_code_never_goes_to_hub_by_default():
    audiences = classify_chunk_audiences(
        source_kind="project",
        relative_path="subagents/core/orchestration/router.py",
    )

    assert "developer-specialist" in audiences
    assert "hub" not in audiences


def test_jira_specific_code_goes_to_developer_and_jira():
    audiences = classify_chunk_audiences(
        source_kind="project",
        relative_path="tools/jira/catalog.py",
    )

    assert "developer-specialist" in audiences
    assert "jira-specialist" in audiences
    assert "hub" not in audiences


def test_provider_neutral_ticketing_is_shared_with_jira_specialist():
    audiences = classify_chunk_audiences(
        source_kind="project",
        relative_path="tools/ticketing/catalog.py",
    )

    assert "developer-specialist" in audiences
    assert "ticket-specialist" in audiences
    assert "jira-specialist" in audiences


def test_developer_code_chunks_hint_code_symbol_chapter():
    hints = curriculum_hints_for_chunk(
        audiences=[
            "developer-specialist",
        ],
        symbol_kind="function",
    )

    assert "dev-04-code-symbols" in hints
