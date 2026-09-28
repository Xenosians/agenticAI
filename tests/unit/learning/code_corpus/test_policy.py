from pathlib import Path

from learning.code_corpus.policy import (
    detect_language,
    should_include_file,
)


def test_project_readme_is_not_ingested_as_specialist_code():
    assert (
        should_include_file(
            relative_path=Path(
                "README.md"
            ),
            source_kind="project",
        )
        is False
    )


def test_project_specialist_prompt_markdown_is_ingested():
    assert (
        should_include_file(
            relative_path=Path(
                "subagents/prompts/hub_router.txt"
            ),
            source_kind="project",
        )
        is False
    )

    assert (
        should_include_file(
            relative_path=Path(
                "subagents/agents/jira-specialist.md"
            ),
            source_kind="project",
        )
        is True
    )


def test_language_detection_is_extension_based_and_bounded():
    assert detect_language(
        Path(
            "router.py"
        )
    ) == "python"

    assert detect_language(
        Path(
            "application.ex"
        )
    ) == "elixir"

    assert detect_language(
        Path(
            "frontend.nim"
        )
    ) == "nim"

    assert detect_language(
        Path(
            "secret.env"
        )
    ) is None
