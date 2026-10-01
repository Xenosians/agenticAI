from learning.developer_contract import (
    SUPPORTED_BEHAVIORAL_LOG_PARSERS,
    behavioral_parser_supported,
    build_developer_issue_messages,
    canonical_language,
    parse_swe_rebench_raw_row,
)


def test_language_aliases_are_canonical():
    assert canonical_language("js") == "javascript"
    assert canonical_language("JavaScript") == "javascript"
    assert canonical_language("ts") == "typescript"
    assert canonical_language("TypeScript") == "typescript"
    assert canonical_language("PY") == "python"
    assert canonical_language("Elixir") == "elixir"


def test_behavioral_parser_capability_is_explicit():
    assert behavioral_parser_supported(
        "parse_log_pytest"
    )
    assert behavioral_parser_supported(
        "parse_log_elixir"
    )
    assert not behavioral_parser_supported(
        "parse_log_js_4"
    )
    assert (
        SUPPORTED_BEHAVIORAL_LOG_PARSERS
        == {
            "parse_log_pytest",
            "parse_log_elixir",
        }
    )


def test_raw_boundary_allows_extra_fields_but_requires_core_identity():
    row = parse_swe_rebench_raw_row(
        {
            "instance_id":
                "task-1",

            "repo":
                "owner/repo",

            "base_commit":
                "abc123",

            "patch":
                "diff --git a/a.py b/a.py",

            "problem_statement":
                "Fix the bug.",

            "language":
                "py",

            "license":
                "MIT",

            "install_config": {
                "log_parser":
                    "parse_log_pytest",
            },

            "test_patch":
                "hidden evaluator data",

            "PASS_TO_PASS": [
                "test_ok"
            ],
        }
    )

    assert (
        row.canonical_language
        == "python"
    )

    assert (
        row.log_parser
        == "parse_log_pytest"
    )


def test_shared_prompt_contract_adds_repository_context_without_hidden_tests():
    messages = build_developer_issue_messages(
        problem_statement=(
            "Fix a parsing failure."
        ),
        repository=(
            "owner/repo"
        ),
        base_commit=(
            "deadbeef"
        ),
        repository_tree=[
            "src/parser.py"
        ],
        selected_files=[
            (
                "src/parser.py",
                "def parse():\n    pass",
            )
        ],
    )

    rendered = "\n".join(
        message["content"]
        for message in messages
    )

    assert "Repository: owner/repo" in rendered
    assert "Base commit: deadbeef" in rendered
    assert "src/parser.py" in rendered
    assert "test_patch" not in rendered
    assert "FAIL_TO_PASS" not in rendered
