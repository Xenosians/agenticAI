from learning.continual.corpus_materializer import (
    _filter_reason,
    normalize_source_row,
)
from learning.continual.corpus_registry import (
    CorpusRegistrySource,
)


def _source():
    return CorpusRegistrySource(
        source_id="swe-rebench-v2",
        provider="huggingface",
        dataset_id=(
            "PrimeIntellect/"
            "SWE-rebench-V2-Filtered-Verified"
        ),
        revision=(
            "03cc767ee33126b7fc7890ad57047e9dd6914cca"
        ),
        target_component=(
            "developer-specialist"
        ),
        objectives=[
            "sft"
        ],
        trust="curated",
        enabled=True,
        training_eligible=True,
        materialization_mode="stream",
        filters={
            "languages": [
                "Python",
                "JavaScript",
                "TypeScript",
                "Elixir",
            ],
            "include_tasks": [],
            "exclude_tasks": [],
            "include_paths": [],
            "exclude_paths": [],
            "require_verified_outcome": True,
            "require_permissive_source_license": True,
            "deduplicate": True,
            "secret_scan": True,
            "benchmark_decontamination": False,
        },
        metadata={
            "adapter":
                "issue_patch",

            "upstream_verified":
                True,

            "require_behavioral_parser_support":
                True,

            "developer_prompt_contract":
                "developer-issue-context.v2",
        },
    )


def _row(
    *,
    language="ts",
    log_parser="parse_log_pytest",
):
    return {
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
            language,

        "license":
            "MIT",

        "install_config": {
            "log_parser":
                log_parser,
        },
    }


def test_alias_language_is_admitted_and_repository_context_is_in_prompt():
    source = _source()

    record = normalize_source_row(
        source,
        index=0,
        row=_row(),
    )

    assert record is not None
    assert record.language == "typescript"
    assert record.verified_outcome is True

    rendered = "\n".join(
        message["content"]
        for message
        in record.prompt_messages
    )

    assert "Repository: owner/repo" in rendered
    assert "Base commit: abc123" in rendered

    reason = _filter_reason(
        source,
        record,
        seen_hashes=set(),
    )

    assert reason is None


def test_unsupported_behavioral_parser_is_rejected_with_explicit_reason():
    source = _source()

    record = normalize_source_row(
        source,
        index=0,
        row=_row(
            log_parser=(
                "parse_log_js_4"
            )
        ),
    )

    assert record is not None

    reason = _filter_reason(
        source,
        record,
        seen_hashes=set(),
    )

    assert (
        reason
        == "behavioral_log_parser_unsupported"
    )
