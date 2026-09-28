import json

from pathlib import Path

from learning.continual.corpus_materializer import (
    CorpusCursorStore,
    materialize_corpus_source,
    normalize_source_row,
)

from learning.continual.corpus_registry import (
    CorpusRegistrySource,
)


def _write_registry(
    path: Path,
    corpus_path: Path,
) -> None:

    path.write_text(
        json.dumps(
            {
                "schema":
                    "continual-corpus-registry.v1",

                "mix": {
                    "max_external_fraction":
                        0.5,

                    "target_runtime_fraction":
                        0.3,

                    "target_replay_fraction":
                        0.2,
                },

                "sources": [
                    {
                        "source_id":
                            "local-instruction-test",

                        "provider":
                            "local",

                        "dataset_id":
                            None,

                        "subset":
                            None,

                        "revision":
                            None,

                        "local_path":
                            str(
                                corpus_path
                            ),

                        "license":
                            "test",

                        "upstream_url":
                            None,

                        "target_component":
                            "hub",

                        "audience": [
                            "hub"
                        ],

                        "objectives": [
                            "sft"
                        ],

                        "trust":
                            "curated",

                        "enabled":
                            True,

                        "training_eligible":
                            True,

                        "evaluation_only":
                            False,

                        "verified_reward":
                            False,

                        "materialization_mode":
                            "local",

                        "contamination_group":
                            "unit-test",

                        "sampling": {
                            "priority":
                                50,

                            "max_records_per_cycle":
                                2,

                            "max_records_per_snapshot":
                                2,

                            "replay_weight":
                                0.2,

                            "max_cycle_fraction":
                                0.25,
                        },

                        "filters": {
                            "languages":
                                [],

                            "include_tasks":
                                [],

                            "exclude_tasks":
                                [],

                            "include_paths":
                                [],

                            "exclude_paths":
                                [],

                            "require_verified_outcome":
                                False,

                            "require_permissive_source_license":
                                False,

                            "deduplicate":
                                True,

                            "secret_scan":
                                True,

                            "benchmark_decontamination":
                                False,
                        },

                        "metadata": {
                            "adapter":
                                "instruction"
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_instruction_normalization():

    source = (
        CorpusRegistrySource(
            source_id="unit-instruction",
            provider="local",
            local_path="/tmp/example.jsonl",
            target_component="hub",
            objectives=[
                "sft"
            ],
            trust="curated",
            enabled=True,
            training_eligible=True,
            metadata={
                "adapter":
                    "instruction"
            },
        )
    )

    record = normalize_source_row(
        source,
        index=0,
        row={
            "instruction":
                "Create a ticket.",

            "response":
                '{"action":"create"}',
        },
    )

    assert record is not None
    assert record.objective == "sft"

    assert (
        record.chosen
        == '{"action":"create"}'
    )

    assert (
        record.prompt_messages[
            -1
        ][
            "content"
        ]
        == "Create a ticket."
    )


def test_materialization_is_bounded_and_cursor_resumes(
    tmp_path: Path,
):

    corpus = (
        tmp_path
        / "corpus.jsonl"
    )

    rows = [
        {
            "instruction":
                f"request-{index}",

            "response":
                f"response-{index}",
        }

        for index
        in range(
            5
        )
    ]

    corpus.write_text(
        "".join(
            json.dumps(
                row
            )
            + "\n"

            for row
            in rows
        ),
        encoding="utf-8",
    )

    registry = (
        tmp_path
        / "registry.json"
    )

    _write_registry(
        registry,
        corpus,
    )

    cursor_store = (
        CorpusCursorStore(
            tmp_path
            / "cursors.json"
        )
    )

    snapshots = (
        tmp_path
        / "snapshots"
    )

    first = (
        materialize_corpus_source(
            registry_path=registry,
            source_id=(
                "local-instruction-test"
            ),
            cursor_store=cursor_store,
            snapshot_root=snapshots,
        )
    )

    assert (
        first.manifest.cursor_start
        == 0
    )

    assert (
        first.manifest.cursor_end
        == 2
    )

    assert (
        first.manifest.accepted_count
        == 2
    )

    second = (
        materialize_corpus_source(
            registry_path=registry,
            source_id=(
                "local-instruction-test"
            ),
            cursor_store=cursor_store,
            snapshot_root=snapshots,
        )
    )

    assert (
        second.manifest.cursor_start
        == 2
    )

    assert (
        second.manifest.cursor_end
        == 4
    )

    assert (
        second.manifest.accepted_count
        == 2
    )


def test_materialized_external_data_is_not_optimizer_eligible_yet(
    tmp_path: Path,
):

    corpus = (
        tmp_path
        / "corpus.jsonl"
    )

    corpus.write_text(
        json.dumps(
            {
                "instruction":
                    "hello",

                "response":
                    "world",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    registry = (
        tmp_path
        / "registry.json"
    )

    _write_registry(
        registry,
        corpus,
    )

    result = (
        materialize_corpus_source(
            registry_path=registry,
            source_id=(
                "local-instruction-test"
            ),
            cursor_store=(
                CorpusCursorStore(
                    tmp_path
                    / "cursor.json"
                )
            ),
            snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert (
        result.manifest.training_eligible
        is False
    )

    assert (
        "objective_specific_optimizer_bridge_not_wired"
        in result.manifest.training_blockers
    )


def test_secret_like_record_is_filtered(
    tmp_path: Path,
):

    corpus = (
        tmp_path
        / "corpus.jsonl"
    )

    corpus.write_text(
        json.dumps(
            {
                "instruction":
                    "Use this credential",

                "response":
                    (
                        "-----BEGIN PRIVATE KEY-----"
                    ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    registry = (
        tmp_path
        / "registry.json"
    )

    _write_registry(
        registry,
        corpus,
    )

    result = (
        materialize_corpus_source(
            registry_path=registry,
            source_id=(
                "local-instruction-test"
            ),
            cursor_store=(
                CorpusCursorStore(
                    tmp_path
                    / "cursor.json"
                )
            ),
            snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert (
        result.manifest.accepted_count
        == 0
    )

    assert (
        result.manifest.rejected_counts[
            "secret_like_content"
        ]
        == 1
    )


def test_huggingface_reader_uses_isolated_worker(
    monkeypatch,
):
    import learning.continual.corpus_materializer as module

    source = CorpusRegistrySource(
        source_id="hf-worker-test",
        provider="huggingface",
        dataset_id="example/test-dataset",
        target_component="developer-specialist",
        objectives=[
            "sft"
        ],
        trust="curated",
        enabled=True,
        training_eligible=True,
        materialization_mode="stream",
    )

    class FakeStdout:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield json.dumps(
                {
                    "index": 25,
                    "row": {
                        "instruction":
                            "Fix the issue.",
                        "response":
                            "Apply the patch.",
                    },
                }
            ) + "\n"

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(
            self,
            command,
            **kwargs,
        ):
            self.command = command
            self.kwargs = kwargs
            self.stdout = FakeStdout()
            self.returncode = None

        def wait(
            self,
            timeout=None,
        ):
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    captured = {}

    def fake_popen(
        command,
        **kwargs,
    ):
        process = FakeProcess(
            command,
            **kwargs,
        )

        captured[
            "process"
        ] = process

        return process

    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        fake_popen,
    )

    rows = list(
        module._iter_huggingface(
            source,
            start_index=25,
            max_rows=100,
        )
    )

    assert rows == [
        (
            25,
            {
                "instruction":
                    "Fix the issue.",
                "response":
                    "Apply the patch.",
            },
        )
    ]

    command = (
        captured[
            "process"
        ]
        .command
    )

    assert (
        "learning.continual.hf_stream_worker"
        in command
    )

    assert (
        command[
            command.index(
                "--start-index"
            )
            + 1
        ]
        == "25"
    )

    assert (
        command[
            command.index(
                "--max-rows"
            )
            + 1
        ]
        == "100"
    )

    assert (
        captured[
            "process"
        ]
        .returncode
        == 0
    )


def test_permissive_repository_license_is_enforced():

    source = CorpusRegistrySource(
        source_id="license-test",
        provider="local",
        local_path="/tmp/test.jsonl",
        target_component="developer-specialist",
        objectives=[
            "sft"
        ],
        trust="curated",
        enabled=True,
        training_eligible=True,
        filters={
            "languages": [],
            "include_tasks": [],
            "exclude_tasks": [],
            "include_paths": [],
            "exclude_paths": [],
            "require_verified_outcome": False,
            "require_permissive_source_license": True,
            "deduplicate": True,
            "secret_scan": True,
            "benchmark_decontamination": False,
        },
        metadata={
            "adapter":
                "issue_patch"
        },
    )

    record = normalize_source_row(
        source,
        index=0,
        row={
            "instance_id":
                "test-1",

            "repo":
                "owner/repo",

            "base_commit":
                "abc123",

            "license":
                "GPL-3.0",

            "problem_statement":
                "Fix the bug.",

            "patch":
                "diff --git a/a.py b/a.py",
        },
    )

    assert record is not None

    import learning.continual.corpus_materializer as module

    reason = module._filter_reason(
        source,
        record,
        seen_hashes=set(),
    )

    assert (
        reason
        == "repository_license_not_permitted"
    )


def test_repository_identity_is_preserved():

    source = CorpusRegistrySource(
        source_id="identity-test",
        provider="local",
        local_path="/tmp/test.jsonl",
        target_component="developer-specialist",
        objectives=[
            "sft"
        ],
        trust="curated",
        enabled=True,
        training_eligible=True,
        metadata={
            "adapter":
                "issue_patch"
        },
    )

    record = normalize_source_row(
        source,
        index=0,
        row={
            "instance_id":
                "repo-task-1",

            "repo":
                "owner/repo",

            "base_commit":
                "abcdef",

            "license":
                "MIT",

            "problem_statement":
                "Fix the bug.",

            "patch":
                "diff --git a/a.py b/a.py",
        },
    )

    assert record is not None

    assert (
        record.metadata[
            "repository"
        ]
        == "owner/repo"
    )

    assert (
        record.metadata[
            "base_commit"
        ]
        == "abcdef"
    )

    assert (
        record.metadata[
            "repository_license"
        ]
        == "MIT"
    )
