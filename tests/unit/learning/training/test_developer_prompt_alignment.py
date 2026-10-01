from learning.continual.corpus_materializer import (
    CorpusSnapshotManifest,
    NormalizedCorpusRecord,
)
from learning.training.developer_corpus_bridge import (
    _phase5_record,
)


def test_developer_bridge_rebuilds_issue_prompt_from_canonical_metadata():
    record = NormalizedCorpusRecord(
        record_id="corpus-record-1",
        source_id="swe-rebench-v2",
        source_record_id="task-1",
        upstream_index=0,
        objective="sft",
        target_component="developer-specialist",
        prompt_messages=[
            {
                "role":
                    "user",

                "content":
                    "legacy prompt",
            }
        ],
        chosen=(
            "diff --git a/a.py b/a.py"
        ),
        language="python",
        verified_outcome=True,
        content_sha256=(
            "1" * 64
        ),
        metadata={
            "adapter":
                "issue_patch",

            "problem_statement":
                "Fix the parser.",

            "repository":
                "owner/repo",

            "base_commit":
                "deadbeef",

            "repository_license":
                "MIT",

            "behavioral_log_parser":
                "parse_log_pytest",
        },
    )

    snapshot = CorpusSnapshotManifest(
        snapshot_id="snapshot-1",
        created_at="2026-01-01T00:00:00+00:00",
        source_id="swe-rebench-v2",
        provider="huggingface",
        dataset_id=(
            "PrimeIntellect/"
            "SWE-rebench-V2-Filtered-Verified"
        ),
        revision="a" * 40,
        split="train",
        target_component="developer-specialist",
        objectives=[
            "sft"
        ],
        source_config_sha256="b" * 64,
        decontamination_policy_sha256="c" * 64,
        cursor_start=0,
        cursor_end=1,
        scanned_count=1,
        accepted_count=1,
        records_sha256="d" * 64,
        training_eligible=True,
        training_blockers=[],
        output_directory="/tmp/snapshot",
    )

    phase5 = _phase5_record(
        snapshot=snapshot,
        record=record,
        partition="train",
    )

    rendered = "\n".join(
        message["content"]
        for message
        in phase5.prompt_messages
    )

    assert "legacy prompt" not in rendered
    assert "Fix the parser." in rendered
    assert "Repository: owner/repo" in rendered
    assert "Base commit: deadbeef" in rendered
