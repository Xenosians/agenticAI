from learning.continual.corpus_materializer import (
    NormalizedCorpusRecord,
)

from learning.evaluation.developer_holdout import (
    DeveloperHoldoutRecord,
)

from learning.evaluation.developer_holdout_eval import (
    _overlap_identities,
    _relative_improvement_percent,
)


def _training_record(
    instance_id: str,
) -> NormalizedCorpusRecord:

    return (
        NormalizedCorpusRecord(
            record_id=(
                "train-"
                + instance_id
            ),

            source_id=(
                "swe-rebench-v2"
            ),

            source_record_id=(
                instance_id
            ),

            upstream_index=1,

            objective="sft",

            target_component=(
                "developer-specialist"
            ),

            prompt_messages=[
                {
                    "role":
                        "user",

                    "content":
                        "Fix it.",
                }
            ],

            chosen="patch",

            verified_outcome=True,

            content_sha256=(
                "a" * 64
            ),

            metadata={
                "repository":
                    "owner/repo",

                "base_commit":
                    "abc",

                "repository_license":
                    "MIT",
            },
        )
    )


def _holdout_record(
    instance_id: str,
) -> DeveloperHoldoutRecord:

    return (
        DeveloperHoldoutRecord(
            record_id=(
                "holdout-"
                + instance_id
            ),

            source_id=(
                "swe-rebench-v2"
            ),

            source_record_id=(
                instance_id
            ),

            source_identity=(
                "instance:"
                + instance_id
            ),

            upstream_index=2,

            prompt_messages=[
                {
                    "role":
                        "user",

                    "content":
                        "Fix it.",
                }
            ],

            reference_completion=(
                "patch"
            ),

            content_sha256=(
                "b" * 64
            ),

            verified_outcome=True,

            repository=(
                "owner/repo"
            ),

            base_commit=(
                "def"
            ),

            repository_license=(
                "MIT"
            ),

            adapter=(
                "issue_patch"
            ),
        )
    )


def test_overlap_detection_is_identity_based():

    overlaps = (
        _overlap_identities(
            training_records=[
                _training_record(
                    "same-instance"
                )
            ],

            holdout_records=[
                _holdout_record(
                    "same-instance"
                )
            ],
        )
    )

    assert overlaps == [
        "instance:same-instance"
    ]


def test_distinct_holdout_has_no_training_overlap():

    overlaps = (
        _overlap_identities(
            training_records=[
                _training_record(
                    "training-instance"
                )
            ],

            holdout_records=[
                _holdout_record(
                    "heldout-instance"
                )
            ],
        )
    )

    assert overlaps == []


def test_relative_loss_improvement_is_positive_when_loss_falls():

    value = (
        _relative_improvement_percent(
            base_loss=2.0,
            candidate_loss=1.5,
        )
    )

    assert value == 25.0


def test_evaluation_budget_can_exceed_training_budget():
    training_budget = 768
    evaluation_budget = 4096

    assert (
        evaluation_budget
        > training_budget
    )


def test_evaluation_identity_changes_with_context_budget():
    from learning.evaluation import (
        developer_holdout_eval as module,
    )

    common = {
        "checkpoint_id":
            "checkpoint-test",

        "adapter_sha256":
            "a" * 64,

        "holdout_id":
            "holdout-test",

        "training_max_sequence_tokens":
            768,

        "compute_dtype":
            "bfloat16",
    }

    first = (
        module._sha256_text(
            module.canonical_json(
                {
                    **common,
                    "evaluation_max_sequence_tokens":
                        3072,
                }
            )
        )
    )

    second = (
        module._sha256_text(
            module.canonical_json(
                {
                    **common,
                    "evaluation_max_sequence_tokens":
                        4096,
                }
            )
        )
    )

    assert first != second


def test_record_exclusion_preserves_case_identity():

    from learning.evaluation.developer_holdout_eval import (
        _record_exclusion,
    )

    counts = {}
    cases = {}

    record = (
        _holdout_record(
            "phoenix-case"
        )
    )

    _record_exclusion(
        counts=counts,
        cases=cases,
        reason="cuda_out_of_memory",
        record=record,
    )

    assert counts == {
        "cuda_out_of_memory":
            1,
    }

    assert cases == {
        "cuda_out_of_memory": [
            "phoenix-case",
        ],
    }
