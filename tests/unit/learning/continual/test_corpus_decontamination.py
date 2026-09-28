from learning.continual.corpus_decontamination import (
    CorpusDecontaminationPolicy,
    decontamination_reason,
    policy_sha256,
)


def test_explicit_instance_is_blocked():

    policy = CorpusDecontaminationPolicy(
        source_id="test",
        blocked_instance_ids=[
            "instance-1"
        ],
    )

    assert (
        decontamination_reason(
            policy=policy,
            source_record_id="instance-1",
            repository="owner/repo",
            base_commit="abc",
            content_sha256="1" * 64,
        )
        == "benchmark_instance_blocked"
    )


def test_explicit_repository_commit_is_blocked():

    policy = CorpusDecontaminationPolicy(
        source_id="test",
        blocked_repository_commits=[
            "owner/repo@abcdef"
        ],
    )

    assert (
        decontamination_reason(
            policy=policy,
            source_record_id=None,
            repository="OWNER/REPO",
            base_commit="ABCDEF",
            content_sha256="2" * 64,
        )
        == "benchmark_repository_commit_blocked"
    )


def test_holdout_partition_is_deterministic():

    policy = CorpusDecontaminationPolicy(
        source_id="test",
        holdout_modulus=10,
        holdout_buckets=[
            0
        ],
    )

    first = decontamination_reason(
        policy=policy,
        source_record_id="stable-instance",
        repository="owner/repo",
        base_commit="abc",
        content_sha256="3" * 64,
    )

    second = decontamination_reason(
        policy=policy,
        source_record_id="stable-instance",
        repository="owner/repo",
        base_commit="abc",
        content_sha256="3" * 64,
    )

    assert first == second


def test_policy_fingerprint_is_stable():

    policy = CorpusDecontaminationPolicy(
        source_id="test",
        holdout_modulus=10,
        holdout_buckets=[
            0
        ],
    )

    assert (
        policy_sha256(
            policy
        )
        == policy_sha256(
            policy
        )
    )

    assert len(
        policy_sha256(
            policy
        )
    ) == 64
