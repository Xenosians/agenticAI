from learning.code_corpus.types import (
    CodeCorpusChunk,
)
from learning.curriculum.catalog import (
    get_curriculum,
)
from learning.curriculum.mastery import (
    CurriculumState,
    initial_curriculum_state,
)
from learning.training.membership import (
    MembershipCandidate,
    assign_partitions,
    cap_code_candidates,
    chapter_ids_for_failure_codes,
    classify_behavior_role,
    classify_code_role,
)


def _hub_state_at(
    chapter_id: str,
    *,
    mastered: list[str] | None = None,
):
    curriculum = get_curriculum(
        "hub"
    )

    initial = initial_curriculum_state(
        curriculum
    )

    return CurriculumState(
        curriculum_id=(
            initial.curriculum_id
        ),
        curriculum_version=(
            initial.curriculum_version
        ),
        target_component=(
            initial.target_component
        ),
        updated_at=(
            initial.updated_at
        ),
        current_chapter_id=(
            chapter_id
        ),
        mastered_chapter_ids=(
            mastered
            or []
        ),
        chapter_attempt_counts={},
        best_mastery_scores={},
        last_evaluation_id=None,
        status="learning",
    )


def _developer_state_at(
    chapter_id: str,
    *,
    mastered: list[str] | None = None,
):
    curriculum = get_curriculum(
        "developer"
    )

    initial = initial_curriculum_state(
        curriculum
    )

    return CurriculumState(
        curriculum_id=(
            initial.curriculum_id
        ),
        curriculum_version=(
            initial.curriculum_version
        ),
        target_component=(
            initial.target_component
        ),
        updated_at=(
            initial.updated_at
        ),
        current_chapter_id=(
            chapter_id
        ),
        mastered_chapter_ids=(
            mastered
            or []
        ),
        chapter_attempt_counts={},
        best_mastery_scores={},
        last_evaluation_id=None,
        status="learning",
    )


def _chunk(
    *,
    source_kind="project",
    audiences=None,
    hints=None,
    eligible=True,
    repository="ai",
    chunk_id="chunk-1",
    content_sha256="a" * 64,
):
    return CodeCorpusChunk(
        chunk_id=chunk_id,
        source_kind=source_kind,
        logical_repository=repository,
        revision_kind="git",
        revision_id="abc123",
        git_commit_sha="abc123",
        git_dirty=False,
        relative_path="module.py",
        language="python",
        symbol="work",
        symbol_kind="function",
        semantic_depth=1,
        start_line=1,
        end_line=3,
        content=(
            "def work():\n"
            "    return True\n"
        ),
        content_sha256=content_sha256,
        audiences=(
            audiences
            or [
                "developer-specialist"
            ]
        ),
        curriculum_hints=(
            hints
            or []
        ),
        training_eligible=eligible,
    )


def test_grounded_argument_failures_map_to_hub_chapter_five():
    curriculum = get_curriculum(
        "hub"
    )

    chapters = (
        chapter_ids_for_failure_codes(
            curriculum=curriculum,
            failure_codes={
                "semantic_argument_unbound",
                "semantic_grounded_argument_invalid",
            },
        )
    )

    assert chapters == [
        "hub-05-grounded-bindings"
    ]


def test_behavior_role_is_current_or_mastered_replay():
    curriculum = get_curriculum(
        "hub"
    )

    current = _hub_state_at(
        "hub-05-grounded-bindings",
        mastered=[
            "hub-04-delegation",
        ],
    )

    assert (
        classify_behavior_role(
            curriculum=curriculum,
            state=current,
            failure_codes={
                "semantic_argument_unbound",
            },
        )
        == "current_behavior"
    )

    assert (
        classify_behavior_role(
            curriculum=curriculum,
            state=current,
            failure_codes={
                "agent_tool_not_allowed",
            },
        )
        == "replay_behavior"
    )


def test_hub_never_consumes_project_specific_code():
    state = _hub_state_at(
        "hub-01-structured-instructions"
    )

    chunk = _chunk(
        source_kind="project",
        audiences=[
            "hub",
            "developer-specialist",
        ],
    )

    assert (
        classify_code_role(
            target_component="hub",
            state=state,
            chunk=chunk,
        )
        is None
    )


def test_hub_general_code_is_foundation_code():
    state = _hub_state_at(
        "hub-01-structured-instructions"
    )

    chunk = _chunk(
        source_kind="general",
        audiences=[
            "hub",
        ],
        repository="general-python",
    )

    assert (
        classify_code_role(
            target_component="hub",
            state=state,
            chunk=chunk,
        )
        == "foundation_code"
    )


def test_developer_code_respects_current_and_mastered_chapters():
    current = _developer_state_at(
        "dev-04-code-symbols",
        mastered=[
            "dev-03-build-test",
        ],
    )

    current_chunk = _chunk(
        hints=[
            "dev-04-code-symbols",
        ],
    )

    replay_chunk = _chunk(
        chunk_id="chunk-2",
        content_sha256="b" * 64,
        hints=[
            "dev-03-build-test",
        ],
    )

    assert (
        classify_code_role(
            target_component=(
                "developer-specialist"
            ),
            state=current,
            chunk=current_chunk,
        )
        == "current_code"
    )

    assert (
        classify_code_role(
            target_component=(
                "developer-specialist"
            ),
            state=current,
            chunk=replay_chunk,
        )
        == "replay_code"
    )


def test_dirty_or_ineligible_code_cannot_enter_membership():
    state = _developer_state_at(
        "dev-04-code-symbols"
    )

    chunk = _chunk(
        eligible=False,
        hints=[
            "dev-04-code-symbols",
        ],
    )

    assert (
        classify_code_role(
            target_component=(
                "developer-specialist"
            ),
            state=state,
            chunk=chunk,
        )
        is None
    )


def _candidate(
    *,
    member_id: str,
    split_group: str,
    role="current_behavior",
    kind="preference",
    repository=None,
):
    return MembershipCandidate(
        member_id=member_id,
        member_kind=kind,
        role=role,
        target_component="hub",
        chapter_id="hub-01-structured-instructions",
        source_id=member_id,
        source_artifact="source.jsonl",
        source_artifact_sha256="f" * 64,
        lineage_id=member_id,
        split_group=split_group,
        logical_repository=repository,
    )


def test_partitioning_never_splits_one_lineage_group():
    candidates = [
        _candidate(
            member_id="a1",
            split_group="trajectory:a",
        ),
        _candidate(
            member_id="a2",
            split_group="trajectory:a",
        ),
        _candidate(
            member_id="b1",
            split_group="trajectory:b",
        ),
        _candidate(
            member_id="c1",
            split_group="trajectory:c",
        ),
    ]

    records = assign_partitions(
        candidates,
        validation_fraction=0.25,
        seed="stable",
    )

    partitions_by_group = {}

    for record in records:
        partitions_by_group.setdefault(
            record.split_group,
            set(),
        ).add(
            record.partition
        )

    assert all(
        len(
            values
        )
        == 1
        for values in (
            partitions_by_group
            .values()
        )
    )

    assert {
        item.partition
        for item in records
    } == {
        "train",
        "validation",
    }


def test_code_cap_preserves_behavior_members():
    behavior = _candidate(
        member_id="behavior-1",
        split_group="trajectory:1",
    )

    code = [
        _candidate(
            member_id=f"code-{index}",
            split_group=f"file:{index}",
            role="current_code",
            kind="code_chunk",
            repository=(
                "ai"
                if index % 2
                else "backend"
            ),
        )
        for index in range(
            10
        )
    ]

    selected, dropped = cap_code_candidates(
        [
            behavior,
            *code,
        ],
        max_code_members=4,
        max_repository_fraction=0.75,
        seed="stable",
    )

    assert behavior in selected
    assert sum(
        1
        for item in selected
        if item.member_kind
        == "code_chunk"
    ) == 4

    assert dropped == 6
