from __future__ import annotations

from learning.curriculum.chapters import (
    CurriculumChapter,
    CurriculumDefinition,
)


HUB_CURRICULUM = CurriculumDefinition(
    curriculum_id="hub-general-language-code-routing-v1",
    version="1.0.0",
    target_component="hub",
    required_safety_pass_rate=1.0,
    chapters=[
        CurriculumChapter(
            chapter_id="hub-01-structured-instructions",
            ordinal=1,
            title="Structured instruction following",
            description=(
                "Follow the user request while preserving strict JSON and "
                "routing-contract structure."
            ),
            focus_tags=[
                "instruction-following",
                "structured-output",
                "json-contract",
            ],
            failure_codes=[
                "tool_parse_error",
                "invalid_tool_call_count",
            ],
            mastery_thresholds={
                "instruction_following": 0.90,
                "json_contract_validity": 0.98,
            },
            min_reviewed_examples=8,
        ),
        CurriculumChapter(
            chapter_id="hub-02-resource-operation",
            ordinal=2,
            title="Primary resource and operation",
            description=(
                "Identify the object the user wants acted on and preserve the "
                "requested operation without substituting a nearby capability."
            ),
            focus_tags=[
                "primary-resource",
                "operation-identity",
                "capability-selection",
            ],
            failure_codes=[
                "semantic_tool_not_allowed",
                "semantic_tool_not_available",
                "unknown_tool",
            ],
            mastery_thresholds={
                "resource_operation_accuracy": 0.92,
                "capability_selection_accuracy": 0.92,
            },
            min_reviewed_examples=10,
        ),
        CurriculumChapter(
            chapter_id="hub-03-scope-context",
            ordinal=3,
            title="Scope, container, and contextual identity",
            description=(
                "Keep repository, project, parent, container, and contextual "
                "identities as scope when they are not the primary resource."
            ),
            focus_tags=[
                "scope",
                "container",
                "context",
                "identifier-grounding",
            ],
            failure_codes=[
                "grounding_failed",
                "semantic_argument_not_allowed",
            ],
            mastery_thresholds={
                "scope_binding_accuracy": 0.94,
                "identifier_grounding_accuracy": 0.96,
            },
            min_reviewed_examples=10,
        ),
        CurriculumChapter(
            chapter_id="hub-04-delegation",
            ordinal=4,
            title="Single-specialist delegation",
            description=(
                "Select the owning specialist and capability family while "
                "keeping general conversation in the primary assistant."
            ),
            focus_tags=[
                "delegation",
                "agent-selection",
                "domain-routing",
            ],
            failure_codes=[
                "agent_tool_not_allowed",
            ],
            mastery_thresholds={
                "delegation_accuracy": 0.94,
                "domain_routing_accuracy": 0.94,
            },
            min_reviewed_examples=10,
        ),
        CurriculumChapter(
            chapter_id="hub-05-grounded-bindings",
            ordinal=5,
            title="Explicit grounded argument binding",
            description=(
                "Bind every concrete grounded value explicitly supplied by "
                "the current request while refusing to invent missing values."
            ),
            focus_tags=[
                "grounded-arguments",
                "exact-literal-binding",
                "argument-recall",
                "argument-precision",
            ],
            failure_codes=[
                "semantic_argument_unbound",
                "semantic_grounded_argument_invalid",
                "semantic_argument_forbidden",
            ],
            mastery_thresholds={
                "grounded_argument_recall": 0.96,
                "grounded_argument_precision": 0.98,
            },
            min_reviewed_examples=12,
        ),
        CurriculumChapter(
            chapter_id="hub-06-governed-mutations",
            ordinal=6,
            title="Mutation and approval boundaries",
            description=(
                "Distinguish reads from mutations, preserve exact mutation "
                "intent, and never treat approval as semantic correctness."
            ),
            focus_tags=[
                "mutation",
                "approval",
                "effect",
                "authorization-boundary",
            ],
            failure_codes=[
                "semantic_effect_mismatch",
                "policy_denied",
                "tool_execution_denied",
            ],
            mastery_thresholds={
                "mutation_effect_accuracy": 0.96,
                "approval_boundary_accuracy": 1.0,
            },
            min_reviewed_examples=12,
        ),
        CurriculumChapter(
            chapter_id="hub-07-repair-clarification",
            ordinal=7,
            title="Bounded repair and clarification",
            description=(
                "Repair malformed semantic plans without changing the user's "
                "requested resource or operation; clarify when safe grounding "
                "cannot be established."
            ),
            focus_tags=[
                "repair",
                "clarification",
                "fail-closed",
            ],
            failure_codes=[
                "semantic_clarification_required",
            ],
            mastery_thresholds={
                "repair_success_rate": 0.92,
                "clarification_accuracy": 0.94,
            },
            min_reviewed_examples=10,
        ),
        CurriculumChapter(
            chapter_id="hub-08-multidomain-workflows",
            ordinal=8,
            title="Cross-domain governed workflows",
            description=(
                "Coordinate multiple specialists while preserving typed "
                "resource context, exact grounding, and authorization."
            ),
            focus_tags=[
                "cross-domain",
                "workflow",
                "typed-context",
            ],
            failure_codes=[
                "conditional_source_failed",
                "conditional_source_missing",
            ],
            mastery_thresholds={
                "workflow_accuracy": 0.92,
                "cross_domain_grounding_accuracy": 0.96,
            },
            min_reviewed_examples=16,
        ),
    ],
)


DEVELOPER_CURRICULUM = CurriculumDefinition(
    curriculum_id="developer-specialist-code-v1",
    version="1.0.0",
    target_component="developer-specialist",
    required_safety_pass_rate=1.0,
    chapters=[
        CurriculumChapter(
            chapter_id="dev-01-workspace-reads",
            ordinal=1,
            title="Workspace reads",
            description=(
                "Read, list, search, and inspect trusted logical repositories "
                "without introducing arbitrary filesystem authority."
            ),
            focus_tags=[
                "workspace-read",
                "repository-scope",
            ],
            failure_codes=[
                "grounding_failed",
            ],
            mastery_thresholds={
                "workspace_read_accuracy": 0.95,
                "repository_scope_accuracy": 0.98,
            },
            min_reviewed_examples=8,
        ),
        CurriculumChapter(
            chapter_id="dev-02-git-observation",
            ordinal=2,
            title="Git observation operations",
            description=(
                "Preserve the distinction between status, history, branches, "
                "diffs, staged diffs, and changed-file overviews."
            ),
            focus_tags=[
                "git-status",
                "git-log",
                "git-diff",
                "git-branches",
            ],
            failure_codes=[
                "semantic_tool_not_allowed",
                "agent_tool_not_allowed",
            ],
            mastery_thresholds={
                "git_read_operation_accuracy": 0.95,
                "git_repository_scope_accuracy": 0.98,
            },
            min_reviewed_examples=10,
        ),
        CurriculumChapter(
            chapter_id="dev-03-build-test",
            ordinal=3,
            title="Build and test selection",
            description=(
                "Choose the correct governed test/build capability and "
                "interpret execution results without substituting raw shell."
            ),
            focus_tags=[
                "tests",
                "build",
                "execution-result",
            ],
            failure_codes=[
                "tool_execution_error",
            ],
            mastery_thresholds={
                "test_build_selection_accuracy": 0.95,
                "execution_result_accuracy": 0.94,
            },
            min_reviewed_examples=10,
        ),
        CurriculumChapter(
            chapter_id="dev-04-code-symbols",
            ordinal=4,
            title="Code symbols and local dependencies",
            description=(
                "Reason about functions, classes, modules, imports, callers, "
                "and related tests from semantically chunked source context."
            ),
            focus_tags=[
                "symbols",
                "dependencies",
                "tests",
                "semantic-code-chunks",
            ],
            failure_codes=[],
            mastery_thresholds={
                "code_symbol_reasoning_accuracy": 0.92,
                "dependency_reasoning_accuracy": 0.90,
            },
            min_reviewed_examples=12,
        ),
        CurriculumChapter(
            chapter_id="dev-05-single-file-patches",
            ordinal=5,
            title="Single-file patches",
            description=(
                "Produce minimal, test-backed changes to one file while "
                "preserving repository contracts."
            ),
            focus_tags=[
                "single-file-patch",
                "regression-tests",
            ],
            failure_codes=[],
            mastery_thresholds={
                "single_file_patch_accuracy": 0.90,
                "regression_test_pass_rate": 0.98,
            },
            min_reviewed_examples=12,
        ),
        CurriculumChapter(
            chapter_id="dev-06-multi-file-patches",
            ordinal=6,
            title="Multi-file patches",
            description=(
                "Coordinate related source and test changes across a module "
                "without unrelated edits."
            ),
            focus_tags=[
                "multi-file-patch",
                "module-contracts",
            ],
            failure_codes=[],
            mastery_thresholds={
                "multi_file_patch_accuracy": 0.88,
                "regression_test_pass_rate": 0.98,
            },
            min_reviewed_examples=14,
        ),
        CurriculumChapter(
            chapter_id="dev-07-cross-repository",
            ordinal=7,
            title="Cross-repository engineering",
            description=(
                "Reason across AI, backend, and frontend contracts while "
                "keeping repository identities explicit and trusted."
            ),
            focus_tags=[
                "cross-repository",
                "contract-reasoning",
            ],
            failure_codes=[],
            mastery_thresholds={
                "cross_repository_accuracy": 0.88,
                "contract_preservation_accuracy": 0.94,
            },
            min_reviewed_examples=16,
        ),
        CurriculumChapter(
            chapter_id="dev-08-governed-mutations",
            ordinal=8,
            title="Governed Git and workspace mutations",
            description=(
                "Apply staging, branch, commit, push, and workspace mutations "
                "only through approval and policy boundaries."
            ),
            focus_tags=[
                "git-mutation",
                "approval",
                "policy",
            ],
            failure_codes=[
                "policy_denied",
                "tool_execution_denied",
            ],
            mastery_thresholds={
                "governed_mutation_accuracy": 0.95,
                "approval_boundary_accuracy": 1.0,
            },
            min_reviewed_examples=16,
        ),
    ],
)


_CURRICULA = {
    "hub": HUB_CURRICULUM,
    "developer": DEVELOPER_CURRICULUM,
    HUB_CURRICULUM.curriculum_id: HUB_CURRICULUM,
    DEVELOPER_CURRICULUM.curriculum_id: DEVELOPER_CURRICULUM,
}


def get_curriculum(name: str) -> CurriculumDefinition:
    key = name.strip()

    try:
        return _CURRICULA[key].model_copy(deep=True)
    except KeyError as exc:
        raise ValueError(
            "Unknown curriculum. Expected one of: "
            + ", ".join(sorted(_CURRICULA))
        ) from exc
