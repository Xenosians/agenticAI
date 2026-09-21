from subagents.core.definitions.types import (
    AgentDefinition,
    SemanticIntent,
)

from subagents.core.orchestration.semantic_guard import (
    SemanticGuard,
)


TOOL = {
    "description":
        "Stage explicitly selected files.",

    "risk":
        "low",

    "requires_approval":
        True,

    "grounded_arguments": [
        "repository",
        "paths",
    ],

    "parameters": {
        "repository": {
            "type":
                "str",
        },

        "paths": {
            "type":
                "list[str]",
        },
    },
}


def lookup(
    name: str,
):

    if (
        name
        == "workspace_git_stage_files"
    ):

        return (
            TOOL
        )

    return None


def agent(
):

    return (
        AgentDefinition(
            name=(
                "developer-specialist"
            ),

            description=(
                "Developer"
            ),

            tools=[
                "workspace_git_stage_files",
            ],

            model=(
                "test-model"
            ),
        )
    )


def intent(
):

    return (
        SemanticIntent(
            summary=(
                "Stage two explicit files."
            ),

            effect=(
                "mutation"
            ),

            allowed_tools=[
                "workspace_git_stage_files",
            ],

            forbidden_tools=[],

            allowed_arguments={
                "repository": [
                    "ai",
                ],

                "paths": [
                    "src/app.py",
                    "tests/test_app.py",
                ],
            },

            forbidden_arguments={},

            max_tool_calls=1,

            clarification_required=False,
        )
    )


def test_exact_grounded_collection_is_allowed():

    decision = (
        SemanticGuard(
            tool_lookup=(
                lookup
            )
        )
        .evaluate(
            agent=(
                agent()
            ),

            intent=(
                intent()
            ),

            tool_name=(
                "workspace_git_stage_files"
            ),

            arguments={
                "repository":
                    "ai",

                "paths": [
                    "tests/test_app.py",
                    "src/app.py",
                ],
            },
        )
    )

    assert (
        decision.allowed
        is True
    )


def test_grounded_collection_cannot_omit_requested_target():

    decision = (
        SemanticGuard(
            tool_lookup=(
                lookup
            )
        )
        .evaluate(
            agent=(
                agent()
            ),

            intent=(
                intent()
            ),

            tool_name=(
                "workspace_git_stage_files"
            ),

            arguments={
                "repository":
                    "ai",

                "paths": [
                    "src/app.py",
                ],
            },
        )
    )

    assert (
        decision.allowed
        is False
    )

    assert (
        decision.decision_code
        == "semantic_argument_not_allowed"
    )


def test_grounded_collection_cannot_add_target():

    decision = (
        SemanticGuard(
            tool_lookup=(
                lookup
            )
        )
        .evaluate(
            agent=(
                agent()
            ),

            intent=(
                intent()
            ),

            tool_name=(
                "workspace_git_stage_files"
            ),

            arguments={
                "repository":
                    "ai",

                "paths": [
                    "src/app.py",
                    "tests/test_app.py",
                    "secret.py",
                ],
            },
        )
    )

    assert (
        decision.allowed
        is False
    )

    assert (
        decision.decision_code
        == "semantic_argument_not_allowed"
    )