import asyncio

from subagents.core.definitions.types import (
    AgentDefinition,
)

from subagents.core.tooling.gateway import (
    ToolGateway,
    validate_grounded_arguments,
)


def test_grounded_list_accepts_all_exact_user_values():

    (
        valid,
        error,
    ) = (
        validate_grounded_arguments(
            user_input=(
                "Stage src/app.py and "
                "tests/test_app.py in the AI repository."
            ),

            arguments={
                "repository":
                    "ai",

                "paths": [
                    "src/app.py",
                    "tests/test_app.py",
                ],
            },

            grounded_arguments=[
                "repository",
                "paths",
            ],
        )
    )

    assert (
        valid
        is True
    )

    assert (
        error
        is None
    )


def test_grounded_list_rejects_invented_member():

    (
        valid,
        error,
    ) = (
        validate_grounded_arguments(
            user_input=(
                "Stage src/app.py in the AI repository."
            ),

            arguments={
                "repository":
                    "ai",

                "paths": [
                    "src/app.py",
                    "secret.py",
                ],
            },

            grounded_arguments=[
                "repository",
                "paths",
            ],
        )
    )

    assert (
        valid
        is False
    )

    assert (
        error
        is not None
    )

    assert (
        "secret.py"
        in error
    )


def test_grounded_list_rejects_empty_collection():

    (
        valid,
        error,
    ) = (
        validate_grounded_arguments(
            user_input=(
                "Stage src/app.py."
            ),

            arguments={
                "paths":
                    [],
            },

            grounded_arguments=[
                "paths",
            ],
        )
    )

    assert (
        valid
        is False
    )

    assert (
        error
        == "paths must not be empty."
    )


def test_grounded_list_rejects_duplicates():

    (
        valid,
        error,
    ) = (
        validate_grounded_arguments(
            user_input=(
                "Stage src/app.py."
            ),

            arguments={
                "paths": [
                    "src/app.py",
                    "src/app.py",
                ],
            },

            grounded_arguments=[
                "paths",
            ],
        )
    )

    assert (
        valid
        is False
    )

    assert (
        error
        == (
            "paths must not contain "
            "duplicate values."
        )
    )


def test_git_like_list_mutation_stops_at_approval():

    agent = (
        AgentDefinition(
            name=(
                "developer-specialist"
            ),

            description=(
                "Developer worker"
            ),

            tools=[
                "workspace_git_stage_files",
            ],

            model=(
                "test-model"
            ),
        )
    )

    tools = {
        "workspace_git_stage_files": {
            "risk":
                "low",

            "requires_approval":
                True,

            "grounded_arguments": [
                "repository",
                "paths",
            ],
        },
    }

    approvals = []

    def lookup(
        name,
    ):

        return (
            tools.get(
                name
            )
        )

    def create_approval(
        tool_name,
        arguments,
        risk=None,
    ):

        approvals.append(
            {
                "tool":
                    tool_name,

                "arguments":
                    arguments,

                "risk":
                    risk,
            }
        )

        return {
            "id":
                "approval-stage-1",
        }

    class FailIfCalledMCP:

        async def call_tool(
            self,
            tool_name,
            arguments,
        ):

            raise AssertionError(
                "Mutation executed before approval."
            )

    gateway = (
        ToolGateway(
            approval_creator=(
                create_approval
            ),

            mcp=(
                FailIfCalledMCP()
            ),

            tool_lookup=(
                lookup
            ),
        )
    )

    result = (
        asyncio.run(
            gateway.execute(
                agent=(
                    agent
                ),

                user_input=(
                    "Stage src/app.py and tests/test_app.py "
                    "in the AI repository."
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
                    ],
                },
            )
        )
    )

    assert (
        result[
            "status"
        ]
        == "approval_required"
    )

    assert (
        result[
            "decision_code"
        ]
        == "approval_required"
    )

    assert (
        result[
            "risk"
        ]
        == "low"
    )

    assert (
        approvals
        == [
            {
                "tool":
                    "workspace_git_stage_files",

                "arguments": {
                    "repository":
                        "ai",

                    "paths": [
                        "src/app.py",
                        "tests/test_app.py",
                    ],
                },

                "risk":
                    "low",
            }
        ]
    )