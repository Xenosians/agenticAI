import asyncio

from dataclasses import (
    asdict,
)

from subagents.core.orchestration.runtime import (
    AgentRuntime,
)

from subagents.core.definitions.types import (
    AgentDefinition,
    AgentTask,
)

from tools.result_cards import (
    RESULT_CARD_SCHEMA,
    validate_result_card,
)

from tools.result_presentation_registry import (
    build_tool_presentation,
)


class FakeAgentRegistry:
    def get(
        self,
        name,
    ):
        if (
            name
            != "developer-specialist"
        ):
            raise KeyError(
                name
            )

        return AgentDefinition(
            name=(
                "developer-specialist"
            ),
            description=(
                "Developer specialist"
            ),
            model=(
                "fake-model"
            ),
            tools=[
                "workspace_service_status",
            ],
            system_prompt=(
                "Use the allowed developer tools."
            ),
        )


class FakeInference:
    async def generate(
        self,
        *,
        model_key,
        messages,
        max_new_tokens,
        priority,
    ):
        return """
        [
            {
                "name": "workspace_service_status",
                "arguments": {}
            }
        ]
        """


class FakeToolGateway:
    def __init__(
        self,
        tool_result,
    ):
        self.tool_result = (
            tool_result
        )

    async def execute(
        self,
        agent,
        user_input,
        tool_name,
        arguments,
    ):
        return {
            "ok": True,

            "status":
                "success",

            "tool":
                tool_name,

            "result":
                self.tool_result,
        }


def test_result_card_validation_accepts_canonical_card():
    card = {
        "schema":
            RESULT_CARD_SCHEMA,

        "kind":
            "example",

        "title":
            "Example result",

        "status":
            "success",

        "fields": [
            {
                "label":
                    "Value",

                "value":
                    "42",
            }
        ],

        "sections": [
            {
                "kind":
                    "text",

                "title":
                    "Details",

                "content":
                    "Everything worked.",
            }
        ],
    }

    assert (
        validate_result_card(
            card
        )
        == card
    )


def test_result_card_validation_rejects_unknown_schema():
    card = {
        "schema":
            "something-else",

        "kind":
            "example",

        "title":
            "Example",

        "status":
            "success",

        "fields":
            [],

        "sections":
            [],
    }

    assert (
        validate_result_card(
            card
        )
        is None
    )


def test_developer_execution_builds_generic_result_card():
    tool_result = {
        "ok": True,
        "status": "success",
        "operation": "tests",
        "project_type": "python",
        "project_path": ".",
        "exit_code": 0,
        "stdout": "55 passed",
        "stderr": "",
        "timed_out": False,
        "duration_ms": 1234,
    }

    card = (
        build_tool_presentation(
            "workspace_run_tests",
            tool_result,
        )
    )

    assert card is not None

    assert card[
        "schema"
    ] == (
        RESULT_CARD_SCHEMA
    )

    assert card[
        "kind"
    ] == (
        "developer_execution"
    )

    assert card[
        "title"
    ] == "Tests"

    assert card[
        "status"
    ] == "success"

    assert {
        "label":
            "Project",

        "value":
            "python",
    } in card[
        "fields"
    ]

    assert card[
        "sections"
    ][
        0
    ][
        "kind"
    ] == (
        "preformatted"
    )

    assert card[
        "sections"
    ][
        0
    ][
        "content"
    ] == (
        "55 passed"
    )


def test_tool_without_presentation_builder_returns_none():
    card = (
        build_tool_presentation(
            "account_status",
            {
                "ok": True,
                "user_id": "jdoe",
                "enabled": True,
                "locked": False,
            },
        )
    )

    assert card is None


def test_agent_runtime_preserves_raw_tool_result_and_card():
    tool_result = {
        "ok": True,
        "status": "success",
        "provider": "docker_compose",
        "services": [
            {
                "service":
                    "samba-ad",

                "container":
                    "itsm-samba-ad",

                "state":
                    "running",

                "status":
                    "Up 2 minutes",

                "health":
                    None,
            }
        ],
        "service_count": 1,
        "error": None,
    }

    runtime = AgentRuntime(
        agent_registry=(
            FakeAgentRegistry()
        ),
        inference=(
            FakeInference()
        ),
        tool_gateway=(
            FakeToolGateway(
                tool_result
            )
        ),
    )

    task = AgentTask(
        task_id=(
            "structured-result-001"
        ),
        agent_name=(
            "developer-specialist"
        ),
        user_request=(
            "Check the workspace services."
        ),
    )

    result = asyncio.run(
        runtime.run(
            task
        )
    )

    assert result.status == (
        "success"
    )

    assert result.tool_result == (
        tool_result
    )

    assert (
        result.presentation
        is not None
    )

    assert result.presentation[
        "schema"
    ] == (
        RESULT_CARD_SCHEMA
    )

    assert result.presentation[
        "kind"
    ] == (
        "service_status"
    )

    serialized = (
        asdict(
            result
        )
    )

    assert serialized[
        "tool_result"
    ] == (
        tool_result
    )

    assert serialized[
        "presentation"
    ][
        "kind"
    ] == (
        "service_status"
    )