import asyncio

from subagents.core.definitions.types import (
    SpecialistRequest,
)

from subagents.core.orchestration.orchestrator import (
    Orchestrator,
)


class FakeRouter:
    async def route(
        self,
        user_request,
    ):

        return [
            SpecialistRequest(
                agent_name=(
                    "account-specialist"
                ),

                instructions=(
                    "Check jdoe."
                ),

                semantic_intent=None,
            )
        ]


class NeverRunRuntime:
    async def run(
        self,
        task,
    ):

        raise AssertionError(
            "Runtime must not execute without "
            "a required semantic intent contract."
        )


class FakePrimaryAssistant:
    async def respond(
        self,
        user_request,
    ):

        return (
            "direct response"
        )

    async def synthesize(
        self,
        user_request,
        results,
    ):

        return (
            "synthesized"
        )


def test_required_semantic_contract_fails_closed_before_runtime(
):

    orchestrator = (
        Orchestrator(
            router=(
                FakeRouter()
            ),

            runtime=(
                NeverRunRuntime()
            ),

            primary_assistant=(
                FakePrimaryAssistant()
            ),

            require_semantic_intent=True,
        )
    )

    result = (
        asyncio.run(
            orchestrator.run(
                "Check jdoe."
            )
        )
    )

    assert (
        result.status
        == "partial_error"
    )

    assert (
        len(
            result.results
        )
        == 1
    )

    assert (
        result.results[
            0
        ].outcome_code
        == "semantic_intent_missing"
    )

    assert (
        "semantic intent contract"
        in result.answer.lower()
    )
