import asyncio

from pathlib import Path

from subagents.integration.support import (
    build_hub_integration_context,
)


class FailIfCalledMCP:
    async def call_tool(
        self,
        tool_name,
        arguments,
    ):
        raise AssertionError(
            "MCP execution must NOT occur "
            "before approval. "
            f"Attempted tool={tool_name}, "
            f"arguments={arguments}"
        )


async def run_mutation_approval_test(
    approval_db_path: Path,
):
    runtime = (
        build_hub_integration_context(
            approval_db_path=(
                approval_db_path
            ),

            mcp=(
                FailIfCalledMCP()
            ),
        )
    )

    try:
        await runtime.inference.warm(
            runtime
            .settings
            .hub_model_key
        )

        result = (
            await runtime.hub.run(
                "Unlock jdoe"
            )
        )

        print()

        print(
            "===== HUB MUTATION TEST ====="
        )

        print(
            f"status: {result.status}"
        )

        print(
            f"routes: {result.routes}"
        )

        print(
            f"answer: {result.answer}"
        )

        for worker_result in (
            result.results
        ):
            print(
                "---"
            )

            print(
                "worker: "
                f"{worker_result.agent_name}"
            )

            print(
                "status: "
                f"{worker_result.status}"
            )

            print(
                "tool: "
                f"{worker_result.proposed_tool}"
            )

            print(
                "arguments: "
                f"{worker_result.proposed_arguments}"
            )

            print(
                "answer: "
                f"{worker_result.answer}"
            )

            print(
                "error: "
                f"{worker_result.error}"
            )

        print(
            "============================="
        )

        # ====================================================
        # ROUTING CONTRACT
        # ====================================================

        assert (
            result.routes
            == [
                "account-specialist"
            ]
        )

        # ====================================================
        # HUB OUTCOME CONTRACT
        # ====================================================

        assert (
            result.status
            == "approval_required"
        )

        assert (
            len(
                result.results
            )
            == 1
        )

        worker = (
            result.results[
                0
            ]
        )

        # ====================================================
        # SPECIALIST CONTRACT
        # ====================================================

        assert (
            worker.agent_name
            == "account-specialist"
        )

        assert (
            worker.proposed_tool
            == "unlock_user"
        )

        assert (
            worker.proposed_arguments
            == {
                "user_id":
                    "jdoe"
            }
        )

        # ====================================================
        # MACHINE-READABLE APPROVAL CONTRACT
        #
        # Human-facing prose is intentionally NOT treated as a
        # stable API. Tool presentation wording may evolve
        # without changing the approval state machine.
        # ====================================================

        assert (
            worker.status
            == "approval_required"
        )

        assert (
            worker.outcome_code
            == "approval_required"
        )

        assert (
            worker.approval_id
            is not None
        )

        # ====================================================
        # HUMAN-FACING PRESENTATION
        #
        # We only require the durable approval identifier to be
        # surfaced to the user. We do not couple the test to a
        # particular English sentence.
        # ====================================================

        assert (
            worker.answer
            is not None
        )

        assert (
            worker.approval_id
            in worker.answer
        )

        # ====================================================
        # DURABLE APPROVAL STORE
        # ====================================================

        approval = (
            runtime
            .approvals
            .get_approval(
                worker.approval_id
            )
        )

        assert (
            approval
            is not None
        )

        assert (
            approval[
                "status"
            ]
            == "pending"
        )

    finally:
        runtime.model_manager.unload_all()


def test_real_hub_mutation_stops_at_approval(
    tmp_path: Path,
):
    asyncio.run(
        run_mutation_approval_test(
            tmp_path
            / "approvals.sqlite3"
        )
    )