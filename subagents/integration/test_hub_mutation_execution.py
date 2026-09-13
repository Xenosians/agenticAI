import asyncio

from pathlib import Path

from subagents.integration.support import (
    build_hub_integration_context,
)


async def run_mutation_execution_test(
    approval_db_path: Path,
):
    """
    Real Hub -> Account specialist -> durable approval ->
    explicit MCP execution.

    The test owns its ApprovalStore and MCP runtime.

    No process-global approval or MCP state is modified.
    """

    runtime = (
        build_hub_integration_context(
            approval_db_path=(
                approval_db_path
            )
        )
    )

    await runtime.mcp.start()

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
            "===== MUTATION PROPOSAL ====="
        )

        print(
            f"status: {result.status}"
        )

        print(
            f"routes: {result.routes}"
        )

        assert (
            result.status
            == "approval_required"
        )

        assert result.routes == [
            "account-specialist"
        ]

        assert (
            len(
                result.results
            )
            == 1
        )

        worker = (
            result.results[0]
        )

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

        assert (
            worker.approval_id
            is not None
        )

        approval_id = (
            worker.approval_id
        )

        print(
            "approval_id: "
            f"{approval_id}"
        )

        approval = (
            runtime
            .approvals
            .get_approval(
                approval_id
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

        assert (
            approval[
                "tool"
            ]
            == worker.proposed_tool
        )

        assert (
            approval[
                "arguments"
            ]
            == worker.proposed_arguments
        )

        # -------------------------------------------------
        # Explicit human-approval execution.
        #
        # No model is invoked again. ApprovalManager executes
        # the exact durable proposal already stored above.
        # -------------------------------------------------

        approval_result = (
            await
            runtime
            .approvals
            .approve_approval(
                approval_id
            )
        )

        print()

        print(
            "===== APPROVAL EXECUTION ====="
        )

        print(
            approval_result
        )

        print(
            "=============================="
        )

        assert (
            approval_result[
                "ok"
            ]
            is True
        )

        executed_approval = (
            approval_result[
                "approval"
            ]
        )

        tool_result = (
            approval_result[
                "result"
            ]
        )

        assert (
            executed_approval[
                "status"
            ]
            == "approved"
        )

        assert (
            executed_approval[
                "tool"
            ]
            == "unlock_user"
        )

        assert (
            executed_approval[
                "arguments"
            ]
            == {
                "user_id":
                    "jdoe"
            }
        )

        assert (
            tool_result[
                "ok"
            ]
            is True
        )

        assert (
            tool_result[
                "status"
            ]
            == "executed"
        )

        assert (
            tool_result[
                "user_id"
            ]
            == "jdoe"
        )

    finally:
        await runtime.mcp.stop()


def test_real_hub_mutation_executes_after_approval(
    tmp_path: Path,
):
    asyncio.run(
        run_mutation_execution_test(
            tmp_path
            / "approvals.sqlite3"
        )
    )