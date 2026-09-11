import asyncio

from pathlib import Path

import agent.approvals as approvals_module

from agent.approvals import (
    APPROVAL_LOCKS,
    approve_approval,
    get_approval,
)

from agent.approval_store import (
    ApprovalStore,
)

from agent.mcp_client import (
    mcp_runtime,
)

from subagents.hub import (
    build_hub,
)


async def run_mutation_execution_test(
    approval_db_path: Path,
):
    """
    Real Hub -> Account specialist -> approval -> MCP execution.

    The test uses its own temporary durable SQLite approval
    database so it cannot consume or modify approvals created
    by manual runs or other tests.
    """

    # ---------------------------------------------------------
    # Isolated durable approval store
    # ---------------------------------------------------------

    store = ApprovalStore(
        approval_db_path
    )

    store.initialize()

    approvals_module._APPROVAL_STORE = (
        store
    )

    APPROVAL_LOCKS.clear()

    await mcp_runtime.start()

    try:
        # -------------------------------------------------
        # Real Hub + real Account worker
        # -------------------------------------------------

        hub = build_hub()

        result = await hub.run(
            "Unlock jdoe"
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

        assert len(
            result.results
        ) == 1

        worker = (
            result.results[0]
        )

        # -------------------------------------------------
        # Worker proposal
        # -------------------------------------------------

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
            f"approval_id: "
            f"{approval_id}"
        )

        # -------------------------------------------------
        # Verify durable approval proposal
        # -------------------------------------------------

        approval = get_approval(
            approval_id
        )

        assert approval is not None

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
        # Explicit approval
        #
        # Important:
        #
        # We DO NOT run Ministral again.
        # We DO NOT run Qwen again.
        #
        # approve_approval() executes the exact proposal
        # already persisted above.
        # -------------------------------------------------

        approval_result = (
            await approve_approval(
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

        # -------------------------------------------------
        # Approval execution succeeded
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Real MCP / LDAP result
        # -------------------------------------------------

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
        await mcp_runtime.stop()

        APPROVAL_LOCKS.clear()

        approvals_module._APPROVAL_STORE = (
            None
        )


def test_real_hub_mutation_executes_after_approval(
    tmp_path: Path,
):
    approval_db_path = (
        tmp_path
        / "approvals.sqlite3"
    )

    asyncio.run(
        run_mutation_execution_test(
            approval_db_path
        )
    )