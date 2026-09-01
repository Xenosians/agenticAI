import asyncio

from agent.mcp_client import mcp_runtime

from subagents.hub import build_hub


async def fail_if_mcp_called(
    tool_name,
    arguments,
):
    """
    Mutation proposals must stop at the approval boundary.

    If this function is called, the security boundary failed.
    """

    raise AssertionError(
        "MCP execution must NOT occur before approval. "
        f"Attempted tool={tool_name}, arguments={arguments}"
    )


async def run_mutation_approval_test(
    monkeypatch,
):
    # -------------------------------------------------
    # Hard guard:
    # Any MCP execution makes this test fail immediately.
    # -------------------------------------------------

    monkeypatch.setattr(
        mcp_runtime,
        "call_tool",
        fail_if_mcp_called,
    )

    # -------------------------------------------------
    # Real Hub + real worker models
    # -------------------------------------------------

    hub = build_hub()

    result = await hub.run(
        "Unlock jdoe"
    )

    print()
    print("===== HUB MUTATION TEST =====")
    print(f"status: {result.status}")
    print(f"routes: {result.routes}")
    print(f"answer: {result.answer}")

    for worker_result in result.results:
        print("---")
        print(
            f"worker: "
            f"{worker_result.agent_name}"
        )
        print(
            f"status: "
            f"{worker_result.status}"
        )
        print(
            f"tool: "
            f"{worker_result.proposed_tool}"
        )
        print(
            f"arguments: "
            f"{worker_result.proposed_arguments}"
        )
        print(
            f"answer: "
            f"{worker_result.answer}"
        )
        print(
            f"error: "
            f"{worker_result.error}"
        )

    print("=============================")

    # -------------------------------------------------
    # Hub routing
    # -------------------------------------------------

    assert result.routes == [
        "account-specialist"
    ]

    # -------------------------------------------------
    # Overall operation must stop at approval.
    # -------------------------------------------------

    assert (
        result.status
        == "approval_required"
    )

    assert len(result.results) == 1

    worker = result.results[0]

    # -------------------------------------------------
    # Correct specialist
    # -------------------------------------------------

    assert (
        worker.agent_name
        == "account-specialist"
    )

    # -------------------------------------------------
    # Correct worker proposal
    # -------------------------------------------------

    assert (
        worker.proposed_tool
        == "unlock_user"
    )

    assert worker.proposed_arguments == {
        "user_id": "jdoe"
    }

    # -------------------------------------------------
    # Critical security assertion
    # -------------------------------------------------

    assert (
        worker.status
        == "approval_required"
    )

    assert worker.answer is not None

    assert (
        "Approval required"
        in worker.answer
    )


def test_real_hub_mutation_stops_at_approval(
    monkeypatch,
):
    asyncio.run(
        run_mutation_approval_test(
            monkeypatch
        )
    )