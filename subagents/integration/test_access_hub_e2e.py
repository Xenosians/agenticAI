import asyncio

from agent.mcp_client import mcp_runtime

from subagents.hub import build_hub


async def run_access_hub_e2e():
    """
    Real Access vertical-slice test.

    Expected runtime path:

        User request
            ↓
        Ministral 3B Hub
            ↓
        access-specialist
            ↓
        Qwen3-0.6B
            ↓
        check_access
            ↓
        ToolGateway
            ↓
        MCP
            ↓
        LDAP / Samba AD
    """

    await mcp_runtime.start()

    try:
        hub = build_hub()

        result = await hub.run(
            "Does jdoe have VPN access?"
        )

        print()
        print("===== ACCESS HUB E2E =====")
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
                f"tool: "
                f"{worker_result.proposed_tool}"
            )
            print(
                f"arguments: "
                f"{worker_result.proposed_arguments}"
            )
            print(
                f"worker answer: "
                f"{worker_result.answer}"
            )

        print("==========================")

        # ========================================================
        # Hub result
        # ========================================================

        assert result.status == "success"

        # Ministral must route specifically to Access.
        assert result.routes == [
            "access-specialist"
        ]

        assert len(result.results) == 1

        worker = result.results[0]

        # ========================================================
        # Worker selection
        # ========================================================

        assert (
            worker.agent_name
            == "access-specialist"
        )

        # ========================================================
        # Qwen3 tool proposal
        # ========================================================

        assert (
            worker.proposed_tool
            == "check_access"
        )

        assert worker.proposed_arguments == {
            "user_id": "jdoe",
            "resource": "VPN",
        }

        # ========================================================
        # Real execution result
        #
        # We intentionally do NOT assert that access is True.
        #
        # Whether jdoe actually has VPN access is directory data.
        # This test verifies that the request successfully travels
        # through the real architecture and returns a result.
        # ========================================================

        assert worker.answer is not None
        assert result.answer is not None

        assert "jdoe" in result.answer

    finally:
        await mcp_runtime.stop()


def test_real_hub_access_request():
    asyncio.run(
        run_access_hub_e2e()
    )