import asyncio

from agent.mcp_client import mcp_runtime

from subagents.hub import build_hub


async def run_hub_e2e():
    await mcp_runtime.start()

    try:
        hub = build_hub()

        result = await hub.run(
            "Is jdoe locked?"
        )

        print()
        print("===== HUB E2E =====")
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

        print("===================")

        assert result.status == "success"

        assert result.routes == [
            "account-specialist"
        ]

        assert len(result.results) == 1

        worker = result.results[0]

        assert (
            worker.agent_name
            == "account-specialist"
        )

        assert (
            worker.proposed_tool
            == "account_status"
        )

        assert worker.proposed_arguments == {
            "user_id": "jdoe"
        }

        assert "jdoe" in result.answer

    finally:
        await mcp_runtime.stop()


def test_real_hub_account_request():
    asyncio.run(
        run_hub_e2e()
    )