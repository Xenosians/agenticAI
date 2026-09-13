import asyncio

from pathlib import Path

from subagents.integration.support import (
    build_hub_integration_context,
)


async def run_access_hub_e2e(
    approval_db_path: Path,
):
    """
    Real Access vertical slice:

        User
          -> Hub inference
          -> access-specialist
          -> worker inference
          -> ToolGateway
          -> MCP
          -> directory provider
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
                "Does jdoe have VPN access?"
            )
        )

        print()

        print(
            "===== ACCESS HUB E2E ====="
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
            print("---")

            print(
                "worker: "
                f"{worker_result.agent_name}"
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
                "worker answer: "
                f"{worker_result.answer}"
            )

        print(
            "=========================="
        )

        assert (
            result.status
            == "success"
        )

        assert result.routes == [
            "access-specialist"
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
            == "access-specialist"
        )

        assert (
            worker.proposed_tool
            == "check_access"
        )

        assert (
            worker.proposed_arguments
            == {
                "user_id":
                    "jdoe",

                "resource":
                    "VPN",
            }
        )

        assert (
            worker.answer
            is not None
        )

        assert (
            result.answer
            is not None
        )

        assert (
            "jdoe"
            in result.answer
        )

    finally:
        await runtime.mcp.stop()


def test_real_hub_access_request(
    tmp_path: Path,
):
    asyncio.run(
        run_access_hub_e2e(
            tmp_path
            / "approvals.sqlite3"
        )
    )