import asyncio

from agent.agent import run_agent
from agent.approvals import (
    approve_approval,
    list_pending_approvals,
)
from agent.mcp_client import mcp_runtime


async def main():
    print("Starting ITSM Agent...")

    await mcp_runtime.start()

    print()
    print("ITSM Agent")
    print("Commands:")
    print("  approvals")
    print("  approve <approval_id>")
    print("  exit")
    print()

    try:
        while True:
            user_input = input(
                "You > "
            ).strip()

            if not user_input:
                continue

            if user_input.lower() in {
                "exit",
                "quit",
            }:
                break

            if user_input.lower() in {
                "approval",
                "approvals",
            }:
                pending = (
                    list_pending_approvals()
                )

                if not pending:
                    print(
                        "\nNo pending approvals.\n"
                    )
                    continue

                print(
                    "\nPending approvals:"
                )

                for approval in pending:
                    print(
                        f"- {approval['id']} | "
                        f"{approval['tool']} | "
                        f"risk={approval['risk']} | "
                        f"args={approval['arguments']}"
                    )

                print()
                continue

            if user_input.lower().startswith(
                "approve "
            ):
                approval_id = (
                    user_input.split(
                        maxsplit=1
                    )[1].strip()
                )

                result = (
                    await approve_approval(
                        approval_id
                    )
                )

                print(
                    "\n[Approval Result]\n"
                    f"{result}\n"
                )

                continue

            try:
                response = await run_agent(
                    user_input
                )

            except Exception as exc:
                response = (
                    f"Agent error: {exc}"
                )

            print(
                f"\nAgent > {response}\n"
            )

    finally:
        await mcp_runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())