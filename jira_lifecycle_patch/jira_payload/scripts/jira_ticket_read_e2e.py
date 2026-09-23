import asyncio
import sys

from pathlib import Path
from pprint import pprint


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from config import Settings

from subagents.integration.support import (
    build_hub_integration_context,
)


TICKET_KEY = "KAN-3"
PROJECT_KEY = "KAN"


CASES = [
    {
        "label":
            "EXACT ISSUE GET",

        "request":
            f"Get Jira ticket {TICKET_KEY}",

        "tool":
            "ticket_get",

        "arguments": {
            "ticket_key":
                TICKET_KEY,
        },
    },

    {
        "label":
            "PROJECT ISSUE SEARCH",

        "request":
            f"Search Jira tickets in project {PROJECT_KEY}",

        "tool":
            "ticket_search",

        "arguments": {
            "project_key":
                PROJECT_KEY,
        },
    },

    {
        "label":
            "ISSUE HISTORY",

        "request":
            f"Show the history of Jira ticket {TICKET_KEY}",

        "tool":
            "ticket_history",

        "arguments": {
            "ticket_key":
                TICKET_KEY,
        },
    },

    {
        "label":
            "ISSUE COMMENTS",

        "request":
            f"Show the comments on Jira ticket {TICKET_KEY}",

        "tool":
            "ticket_comments",

        "arguments": {
            "ticket_key":
                TICKET_KEY,
        },
    },
]


async def main() -> None:

    settings = Settings()

    if (
        settings.ticketing_backend
        != "jira"
    ):
        raise SystemExit(
            "TICKETING_BACKEND must be 'jira'."
        )

    approval_db = (
        settings.resolve_runtime_path(
            Path(
                ".runtime/"
                "jira_ticket_read_e2e_approvals.sqlite3"
            )
        )
    )

    runtime = (
        build_hub_integration_context(
            approval_db_path=(
                approval_db
            )
        )
    )

    await runtime.mcp.start()

    try:

        await (
            runtime
            .inference
            .warm(
                runtime
                .settings
                .hub_model_key
            )
        )

        for case in CASES:

            print()
            print(
                "========================================"
            )

            print(
                case[
                    "label"
                ]
            )

            print(
                "========================================"
            )

            print(
                "USER:",
                case[
                    "request"
                ],
            )

            result = (
                await runtime.hub.run(
                    case[
                        "request"
                    ]
                )
            )

            print()
            print(
                "hub status:",
                result.status,
            )

            print(
                "routes:",
                result.routes,
            )

            if (
                result.routes
                != [
                    "jira-specialist"
                ]
            ):
                raise AssertionError(
                    "Unexpected route: "
                    f"{result.routes!r}"
                )

            if (
                len(
                    result.results
                )
                != 1
            ):
                raise AssertionError(
                    "Expected exactly one specialist result."
                )

            worker = (
                result.results[
                    0
                ]
            )

            print(
                "worker:",
                worker.agent_name,
            )

            print(
                "worker status:",
                worker.status,
            )

            print(
                "tool:",
                worker.proposed_tool,
            )

            print(
                "arguments:",
                worker.proposed_arguments,
            )

            print(
                "outcome:",
                worker.outcome_code,
            )

            print(
                "tool result:"
            )

            pprint(
                worker.tool_result,
                sort_dicts=False,
            )

            if (
                worker.agent_name
                != "jira-specialist"
            ):
                raise AssertionError(
                    "Wrong specialist."
                )

            if (
                worker.proposed_tool
                != case[
                    "tool"
                ]
            ):
                raise AssertionError(
                    "Wrong capability: "
                    f"{worker.proposed_tool!r}"
                )

            proposed = (
                worker.proposed_arguments
                or {}
            )

            for (
                field,
                expected,
            ) in (
                case[
                    "arguments"
                ]
                .items()
            ):

                if (
                    proposed.get(
                        field
                    )
                    != expected
                ):
                    raise AssertionError(
                        f"{field} mismatch: "
                        f"{proposed.get(field)!r} "
                        f"!= {expected!r}"
                    )

            tool_result = (
                worker.tool_result
                or {}
            )

            if (
                tool_result.get(
                    "ok"
                )
                is not True
            ):
                raise AssertionError(
                    "Trusted Jira read failed: "
                    + repr(
                        tool_result
                    )
                )

            if (
                tool_result.get(
                    "status"
                )
                != "success"
            ):
                raise AssertionError(
                    "Unexpected trusted result status: "
                    + repr(
                        tool_result.get(
                            "status"
                        )
                    )
                )

            # -----------------------------------------------
            # Operation-specific provider truth
            # -----------------------------------------------

            if (
                case[
                    "tool"
                ]
                == "ticket_get"
            ):
                ticket = (
                    tool_result.get(
                        "ticket"
                    )
                    or {}
                )

                assert (
                    ticket.get(
                        "key"
                    )
                    == TICKET_KEY
                )

                assert (
                    ticket.get(
                        "provider"
                    )
                    == "jira"
                )

            elif (
                case[
                    "tool"
                ]
                == "ticket_search"
            ):
                tickets = (
                    tool_result.get(
                        "tickets"
                    )
                    or []
                )

                assert (
                    isinstance(
                        tickets,
                        list,
                    )
                )

                assert (
                    all(
                        ticket.get(
                            "provider"
                        )
                        == "jira"

                        for ticket
                        in tickets
                    )
                )

                assert (
                    all(
                        ticket.get(
                            "project_key"
                        )
                        == PROJECT_KEY

                        for ticket
                        in tickets
                    )
                )

            elif (
                case[
                    "tool"
                ]
                == "ticket_history"
            ):
                assert (
                    tool_result.get(
                        "ticket_key"
                    )
                    == TICKET_KEY
                )

                assert (
                    tool_result.get(
                        "provider"
                    )
                    == "jira"
                )

            elif (
                case[
                    "tool"
                ]
                == "ticket_comments"
            ):
                assert (
                    tool_result.get(
                        "ticket_key"
                    )
                    == TICKET_KEY
                )

                assert (
                    tool_result.get(
                        "provider"
                    )
                    == "jira"
                )

            print()
            print(
                case[
                    "label"
                ],
                "PASS",
            )

        print()
        print(
            "========================================"
        )

        print(
            "JIRA LIVE READ E2E: PASS"
        )

        print(
            "========================================"
        )

        print(
            "ticket_get: PASS"
        )

        print(
            "ticket_search: PASS"
        )

        print(
            "ticket_history: PASS"
        )

        print(
            "ticket_comments: PASS"
        )

        print(
            "No Jira mutation was executed."
        )

    finally:

        await runtime.mcp.stop()

        runtime.model_manager.unload_all()


if __name__ == "__main__":
    asyncio.run(
        main()
    )
