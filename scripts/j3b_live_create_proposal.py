from __future__ import annotations

import asyncio
import sys

from pathlib import (
    Path,
)


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


from config import (
    Settings,
)

from services.ticketing import (
    JiraTicketService,
    build_ticket_service,
)

from services.ticketing.jira_create_policy import (
    prepare_jira_ticket_create,
)

from subagents.integration.support import (
    build_hub_integration_context,
)


PROJECT_KEY = (
    "KAN"
)

SUMMARY = (
    "J3B governed create proof"
)

TICKET_TYPE = (
    "Task"
)

USER_REQUEST = (
    "Create a Jira ticket in project "
    f"{PROJECT_KEY} titled "
    f'"{SUMMARY}" with type '
    f"{TICKET_TYPE}"
)


class FailIfCalledMCP:

    async def call_tool(
        self,
        tool_name,
        arguments,
    ):
        raise AssertionError(
            "MCP execution must NOT occur before approval. "
            f"Attempted tool={tool_name!r} "
            f"arguments={arguments!r}"
        )


async def main() -> None:

    settings = (
        Settings()
    )

    if (
        settings.ticketing_backend
        != "jira"
    ):
        raise SystemExit(
            "TICKETING_BACKEND must be jira "
            "for J3B live proof."
        )

    approval_db = (
        settings
        .resolve_runtime_path(
            Path(
                ".runtime/"
                "j3b_create_approvals.sqlite3"
            )
        )
    )

    # Dedicated DB only for this proof.
    for suffix in (
        "",
        "-wal",
        "-shm",
    ):
        candidate = Path(
            str(
                approval_db
            )
            + suffix
        )

        if candidate.exists():
            candidate.unlink()

    # ========================================================
    # REAL JIRA / READ-ONLY PRE-PROPOSAL PROOF
    # ========================================================

    ticket_service = (
        build_ticket_service(
            settings
        )
    )

    if not isinstance(
        ticket_service,
        JiraTicketService,
    ):
        raise SystemExit(
            "Configured ticket provider is not Jira."
        )

    try:
        prepared = (
            prepare_jira_ticket_create(
                ticket_service,
                project_key=(
                    PROJECT_KEY
                ),
                summary=(
                    SUMMARY
                ),
                ticket_type=(
                    TICKET_TYPE
                ),
            )
        )

        print(
            "=== REAL JIRA CREATE PREPARATION ==="
        )

        print(
            "ok:",
            prepared.get(
                "ok"
            ),
        )

        print(
            "status:",
            prepared.get(
                "status"
            ),
        )

        print(
            "risk:",
            prepared.get(
                "risk"
            ),
        )

        print(
            "requires approval:",
            prepared.get(
                "requires_approval"
            ),
        )

        if (
            prepared.get(
                "ok"
            )
            is not True
        ):
            raise AssertionError(
                "Real Jira create preparation failed: "
                + str(
                    prepared
                )
            )

        if (
            prepared.get(
                "status"
            )
            != "ready"
        ):
            raise AssertionError(
                "Expected preparation status=ready."
            )

        if (
            prepared.get(
                "risk"
            )
            != "medium"
        ):
            raise AssertionError(
                "ticket_create must be MEDIUM risk."
            )

        if (
            prepared.get(
                "requires_approval"
            )
            is not True
        ):
            raise AssertionError(
                "ticket_create must require approval."
            )

        prepared_args = (
            prepared[
                "execution_arguments"
            ]
        )

        assert (
            prepared_args[
                "project_key"
            ]
            == PROJECT_KEY
        )

        assert (
            prepared_args[
                "summary"
            ]
            == SUMMARY
        )

        assert (
            prepared_args[
                "ticket_type"
            ]
            == TICKET_TYPE
        )

        assert isinstance(
            prepared_args.get(
                "expected_project_id"
            ),
            str,
        )

        assert (
            prepared_args.get(
                "expected_project_key"
            )
            == PROJECT_KEY
        )

        assert isinstance(
            prepared_args.get(
                "expected_project_name"
            ),
            str,
        )

        assert isinstance(
            prepared_args.get(
                "expected_ticket_type_id"
            ),
            str,
        )

        assert (
            prepared_args.get(
                "expected_ticket_type_name"
            )
            == TICKET_TYPE
        )

        print()
        print(
            "project id:",
            prepared_args[
                "expected_project_id"
            ],
        )

        print(
            "project key:",
            prepared_args[
                "expected_project_key"
            ],
        )

        print(
            "project name:",
            repr(
                prepared_args[
                    "expected_project_name"
                ]
            ),
        )

        print(
            "issue type id:",
            prepared_args[
                "expected_ticket_type_id"
            ],
        )

        print(
            "issue type name:",
            prepared_args[
                "expected_ticket_type_name"
            ],
        )

    finally:
        ticket_service.close()

    # ========================================================
    # REAL HUB + REAL MODEL + JIRA SPECIALIST +
    # SEMANTIC GUARD + TOOLGATEWAY + REAL POLICY
    #
    # MCP EXECUTION IS DELIBERATELY IMPOSSIBLE.
    # ========================================================

    runtime = (
        build_hub_integration_context(
            approval_db_path=(
                approval_db
            ),
            mcp=(
                FailIfCalledMCP()
            ),
        )
    )

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

        result = (
            await runtime
            .hub
            .run(
                USER_REQUEST
            )
        )

        print()
        print(
            "=== HUB CREATE PROPOSAL ==="
        )

        print(
            "request:",
            USER_REQUEST,
        )

        print(
            "status:",
            result.status,
        )

        print(
            "routes:",
            result.routes,
        )

        print(
            "workers:",
            len(
                result.results
            ),
        )

        if (
            result.status
            != "approval_required"
        ):
            raise AssertionError(
                "Expected approval_required, got "
                f"{result.status!r}"
            )

        if (
            result.routes
            != [
                "jira-specialist"
            ]
        ):
            raise AssertionError(
                "Expected jira-specialist only, got "
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

        print()
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
            "model arguments:",
            worker.proposed_arguments,
        )

        print(
            "approval id:",
            worker.approval_id,
        )

        assert (
            worker.agent_name
            == "jira-specialist"
        )

        assert (
            worker.status
            == "approval_required"
        )

        assert (
            worker.proposed_tool
            == "ticket_create"
        )

        assert (
            worker.proposed_arguments
            == {
                "project_key":
                    PROJECT_KEY,

                "summary":
                    SUMMARY,

                "ticket_type":
                    TICKET_TYPE,
            }
        )

        assert (
            worker.approval_id
            is not None
        )

        # ====================================================
        # DURABLE APPROVAL MUST CONTAIN EXACT MODEL ARGS
        # + EXACT TRUSTED PROVIDER SNAPSHOT
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

        assert (
            approval[
                "risk"
            ]
            == "medium"
        )

        assert (
            approval[
                "tool"
            ]
            == "ticket_create"
        )

        args = (
            approval[
                "arguments"
            ]
        )

        assert (
            args[
                "project_key"
            ]
            == PROJECT_KEY
        )

        assert (
            args[
                "summary"
            ]
            == SUMMARY
        )

        assert (
            args[
                "ticket_type"
            ]
            == TICKET_TYPE
        )

        assert isinstance(
            args.get(
                "expected_project_id"
            ),
            str,
        )

        assert (
            args.get(
                "expected_project_key"
            )
            == PROJECT_KEY
        )

        assert isinstance(
            args.get(
                "expected_project_name"
            ),
            str,
        )

        assert isinstance(
            args.get(
                "expected_ticket_type_id"
            ),
            str,
        )

        assert (
            args.get(
                "expected_ticket_type_name"
            )
            == TICKET_TYPE
        )

        # Trusted snapshot from Hub path must agree with the
        # independent read-only provider preparation above.
        assert (
            args[
                "expected_project_id"
            ]
            == prepared_args[
                "expected_project_id"
            ]
        )

        assert (
            args[
                "expected_project_key"
            ]
            == prepared_args[
                "expected_project_key"
            ]
        )

        assert (
            args[
                "expected_project_name"
            ]
            == prepared_args[
                "expected_project_name"
            ]
        )

        assert (
            args[
                "expected_ticket_type_id"
            ]
            == prepared_args[
                "expected_ticket_type_id"
            ]
        )

        assert (
            args[
                "expected_ticket_type_name"
            ]
            == prepared_args[
                "expected_ticket_type_name"
            ]
        )

        print()
        print(
            "=== PERSISTED APPROVAL ==="
        )

        print(
            "status:",
            approval[
                "status"
            ],
        )

        print(
            "risk:",
            approval[
                "risk"
            ],
        )

        print(
            "tool:",
            approval[
                "tool"
            ],
        )

        print(
            "project id:",
            args[
                "expected_project_id"
            ],
        )

        print(
            "project key:",
            args[
                "expected_project_key"
            ],
        )

        print(
            "project name:",
            repr(
                args[
                    "expected_project_name"
                ]
            ),
        )

        print(
            "issue type id:",
            args[
                "expected_ticket_type_id"
            ],
        )

        print(
            "issue type name:",
            args[
                "expected_ticket_type_name"
            ],
        )

        print()
        print(
            "========================================"
        )

        print(
            "J3B-5A APPROVAL BARRIER: PASS"
        )

        print(
            "NO Jira issue was created."
        )

        print(
            "========================================"
        )

        print()
        print(
            "Approval ID:",
            worker.approval_id,
        )

        print(
            "Approval DB:",
            approval_db,
        )

    finally:
        runtime.model_manager.unload_all()


if __name__ == "__main__":
    asyncio.run(
        main()
    )
