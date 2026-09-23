import asyncio

from pathlib import (
    Path,
)

from config import (
    Settings,
)

from services.jira import (
    build_jira_project_read_service,
)

from subagents.integration.support import (
    build_hub_integration_context,
)


PROJECT_KEY = (
    "JIRATST"
)

PROJECT_NAME = (
    "Jira Project Create Proof"
)

TEMPLATE = (
    "software-kanban"
)

USER_REQUEST = (
    "Create a Jira project with key "
    f"{PROJECT_KEY} named "
    f'"{PROJECT_NAME}" using template '
    f"{TEMPLATE}"
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


async def main(
) -> None:

    settings = (
        Settings()
    )

    approval_db = (
        settings.resolve_runtime_path(
            Path(
                ".runtime/"
                "jira_project_create_approvals.sqlite3"
            )
        )
    )

    # --------------------------------------------------------
    # This database is dedicated only to this JIRA CREATE live proof.
    # --------------------------------------------------------

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
    # PROVE TARGET DOES NOT ALREADY EXIST
    # ========================================================

    jira = (
        build_jira_project_read_service(
            settings
        )
    )

    before = (
        jira.get_project(
            PROJECT_KEY
        )
    )

    print(
        "=== BEFORE PROPOSAL ==="
    )

    print(
        "project status:",
        before.get(
            "status"
        ),
    )

    if (
        before.get(
            "ok"
        )
        is True
    ):
        raise SystemExit(
            f"Project {PROJECT_KEY} already exists. "
            "Choose a fresh key before continuing."
        )

    if (
        before.get(
            "status"
        )
        != "not_found"
    ):
        raise SystemExit(
            "Could not prove the target project is absent: "
            + str(
                before
            )
        )

    # ========================================================
    # REAL HUB + REAL MODELS + REAL TRUSTED POLICY
    #
    # MCP execution is deliberately impossible in this phase.
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
            await runtime.hub.run(
                USER_REQUEST
            )
        )

        print()
        print(
            "=== HUB PROPOSAL ==="
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
                "Expected only jira-specialist route, got "
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
            == "jira_project_create"
        )

        assert (
            worker.proposed_arguments
            == {
                "project_key":
                    PROJECT_KEY,

                "project_name":
                    PROJECT_NAME,

                "template":
                    TEMPLATE,
            }
        )

        assert (
            worker.approval_id
            is not None
        )

        # ====================================================
        # DURABLE APPROVAL MUST CONTAIN ORIGINAL + TRUSTED
        # SNAPSHOT ARGUMENTS.
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
                "tool"
            ]
            == "jira_project_create"
        )

        arguments = (
            approval[
                "arguments"
            ]
        )

        assert (
            arguments[
                "project_key"
            ]
            == PROJECT_KEY
        )

        assert (
            arguments[
                "project_name"
            ]
            == PROJECT_NAME
        )

        assert (
            arguments[
                "template"
            ]
            == TEMPLATE
        )

        assert (
            isinstance(
                arguments.get(
                    "expected_project_type_key"
                ),
                str,
            )
        )

        assert (
            isinstance(
                arguments.get(
                    "expected_project_template_key"
                ),
                str,
            )
        )

        assert (
            isinstance(
                arguments.get(
                    "expected_lead_account_id"
                ),
                str,
            )
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
            "original project key preserved:",
            (
                arguments[
                    "project_key"
                ]
                == PROJECT_KEY
            ),
        )

        print(
            "original project name preserved:",
            (
                arguments[
                    "project_name"
                ]
                == PROJECT_NAME
            ),
        )

        print(
            "template alias preserved:",
            (
                arguments[
                    "template"
                ]
                == TEMPLATE
            ),
        )

        print(
            "trusted project type bound:",
            bool(
                arguments.get(
                    "expected_project_type_key"
                )
            ),
        )

        print(
            "trusted native template bound:",
            bool(
                arguments.get(
                    "expected_project_template_key"
                )
            ),
        )

        print(
            "trusted lead identity bound:",
            bool(
                arguments.get(
                    "expected_lead_account_id"
                )
            ),
        )

        # ====================================================
        # CRITICAL PROOF:
        #
        # Project must STILL NOT EXIST after approval proposal.
        # ====================================================

        after = (
            jira.get_project(
                PROJECT_KEY
            )
        )

        print()
        print(
            "=== AFTER PROPOSAL / BEFORE APPROVAL ==="
        )

        print(
            "project status:",
            after.get(
                "status"
            ),
        )

        assert (
            after.get(
                "ok"
            )
            is False
        )

        assert (
            after.get(
                "status"
            )
            == "not_found"
        )

        print()
        print(
            "JIRA CREATE APPROVAL BARRIER: PASS"
        )

        print(
            "No Jira project was created."
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
