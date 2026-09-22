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
    "J2BTST"
)

OLD_NAME = (
    "J2B Create Preflight"
)

NEW_NAME = (
    "J2C Rename Proof"
)

USER_REQUEST = (
    f'Rename Jira project {PROJECT_KEY} '
    f'to "{NEW_NAME}"'
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
                "j2c1_live_approvals.sqlite3"
            )
        )
    )

    # Dedicated DB for this live proof only.
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

    jira = (
        build_jira_project_read_service(
            settings
        )
    )

    # ========================================================
    # PROVE EXACT PRE-MUTATION PROVIDER STATE
    # ========================================================

    before = (
        jira.get_project(
            PROJECT_KEY
        )
    )

    print(
        "=== BEFORE PROPOSAL ==="
    )

    print(
        "status:",
        before.get(
            "status"
        ),
    )

    if (
        before.get(
            "ok"
        )
        is not True
    ):
        raise SystemExit(
            "Could not read the Jira project before proposal: "
            + str(
                before
            )
        )

    before_project = (
        before.get(
            "project"
        )
        or {}
    )

    print(
        "project id:",
        before_project.get(
            "id"
        ),
    )

    print(
        "project key:",
        before_project.get(
            "key"
        ),
    )

    print(
        "project name:",
        repr(
            before_project.get(
                "name"
            )
        ),
    )

    if (
        before_project.get(
            "key"
        )
        != PROJECT_KEY
    ):
        raise SystemExit(
            "Unexpected Jira project key."
        )

    if (
        before_project.get(
            "name"
        )
        != OLD_NAME
    ):
        raise SystemExit(
            "Project is not in the expected "
            "pre-rename state."
        )

    project_id = (
        before_project.get(
            "id"
        )
    )

    if (
        not isinstance(
            project_id,
            str,
        )
        or not project_id
    ):
        raise SystemExit(
            "Project ID is unavailable."
        )

    # ========================================================
    # REAL HUB + REAL MODELS + REAL TRUSTED POLICY
    #
    # MCP execution is impossible at this checkpoint.
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
            == "jira_project_update"
        )

        assert (
            worker.proposed_arguments
            == {
                "project_id_or_key":
                    PROJECT_KEY,

                "new_name":
                    NEW_NAME,
            }
        )

        assert (
            worker.approval_id
            is not None
        )

        # ====================================================
        # INSPECT EXACT DURABLE APPROVAL SNAPSHOT
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
            == "jira_project_update"
        )

        assert (
            approval[
                "risk"
            ]
            == "medium"
        )

        arguments = (
            approval[
                "arguments"
            ]
        )

        # ----------------------------------------------------
        # User-controlled arguments remain exact.
        # ----------------------------------------------------

        assert (
            arguments[
                "project_id_or_key"
            ]
            == PROJECT_KEY
        )

        assert (
            arguments[
                "new_name"
            ]
            == NEW_NAME
        )

        # ----------------------------------------------------
        # Trusted snapshot must match the provider state we
        # observed before approval creation.
        # ----------------------------------------------------

        assert (
            arguments[
                "expected_project_id"
            ]
            == project_id
        )

        assert (
            arguments[
                "expected_project_key"
            ]
            == PROJECT_KEY
        )

        assert (
            arguments[
                "expected_project_name"
            ]
            == OLD_NAME
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
            "project reference exact:",
            (
                arguments[
                    "project_id_or_key"
                ]
                == PROJECT_KEY
            ),
        )

        print(
            "new name exact:",
            (
                arguments[
                    "new_name"
                ]
                == NEW_NAME
            ),
        )

        print(
            "trusted project ID exact:",
            (
                arguments[
                    "expected_project_id"
                ]
                == project_id
            ),
        )

        print(
            "trusted project key exact:",
            (
                arguments[
                    "expected_project_key"
                ]
                == PROJECT_KEY
            ),
        )

        print(
            "trusted old name exact:",
            (
                arguments[
                    "expected_project_name"
                ]
                == OLD_NAME
            ),
        )

        # ====================================================
        # CRITICAL APPROVAL BARRIER PROOF
        #
        # Jira must STILL show the old project name.
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
            "status:",
            after.get(
                "status"
            ),
        )

        if (
            after.get(
                "ok"
            )
            is not True
        ):
            raise AssertionError(
                "Could not read Jira after proposal."
            )

        after_project = (
            after.get(
                "project"
            )
            or {}
        )

        print(
            "project id:",
            after_project.get(
                "id"
            ),
        )

        print(
            "project key:",
            after_project.get(
                "key"
            ),
        )

        print(
            "project name:",
            repr(
                after_project.get(
                    "name"
                )
            ),
        )

        assert (
            after_project.get(
                "id"
            )
            == project_id
        )

        assert (
            after_project.get(
                "key"
            )
            == PROJECT_KEY
        )

        assert (
            after_project.get(
                "name"
            )
            == OLD_NAME
        )

        print()
        print(
            "J2C.1 APPROVAL BARRIER: PASS"
        )

        print(
            "Jira project name remains unchanged."
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
