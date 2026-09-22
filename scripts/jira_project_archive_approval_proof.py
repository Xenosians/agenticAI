import asyncio

from pathlib import (
    Path,
)

from config import (
    Settings,
)

from services.jira import (
    build_jira_project_mutation_service,
)

from subagents.integration.support import (
    build_hub_integration_context,
)


PROJECT_ID = "10033"
PROJECT_KEY = "J2BTST"
PROJECT_NAME = "J2C Rename Proof"

USER_REQUEST = (
    f"Archive Jira project {PROJECT_KEY}"
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


def assert_project_still_live(
    service,
    *,
    label: str,
) -> None:

    result = (
        service.prepare_archive_project(
            project_id_or_key=(
                PROJECT_KEY
            )
        )
    )

    print()
    print(
        f"=== {label} ==="
    )

    print(
        "ok:",
        result.get(
            "ok"
        ),
    )

    print(
        "status:",
        result.get(
            "status"
        ),
    )

    print(
        "risk:",
        result.get(
            "risk"
        ),
    )

    if (
        result.get(
            "ok"
        )
        is not True
    ):
        raise AssertionError(
            "Could not prove Jira project is still live: "
            + str(
                result
            )
        )

    if (
        result.get(
            "status"
        )
        != "ready"
    ):
        raise AssertionError(
            "Expected trusted archive preparation "
            "status='ready'."
        )

    arguments = (
        result.get(
            "execution_arguments"
        )
        or {}
    )

    assert (
        arguments.get(
            "project_id_or_key"
        )
        == PROJECT_KEY
    )

    assert (
        arguments.get(
            "expected_project_id"
        )
        == PROJECT_ID
    )

    assert (
        arguments.get(
            "expected_project_key"
        )
        == PROJECT_KEY
    )

    assert (
        arguments.get(
            "expected_project_name"
        )
        == PROJECT_NAME
    )

    print(
        "trusted lifecycle state:",
        "live",
    )

    print(
        "project id:",
        arguments.get(
            "expected_project_id"
        ),
    )

    print(
        "project key:",
        arguments.get(
            "expected_project_key"
        ),
    )

    print(
        "project name:",
        repr(
            arguments.get(
                "expected_project_name"
            )
        ),
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
                "j2c2_live_approvals.sqlite3"
            )
        )
    )

    # --------------------------------------------------------
    # Dedicated approval database for this proof.
    # --------------------------------------------------------

    for suffix in (
        "",
        "-wal",
        "-shm",
    ):
        candidate = (
            Path(
                str(
                    approval_db
                )
                + suffix
            )
        )

        if candidate.exists():
            candidate.unlink()

    jira_mutations = (
        build_jira_project_mutation_service(
            settings
        )
    )

    # ========================================================
    # PROVE LIVE STATE BEFORE THE MODEL PIPELINE
    # ========================================================

    assert_project_still_live(
        jira_mutations,
        label=(
            "BEFORE PROPOSAL"
        ),
    )

    # ========================================================
    # REAL HUB + REAL MODEL + REAL TRUSTED POLICY
    #
    # MCP is intentionally replaced with a guard object.
    # Any premature execution makes this proof fail.
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

        # ====================================================
        # MODEL / SEMANTIC CONTRACT
        # ====================================================

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
            == "jira_project_archive"
        )

        assert (
            worker.proposed_arguments
            == {
                "project_id_or_key":
                    PROJECT_KEY,
            }
        )

        assert (
            worker.approval_id
            is not None
        )

        # ====================================================
        # DURABLE HIGH-RISK APPROVAL
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
            == "jira_project_archive"
        )

        assert (
            approval[
                "risk"
            ]
            == "high"
        )

        arguments = (
            approval[
                "arguments"
            ]
        )

        # ----------------------------------------------------
        # Original model/user-controlled value remains exact.
        # ----------------------------------------------------

        assert (
            arguments[
                "project_id_or_key"
            ]
            == PROJECT_KEY
        )

        # ----------------------------------------------------
        # Trusted provider state is frozen into the approval.
        # ----------------------------------------------------

        assert (
            arguments[
                "expected_project_id"
            ]
            == PROJECT_ID
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
            == PROJECT_NAME
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
            "trusted project ID exact:",
            (
                arguments[
                    "expected_project_id"
                ]
                == PROJECT_ID
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
            "trusted project name exact:",
            (
                arguments[
                    "expected_project_name"
                ]
                == PROJECT_NAME
            ),
        )

        # ====================================================
        # CRITICAL APPROVAL BARRIER
        #
        # Project must remain LIVE after proposal creation.
        # ====================================================

        assert_project_still_live(
            jira_mutations,
            label=(
                "AFTER PROPOSAL / BEFORE APPROVAL"
            ),
        )

        print()
        print(
            "J2C.2 APPROVAL BARRIER: PASS"
        )

        print(
            "Jira project remains live and unarchived."
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
