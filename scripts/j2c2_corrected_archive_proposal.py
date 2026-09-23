import asyncio
import sys

from pathlib import Path


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

from services.jira import (
    build_jira_project_mutation_service,
)

from subagents.integration.support import (
    build_hub_integration_context,
)


PROJECT_ID = "10034"
PROJECT_KEY = "J2C2V2"
PROJECT_NAME = "J2C2 Corrected Archive Proof"

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


def prove_live_state(
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

    if result.get("ok") is not True:
        raise AssertionError(
            "Could not prove project is live: "
            + str(result)
        )

    if result.get("status") != "ready":
        raise AssertionError(
            "Expected archive preparation "
            "status='ready'."
        )

    if result.get("risk") != "high":
        raise AssertionError(
            "Archive preparation is not HIGH risk."
        )

    args = (
        result.get(
            "execution_arguments"
        )
        or {}
    )

    assert (
        args.get(
            "project_id_or_key"
        )
        == PROJECT_KEY
    )

    assert (
        args.get(
            "expected_project_id"
        )
        == PROJECT_ID
    )

    assert (
        args.get(
            "expected_project_key"
        )
        == PROJECT_KEY
    )

    assert (
        args.get(
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
        args.get(
            "expected_project_id"
        ),
    )

    print(
        "project key:",
        args.get(
            "expected_project_key"
        ),
    )

    print(
        "project name:",
        repr(
            args.get(
                "expected_project_name"
            )
        ),
    )


async def main() -> None:

    settings = Settings()

    approval_db = (
        settings.resolve_runtime_path(
            Path(
                ".runtime/"
                "j2c2_corrected_archive_approvals.sqlite3"
            )
        )
    )

    # Dedicated DB for this proof.
    for suffix in (
        "",
        "-wal",
        "-shm",
    ):
        candidate = Path(
            str(approval_db)
            + suffix
        )

        if candidate.exists():
            candidate.unlink()

    jira = (
        build_jira_project_mutation_service(
            settings
        )
    )

    # ========================================================
    # PROVE THE FRESH FIXTURE IS LIVE
    # ========================================================

    prove_live_state(
        jira,
        label=(
            "BEFORE ARCHIVE PROPOSAL"
        ),
    )

    # ========================================================
    # REAL HUB + SPECIALIST + SEMANTIC GUARD + GATEWAY
    #
    # MCP cannot execute here.
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
            "=== ARCHIVE PROPOSAL ==="
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
                "Unexpected routes: "
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
        # DURABLE APPROVAL SNAPSHOT
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
            == "high"
        )

        assert (
            approval[
                "tool"
            ]
            == "jira_project_archive"
        )

        args = (
            approval[
                "arguments"
            ]
        )

        assert (
            args[
                "project_id_or_key"
            ]
            == PROJECT_KEY
        )

        assert (
            args[
                "expected_project_id"
            ]
            == PROJECT_ID
        )

        assert (
            args[
                "expected_project_key"
            ]
            == PROJECT_KEY
        )

        assert (
            args[
                "expected_project_name"
            ]
            == PROJECT_NAME
        )

        print()
        print(
            "=== PERSISTED ARCHIVE APPROVAL ==="
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
                args[
                    "project_id_or_key"
                ]
                == PROJECT_KEY
            ),
        )

        print(
            "trusted project ID exact:",
            (
                args[
                    "expected_project_id"
                ]
                == PROJECT_ID
            ),
        )

        print(
            "trusted project key exact:",
            (
                args[
                    "expected_project_key"
                ]
                == PROJECT_KEY
            ),
        )

        print(
            "trusted project name exact:",
            (
                args[
                    "expected_project_name"
                ]
                == PROJECT_NAME
            ),
        )

        # ====================================================
        # APPROVAL BARRIER
        # ====================================================

        prove_live_state(
            jira,
            label=(
                "AFTER PROPOSAL / BEFORE APPROVAL"
            ),
        )

        print()
        print(
            "J2C.2 CORRECTED APPROVAL BARRIER: PASS"
        )

        print(
            "Fresh Jira project remains live."
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
