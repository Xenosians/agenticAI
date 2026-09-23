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


PROJECT_ID = "10035"
PROJECT_KEY = "J2DDEL1"
PROJECT_NAME = "J2D Delete Proof"

USER_REQUEST = (
    f"Delete Jira project {PROJECT_KEY}"
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


def prove_delete_ready(
    service,
    *,
    label: str,
) -> None:

    result = (
        service.prepare_delete_project(
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

    print(
        "requires approval:",
        result.get(
            "requires_approval"
        ),
    )

    if result.get("ok") is not True:
        raise AssertionError(
            "Could not prove project is eligible "
            "for governed deletion: "
            + str(result)
        )

    if result.get("status") != "ready":
        raise AssertionError(
            "Expected delete preparation "
            "status='ready'."
        )

    if result.get("risk") != "high":
        raise AssertionError(
            "Delete preparation is not HIGH risk."
        )

    if (
        result.get(
            "requires_approval"
        )
        is not True
    ):
        raise AssertionError(
            "Delete preparation does not require approval."
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

    print(
        "trusted live-delete precondition:",
        "PASS",
    )


async def main() -> None:

    settings = Settings()

    approval_db = (
        settings.resolve_runtime_path(
            Path(
                ".runtime/"
                "j2d_delete_approvals.sqlite3"
            )
        )
    )

    # Dedicated durable approval DB for this live proof.
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
    # READ-ONLY PRE-PROPOSAL PROVIDER PROOF
    # ========================================================

    prove_delete_ready(
        jira,
        label=(
            "BEFORE DELETE PROPOSAL"
        ),
    )

    # ========================================================
    # REAL HUB + SPECIALIST + SEMANTIC GUARD + TOOLGATEWAY
    #
    # MCP execution is deliberately impossible here.
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
            "=== DELETE PROPOSAL ==="
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
            == "jira_project_delete"
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
        # EXACT DURABLE APPROVAL SNAPSHOT
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
            == "jira_project_delete"
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

        # Critical boundary:
        # enableUndo is NOT persisted as a model argument.
        # It belongs entirely to trusted provider execution.

        assert (
            "enableUndo"
            not in args
        )

        assert (
            "enable_undo"
            not in args
        )

        print()
        print(
            "=== PERSISTED DELETE APPROVAL ==="
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

        print(
            "model/provider undo flag exposed:",
            (
                "enableUndo"
                in args
                or "enable_undo"
                in args
            ),
        )

        # ====================================================
        # APPROVAL BARRIER:
        #
        # prove the provider project is STILL live and eligible.
        # FailIfCalledMCP proves no DELETE could have happened.
        # ====================================================

        prove_delete_ready(
            jira,
            label=(
                "AFTER PROPOSAL / BEFORE APPROVAL"
            ),
        )

        print()
        print(
            "======================================"
        )

        print(
            "J2D HIGH-RISK APPROVAL BARRIER: PASS"
        )

        print(
            "======================================"
        )

        print(
            "Fresh Jira project remains live."
        )

        print(
            "No DELETE request was executed."
        )

        print(
            "Provider undo behavior remains trusted-side."
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
