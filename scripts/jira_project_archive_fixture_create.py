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
    build_jira_project_read_service,
)

from subagents.integration.support import (
    build_hub_integration_context,
)


PROJECT_KEY = "JIRAARC1"
PROJECT_NAME = "Jira Project Archive Proof"
TEMPLATE = "software-kanban"

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


async def main() -> None:

    settings = Settings()

    approval_db = (
        settings.resolve_runtime_path(
            Path(
                ".runtime/"
                "jira_project_archive_fixture_approvals.sqlite3"
            )
        )
    )

    # Dedicated DB for this disposable fixture proof.
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
        build_jira_project_read_service(
            settings
        )
    )

    # ========================================================
    # PROVE FRESH KEY IS ABSENT
    # ========================================================

    before = (
        jira.get_project(
            PROJECT_KEY
        )
    )

    print(
        "=== BEFORE CREATE PROPOSAL ==="
    )

    print(
        "project status:",
        before.get(
            "status"
        ),
    )

    if before.get("ok") is True:
        raise SystemExit(
            f"Project {PROJECT_KEY} already exists. "
            "Use a fresh disposable key."
        )

    if before.get("status") != "not_found":
        raise SystemExit(
            "Could not prove the fixture key is absent: "
            + str(before)
        )

    # ========================================================
    # REAL HUB / SEMANTIC / TOOLGATEWAY
    #
    # MCP execution is impossible at this checkpoint.
    # ========================================================

    runtime = (
        build_hub_integration_context(
            approval_db_path=approval_db,
            mcp=FailIfCalledMCP(),
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
            "=== CREATE PROPOSAL ==="
        )

        print(
            "status:",
            result.status,
        )

        print(
            "routes:",
            result.routes,
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
                "Unexpected routing: "
                f"{result.routes!r}"
            )

        if (
            len(result.results)
            != 1
        ):
            raise AssertionError(
                "Expected exactly one worker result."
            )

        worker = (
            result.results[0]
        )

        print(
            "worker:",
            worker.agent_name,
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
            approval["status"]
            == "pending"
        )

        assert (
            approval["risk"]
            == "medium"
        )

        assert (
            approval["tool"]
            == "jira_project_create"
        )

        args = (
            approval["arguments"]
        )

        assert (
            args["project_key"]
            == PROJECT_KEY
        )

        assert (
            args["project_name"]
            == PROJECT_NAME
        )

        assert (
            args["template"]
            == TEMPLATE
        )

        for field in (
            "expected_project_type_key",
            "expected_project_template_key",
            "expected_lead_account_id",
        ):
            assert (
                isinstance(
                    args.get(field),
                    str,
                )
                and args[field]
            )

        print()
        print(
            "=== PERSISTED CREATE APPROVAL ==="
        )

        print(
            "status:",
            approval["status"],
        )

        print(
            "risk:",
            approval["risk"],
        )

        print(
            "project key exact:",
            args["project_key"]
            == PROJECT_KEY,
        )

        print(
            "project name exact:",
            args["project_name"]
            == PROJECT_NAME,
        )

        print(
            "template exact:",
            args["template"]
            == TEMPLATE,
        )

        print(
            "trusted project type bound:",
            bool(
                args.get(
                    "expected_project_type_key"
                )
            ),
        )

        print(
            "trusted native template bound:",
            bool(
                args.get(
                    "expected_project_template_key"
                )
            ),
        )

        print(
            "trusted lead bound:",
            bool(
                args.get(
                    "expected_lead_account_id"
                )
            ),
        )

        # ====================================================
        # APPROVAL BARRIER
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
            after.get("ok")
            is False
        )

        assert (
            after.get("status")
            == "not_found"
        )

        print()
        print(
            "FRESH FIXTURE CREATE BARRIER: PASS"
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
