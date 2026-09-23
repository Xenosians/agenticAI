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


from agent.approval_store import (
    ApprovalStore,
)

from agent.approvals import (
    ApprovalManager,
)

from agent.mcp_client import (
    MCPRuntime,
)

from config import (
    Settings,
)

from services.ticketing import (
    JiraTicketService,
    build_ticket_service,
)


PROJECT_KEY = (
    "KAN"
)

SUMMARY = (
    "Jira governed create proof"
)

TICKET_TYPE = (
    "Task"
)

EXPECTED_APPROVAL_ARGUMENTS = {
    "project_key":
        "KAN",

    "summary":
        "Jira governed create proof",

    "ticket_type":
        "Task",

    "expected_project_id":
        "10001",

    "expected_project_key":
        "KAN",

    "expected_project_name":
        "My Software Team",

    "expected_ticket_type_id":
        "10007",

    "expected_ticket_type_name":
        "Task",
}


async def main() -> int:

    if (
        len(
            sys.argv
        )
        != 2
    ):
        print(
            "Usage:"
        )

        print(
            "  python "
            "scripts/jira_ticket_create_execute.py "
            "<approval-id>"
        )

        return 2

    approval_id = (
        sys.argv[
            1
        ]
        .strip()
    )

    if not approval_id:
        print(
            "Approval ID must not be empty."
        )

        return 2

    settings = (
        Settings()
    )

    if (
        settings.ticketing_backend
        != "jira"
    ):
        print(
            "TICKETING_BACKEND must be jira."
        )

        return 2

    approval_db = (
        settings
        .resolve_runtime_path(
            Path(
                ".runtime/"
                "jira_ticket_create_approvals.sqlite3"
            )
        )
    )

    if not (
        approval_db.exists()
    ):
        print(
            "Approval DB does not exist:",
            approval_db,
        )

        return 2

    store = (
        ApprovalStore(
            approval_db
        )
    )

    store.initialize()

    approval = (
        store.get(
            approval_id
        )
    )

    if approval is None:
        print(
            "Approval not found:",
            approval_id,
        )

        return 2

    print(
        "=== STORED APPROVAL ==="
    )

    print(
        "id:",
        approval.approval_id,
    )

    print(
        "status:",
        approval.status,
    )

    print(
        "tool:",
        approval.tool,
    )

    print(
        "risk:",
        approval.risk,
    )

    print(
        "arguments:",
        approval.arguments,
    )

    # ========================================================
    # EXACT APPROVAL BINDING
    # ========================================================

    if (
        approval.status
        != "pending"
    ):
        print()
        print(
            "REFUSED:"
        )

        print(
            "Approval is not pending. "
            "No execution will occur."
        )

        return 2

    if (
        approval.tool
        != "ticket_create"
    ):
        print(
            "REFUSED: approval tool is not ticket_create."
        )

        return 2

    if (
        approval.risk
        != "medium"
    ):
        print(
            "REFUSED: approval risk is not medium."
        )

        return 2

    if (
        approval.arguments
        != EXPECTED_APPROVAL_ARGUMENTS
    ):
        print(
            "REFUSED:"
        )

        print(
            "Stored approval does not exactly match "
            "the JIRA-CREATE-PROPOSAL proven snapshot."
        )

        print()
        print(
            "expected:",
            EXPECTED_APPROVAL_ARGUMENTS,
        )

        print(
            "actual:",
            approval.arguments,
        )

        return 2

    print()
    print(
        "Exact persisted approval snapshot: PASS"
    )

    # ========================================================
    # START REAL MCP
    #
    # From here onward the exact approved mutation MAY execute.
    # ========================================================

    mcp = (
        MCPRuntime()
    )

    await mcp.start()

    try:
        manager = (
            ApprovalManager(
                store=(
                    store
                ),
                mcp=(
                    mcp
                ),
            )
        )

        print()
        print(
            "=== EXECUTING EXACT STORED APPROVAL ==="
        )

        result = (
            await manager
            .approve_approval(
                approval_id
            )
        )

        print()
        print(
            "manager ok:",
            result.get(
                "ok"
            ),
        )

        print(
            "replayed:",
            result.get(
                "replayed"
            ),
        )

        provider_result = (
            result.get(
                "result"
            )
        )

        print(
            "provider result:",
            provider_result,
        )

        persisted = (
            store.get(
                approval_id
            )
        )

        if persisted is None:
            raise RuntimeError(
                "Approval disappeared after execution."
            )

        print()
        print(
            "=== DURABLE APPROVAL AFTER EXECUTION ==="
        )

        print(
            "status:",
            persisted.status,
        )

        print(
            "result:",
            persisted.result,
        )

        # ====================================================
        # AMBIGUOUS OUTCOME
        #
        # CRITICAL:
        # DO NOT RETRY.
        # ====================================================

        if (
            persisted.status
            == "executing"
        ):
            print()
            print(
                "========================================"
            )

            print(
                "JIRA-CREATE-EXECUTE OUTCOME: UNRESOLVED"
            )

            print(
                "Approval remains EXECUTING."
            )

            print(
                "DO NOT RETRY THIS APPROVAL."
            )

            print(
                "Manual/trusted reconciliation is required."
            )

            print(
                "========================================"
            )

            return 3

        # ====================================================
        # KNOWN FAILURE
        # ====================================================

        if (
            persisted.status
            == "failed"
        ):
            print()
            print(
                "========================================"
            )

            print(
                "JIRA-CREATE-EXECUTE OUTCOME: KNOWN FAILURE"
            )

            print(
                "No successful mutation was accepted "
                "by the approval lifecycle."
            )

            print(
                "========================================"
            )

            return 1

        # ====================================================
        # VERIFIED SUCCESS REQUIRED
        # ====================================================

        if (
            persisted.status
            != "approved"
        ):
            raise RuntimeError(
                "Unexpected terminal approval status: "
                f"{persisted.status!r}"
            )

        if not isinstance(
            provider_result,
            dict,
        ):
            raise RuntimeError(
                "Approved execution did not return "
                "a structured provider result."
            )

        if (
            provider_result.get(
                "ok"
            )
            is not True
        ):
            raise RuntimeError(
                "Approval is approved but provider result "
                "does not report ok=True."
            )

        if (
            provider_result.get(
                "status"
            )
            != "success"
        ):
            raise RuntimeError(
                "Approval is approved but provider result "
                "does not report status=success."
            )

        if (
            provider_result.get(
                "mutation_performed"
            )
            is not True
        ):
            raise RuntimeError(
                "Provider did not confirm mutation_performed."
            )

        if (
            provider_result.get(
                "verification_ok"
            )
            is not True
        ):
            raise RuntimeError(
                "Provider did not confirm verification_ok."
            )

        ticket_key = (
            provider_result.get(
                "ticket_key"
            )
        )

        if not isinstance(
            ticket_key,
            str,
        ):
            raise RuntimeError(
                "Verified result did not return a ticket key."
            )

        # ====================================================
        # INDEPENDENT READ-ONLY POST-PROOF
        #
        # The mutation provider already performed trusted
        # read-back. This is an independent final proof.
        # ====================================================

        ticket_service = (
            build_ticket_service(
                settings
            )
        )

        if not isinstance(
            ticket_service,
            JiraTicketService,
        ):
            raise RuntimeError(
                "Configured provider is no longer Jira."
            )

        try:
            lookup = (
                ticket_service
                .get_ticket(
                    ticket_key
                )
            )

        finally:
            ticket_service.close()

        print()
        print(
            "=== INDEPENDENT JIRA READ-BACK ==="
        )

        print(
            "ok:",
            lookup.ok,
        )

        print(
            "status:",
            lookup.status,
        )

        print(
            "ticket:",
            (
                lookup.ticket
                .model_dump()
                if lookup.ticket
                is not None
                else None
            ),
        )

        if (
            lookup.ok
            is not True
            or lookup.ticket
            is None
        ):
            raise RuntimeError(
                "Independent Jira read-back failed."
            )

        ticket = (
            lookup.ticket
        )

        if (
            ticket.key
            != ticket_key
        ):
            raise RuntimeError(
                "Read-back ticket key mismatch."
            )

        if (
            ticket.summary
            != SUMMARY
        ):
            raise RuntimeError(
                "Read-back summary mismatch."
            )

        if (
            ticket.project_key
            != PROJECT_KEY
        ):
            raise RuntimeError(
                "Read-back project mismatch."
            )

        if (
            ticket.ticket_type
            != TICKET_TYPE
        ):
            raise RuntimeError(
                "Read-back ticket type mismatch."
            )

        print()
        print(
            "========================================"
        )

        print(
            "JIRA-CREATE-EXECUTE LIVE CREATE: PASS"
        )

        print(
            "approval:",
            approval_id,
        )

        print(
            "created ticket:",
            ticket_key,
        )

        print(
            "durable approval status:",
            persisted.status,
        )

        print(
            "mutation_performed: True"
        )

        print(
            "verification_ok: True"
        )

        print(
            "independent read-back: PASS"
        )

        print(
            "========================================"
        )

        return 0

    finally:
        await mcp.stop()


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main()
        )
    )
