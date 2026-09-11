import asyncio
import uuid

from pathlib import Path

from config import (
    get_settings,
)

from .approval_store import (
    ApprovalEntry,
    ApprovalStore,
)

from .mcp_client import (
    mcp_runtime,
)

from tools.registry import (
    get_tool,
)


APPROVAL_LOCKS: dict[
    str,
    asyncio.Lock,
] = {}


_APPROVAL_STORE: ApprovalStore | None = None


def approval_store_path() -> Path:
    settings = get_settings()

    return settings.resolve_runtime_path(
        settings.approval_store_path
    )


def approval_store() -> ApprovalStore:
    global _APPROVAL_STORE

    if _APPROVAL_STORE is None:
        store = ApprovalStore(
            approval_store_path()
        )

        store.initialize()

        _APPROVAL_STORE = store

    return _APPROVAL_STORE


def approval_to_dict(
    approval: ApprovalEntry,
) -> dict:
    """
    Preserve the approval dictionary contract used by
    ToolGateway, the API, and the CLI.
    """

    return {
        "id":
            approval.approval_id,

        "tool":
            approval.tool,

        "arguments":
            approval.arguments,

        "risk":
            approval.risk,

        "status":
            approval.status,

        "result":
            approval.result,
    }


def create_approval(
    tool_name: str,
    arguments: dict,
    risk: str | None = None,
) -> dict:
    """
    Create a durable approval record.

    The tool registry remains the conservative fallback.

    A trusted deterministic policy resolver may supply a more
    specific risk classification for a concrete action.
    """

    tool = get_tool(
        tool_name
    )

    if tool is None:
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    if not tool[
        "requires_approval"
    ]:
        raise ValueError(
            f"Tool '{tool_name}' "
            "does not support approval."
        )

    effective_risk = (
        risk
        if risk is not None
        else tool["risk"]
    )

    if not isinstance(
        effective_risk,
        str,
    ):
        raise ValueError(
            "Approval risk must be a string."
        )

    approval_id = (
        uuid.uuid4()
        .hex[:8]
    )

    approval = (
        approval_store().create(
            approval_id=approval_id,
            tool=tool_name,
            arguments=arguments,
            risk=effective_risk,
        )
    )

    return approval_to_dict(
        approval
    )


def get_approval(
    approval_id: str,
) -> dict | None:
    approval = (
        approval_store().get(
            approval_id
        )
    )

    if approval is None:
        return None

    return approval_to_dict(
        approval
    )


def list_pending_approvals() -> list[dict]:
    return [
        approval_to_dict(
            approval
        )

        for approval
        in approval_store().list_pending()
    ]


def execution_unresolved_result(
    approval: ApprovalEntry,
) -> dict:
    return {
        "ok": False,

        "approval":
            approval_to_dict(
                approval
            ),

        "replayed": True,

        "error": (
            f"Approval '{approval.approval_id}' "
            "is already executing or was interrupted "
            "during execution. Its side-effect outcome "
            "must be reconciled before retrying."
        ),
    }


async def approve_approval(
    approval_id: str,
) -> dict:
    store = approval_store()

    approval = store.get(
        approval_id
    )

    if approval is None:
        return {
            "ok": False,

            "error": (
                f"Approval '{approval_id}' "
                "not found."
            ),
        }

    lock = APPROVAL_LOCKS.setdefault(
        approval_id,
        asyncio.Lock(),
    )

    async with lock:
        approval = store.get(
            approval_id
        )

        if approval is None:
            return {
                "ok": False,

                "error": (
                    f"Approval '{approval_id}' "
                    "not found."
                ),
            }

        # ----------------------------------------------------
        # Durable idempotent replay
        # ----------------------------------------------------

        if (
            approval.status
            == "approved"
        ):
            return {
                "ok": True,

                "approval":
                    approval_to_dict(
                        approval
                    ),

                "result":
                    approval.result,

                "replayed": True,
            }

        if (
            approval.status
            == "failed"
        ):
            return {
                "ok": False,

                "approval":
                    approval_to_dict(
                        approval
                    ),

                "result":
                    approval.result,

                "replayed": True,

                "error": (
                    f"Approval '{approval_id}' "
                    "previously failed."
                ),
            }

        # ----------------------------------------------------
        # Crash-safe ambiguous state
        #
        # Never automatically retry an approval left in
        # executing state. The OS mutation may already have
        # happened before the previous process stopped.
        # ----------------------------------------------------

        if (
            approval.status
            == "executing"
        ):
            return execution_unresolved_result(
                approval
            )

        if (
            approval.status
            != "pending"
        ):
            return {
                "ok": False,

                "approval":
                    approval_to_dict(
                        approval
                    ),

                "error": (
                    f"Approval '{approval_id}' "
                    "has invalid status "
                    f"'{approval.status}'."
                ),
            }

        # ----------------------------------------------------
        # Atomic durable execution claim
        #
        # SQLite performs:
        #
        #   pending -> executing
        #
        # before MCP is allowed to touch the host.
        # ----------------------------------------------------

        claimed = (
            store.claim_for_execution(
                approval_id
            )
        )

        if not claimed:
            approval = store.get(
                approval_id
            )

            if approval is None:
                return {
                    "ok": False,

                    "error": (
                        f"Approval '{approval_id}' "
                        "not found after claim."
                    ),
                }

            if (
                approval.status
                == "approved"
            ):
                return {
                    "ok": True,

                    "approval":
                        approval_to_dict(
                            approval
                        ),

                    "result":
                        approval.result,

                    "replayed": True,
                }

            if (
                approval.status
                == "failed"
            ):
                return {
                    "ok": False,

                    "approval":
                        approval_to_dict(
                            approval
                        ),

                    "result":
                        approval.result,

                    "replayed": True,

                    "error": (
                        f"Approval '{approval_id}' "
                        "previously failed."
                    ),
                }

            if (
                approval.status
                == "executing"
            ):
                return execution_unresolved_result(
                    approval
                )

            return {
                "ok": False,

                "approval":
                    approval_to_dict(
                        approval
                    ),

                "error": (
                    f"Approval '{approval_id}' "
                    "could not be claimed."
                ),
            }

        approval = store.get(
            approval_id
        )

        if approval is None:
            return {
                "ok": False,

                "error": (
                    f"Approval '{approval_id}' "
                    "disappeared after execution claim."
                ),
            }

        print(
            "\n[MCP Mutation] "
            f"{approval.tool} "
            f"{approval.arguments}"
        )

        # ----------------------------------------------------
        # Execute the exact persisted action.
        # ----------------------------------------------------

        try:
            result = (
                await mcp_runtime.call_tool(
                    approval.tool,
                    approval.arguments,
                )
            )

        except Exception as exc:
            # Do NOT move back to pending.
            #
            # The tool boundary threw while execution was in
            # progress. The mutation may already have happened.
            # Keep the durable "executing" state so restart or
            # retry cannot execute it again automatically.

            current = store.get(
                approval_id
            )

            return {
                "ok": False,

                "approval": (
                    approval_to_dict(
                        current
                    )
                    if current is not None
                    else None
                ),

                "replayed": False,

                "error": (
                    "Approval execution outcome could "
                    "not be confirmed. The approval "
                    "remains in executing state and "
                    "requires reconciliation. "
                    f"Error: {exc}"
                ),
            }

        if not isinstance(
            result,
            dict,
        ):
            current = store.get(
                approval_id
            )

            return {
                "ok": False,

                "approval": (
                    approval_to_dict(
                        current
                    )
                    if current is not None
                    else None
                ),

                "replayed": False,

                "error": (
                    "Approval execution returned an "
                    "invalid result. The approval "
                    "remains in executing state because "
                    "the side-effect outcome is unknown."
                ),
            }

        successful_statuses = {
            "executed",
            "success",
        }

        execution_succeeded = (
            result.get(
                "ok"
            ) is True
            and result.get(
                "status"
            ) in successful_statuses
        )

        # ----------------------------------------------------
        # Persist terminal result before responding.
        # ----------------------------------------------------

        try:
            if execution_succeeded:
                approval = (
                    store.mark_approved(
                        approval_id,
                        result,
                    )
                )

            else:
                approval = (
                    store.mark_failed(
                        approval_id,
                        result,
                    )
                )

        except Exception as exc:
            # Execution has already returned from MCP but the
            # terminal result could not be durably committed.
            #
            # Keep "executing". Retrying the mutation would be
            # unsafe.

            current = store.get(
                approval_id
            )

            return {
                "ok": False,

                "approval": (
                    approval_to_dict(
                        current
                    )
                    if current is not None
                    else None
                ),

                "result":
                    result,

                "replayed": False,

                "error": (
                    "Approval execution finished but "
                    "its terminal state could not be "
                    "persisted. Automatic retry is "
                    "disabled. "
                    f"Error: {exc}"
                ),
            }

        return {
            "ok":
                execution_succeeded,

            "approval":
                approval_to_dict(
                    approval
                ),

            "result":
                result,

            "replayed":
                False,
        }