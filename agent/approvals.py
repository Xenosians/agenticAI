import asyncio
import uuid

from typing import (
    Callable,
)

from agent.approval_store import (
    ApprovalEntry,
    ApprovalStore,
)

from agent.mcp_client import (
    MCPRuntime,
)

from tools.registry import (
    get_tool,
)


def approval_to_dict(
    approval: ApprovalEntry,
) -> dict:
    """
    Preserve the approval dictionary contract used by
    ToolGateway, FastAPI, and integration tests.
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


def execution_unresolved_result(
    approval: ApprovalEntry,
) -> dict:
    """
    Fail closed for an approval whose side-effect outcome is
    ambiguous.

    An executing approval is never automatically retried.
    """

    return {
        "ok":
            False,

        "approval":
            approval_to_dict(
                approval
            ),

        "result":
            approval.result,

        "replayed":
            True,

        "error": (
            f"Approval '{approval.approval_id}' "
            "is already executing or was interrupted "
            "during execution. Its side-effect outcome "
            "must be reconciled before retrying."
        ),
    }


class ApprovalManager:
    """
    Explicit lifecycle owner for durable approval execution.

    One application runtime normally owns one ApprovalManager.

    The manager owns:
    - the durable ApprovalStore
    - per-approval concurrency locks
    - the MCP execution dependency

    It does not use module-global mutable state.
    """

    def __init__(
        self,
        *,
        store: ApprovalStore,
        mcp: MCPRuntime,
        tool_lookup: Callable = (
            get_tool
        ),
    ) -> None:
        self.store = (
            store
        )

        self.mcp = (
            mcp
        )

        self.tool_lookup = (
            tool_lookup
        )

        self._locks: dict[
            str,
            asyncio.Lock,
        ] = {}

    def create_approval(
        self,
        tool_name: str,
        arguments: dict,
        risk: str | None = None,
    ) -> dict:
        """
        Create a durable approval proposal.

        The trusted tool registry remains the conservative
        policy fallback.
        """

        tool = (
            self.tool_lookup(
                tool_name
            )
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
            else tool[
                "risk"
            ]
        )

        if not isinstance(
            effective_risk,
            str,
        ):
            raise ValueError(
                "Approval risk must "
                "be a string."
            )

        approval_id = (
            uuid.uuid4()
            .hex[:8]
        )

        approval = (
            self.store.create(
                approval_id=(
                    approval_id
                ),
                tool=(
                    tool_name
                ),
                arguments=(
                    arguments
                ),
                risk=(
                    effective_risk
                ),
            )
        )

        return approval_to_dict(
            approval
        )

    def get_approval(
        self,
        approval_id: str,
    ) -> dict | None:
        approval = (
            self.store.get(
                approval_id
            )
        )

        if approval is None:
            return None

        return approval_to_dict(
            approval
        )

    def list_pending_approvals(
        self,
    ) -> list[dict]:
        return [
            approval_to_dict(
                approval
            )

            for approval
            in self.store.list_pending()
        ]

    async def approve_approval(
        self,
        approval_id: str,
    ) -> dict:
        approval = (
            self.store.get(
                approval_id
            )
        )

        if approval is None:
            return {
                "ok":
                    False,

                "error": (
                    f"Approval '{approval_id}' "
                    "not found."
                ),
            }

        lock = (
            self._locks.setdefault(
                approval_id,
                asyncio.Lock(),
            )
        )

        async with lock:
            return (
                await
                self._approve_locked(
                    approval_id
                )
            )

    async def _approve_locked(
        self,
        approval_id: str,
    ) -> dict:
        approval = (
            self.store.get(
                approval_id
            )
        )

        if approval is None:
            return {
                "ok":
                    False,

                "error": (
                    f"Approval '{approval_id}' "
                    "not found."
                ),
            }

        # ========================================================
        # DURABLE IDEMPOTENT REPLAY
        # ========================================================

        if (
            approval.status
            == "approved"
        ):
            return {
                "ok":
                    True,

                "approval":
                    approval_to_dict(
                        approval
                    ),

                "result":
                    approval.result,

                "replayed":
                    True,
            }

        if (
            approval.status
            == "failed"
        ):
            return {
                "ok":
                    False,

                "approval":
                    approval_to_dict(
                        approval
                    ),

                "result":
                    approval.result,

                "replayed":
                    True,

                "error": (
                    f"Approval '{approval_id}' "
                    "previously failed."
                ),
            }

        # ========================================================
        # AMBIGUOUS EXECUTION STATE
        # ========================================================

        if (
            approval.status
            == "executing"
        ):
            return (
                execution_unresolved_result(
                    approval
                )
            )

        if (
            approval.status
            != "pending"
        ):
            return {
                "ok":
                    False,

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

        # ========================================================
        # DURABLE EXECUTION CLAIM
        #
        # pending -> executing happens before MCP touches the host.
        # ========================================================

        claimed = (
            self.store
            .claim_for_execution(
                approval_id
            )
        )

        if not claimed:
            return (
                self._claim_failure_result(
                    approval_id
                )
            )

        approval = (
            self.store.get(
                approval_id
            )
        )

        if approval is None:
            return {
                "ok":
                    False,

                "error": (
                    f"Approval '{approval_id}' "
                    "disappeared after "
                    "execution claim."
                ),
            }

        print(
            "\n[MCP Mutation] "
            f"{approval.tool} "
            f"{approval.arguments}"
        )

        # ========================================================
        # EXECUTE EXACT PERSISTED ACTION
        # ========================================================

        try:
            result = (
                await
                self.mcp.call_tool(
                    approval.tool,
                    approval.arguments,
                )
            )

        except Exception as exc:
            current = (
                self.store.get(
                    approval_id
                )
            )

            return {
                "ok":
                    False,

                "approval": (
                    approval_to_dict(
                        current
                    )
                    if current
                    is not None
                    else None
                ),

                "replayed":
                    False,

                "error": (
                    "Approval execution outcome "
                    "could not be confirmed. "
                    "The approval remains in "
                    "executing state and requires "
                    "reconciliation. "
                    f"Error: {exc}"
                ),
            }

        if not isinstance(
            result,
            dict,
        ):
            current = (
                self.store.get(
                    approval_id
                )
            )

            return {
                "ok":
                    False,

                "approval": (
                    approval_to_dict(
                        current
                    )
                    if current
                    is not None
                    else None
                ),

                "replayed":
                    False,

                "error": (
                    "Approval execution returned "
                    "an invalid result. "
                    "The approval remains in "
                    "executing state because the "
                    "side-effect outcome is unknown."
                ),
            }

        # ========================================================
        # AMBIGUOUS REMOTE MUTATION OUTCOME
        #
        # A provider may have received/applied the side effect even
        # though trusted code cannot determine the final state.
        #
        # DO NOT:
        #     mark failed
        #     return to pending
        #     automatically retry
        #
        # Keep the durable approval in executing state and persist
        # the provider evidence for later reconciliation.
        # ========================================================

        ambiguous_statuses = {
            "outcome_unknown",

            # Compatibility with older mutation adapters.
            # New hardened providers should prefer
            # ``outcome_unknown``.
            "unknown",
        }

        if (
            result.get(
                "status"
            )
            in ambiguous_statuses
        ):

            try:
                approval = (
                    self.store
                    .record_execution_result(
                        approval_id,
                        result,
                    )
                )

            except Exception as exc:
                current = (
                    self.store.get(
                        approval_id
                    )
                )

                return {
                    "ok":
                        False,

                    "approval": (
                        approval_to_dict(
                            current
                        )
                        if current
                        is not None
                        else None
                    ),

                    "result":
                        result,

                    "replayed":
                        False,

                    "error": (
                        "Approval execution reached an "
                        "ambiguous remote outcome. "
                        "Automatic retry is disabled. "
                        "The approval remains unresolved, "
                        "but the provider result could not "
                        "be durably recorded. "
                        f"Error: {exc}"
                    ),
                }

            return {
                "ok":
                    False,

                "approval":
                    approval_to_dict(
                        approval
                    ),

                "result":
                    result,

                "replayed":
                    False,

                "error": (
                    "Approval execution outcome is unresolved. "
                    "The approval remains in executing state "
                    "and automatic retry is disabled pending "
                    "trusted reconciliation."
                ),
            }

        successful_statuses = {
            "executed",
            "success",
        }

        execution_succeeded = (
            result.get(
                "ok"
            )
            is True
            and result.get(
                "status"
            )
            in successful_statuses
        )

        # ========================================================
        # DURABLE TERMINAL RESULT
        # ========================================================

        try:
            if execution_succeeded:
                approval = (
                    self.store
                    .mark_approved(
                        approval_id,
                        result,
                    )
                )

            else:
                approval = (
                    self.store
                    .mark_failed(
                        approval_id,
                        result,
                    )
                )

        except Exception as exc:
            current = (
                self.store.get(
                    approval_id
                )
            )

            return {
                "ok":
                    False,

                "approval": (
                    approval_to_dict(
                        current
                    )
                    if current
                    is not None
                    else None
                ),

                "result":
                    result,

                "replayed":
                    False,

                "error": (
                    "Approval execution finished "
                    "but its terminal state could "
                    "not be persisted. "
                    "Automatic retry is disabled. "
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

    def _claim_failure_result(
        self,
        approval_id: str,
    ) -> dict:
        approval = (
            self.store.get(
                approval_id
            )
        )

        if approval is None:
            return {
                "ok":
                    False,

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
                "ok":
                    True,

                "approval":
                    approval_to_dict(
                        approval
                    ),

                "result":
                    approval.result,

                "replayed":
                    True,
            }

        if (
            approval.status
            == "failed"
        ):
            return {
                "ok":
                    False,

                "approval":
                    approval_to_dict(
                        approval
                    ),

                "result":
                    approval.result,

                "replayed":
                    True,

                "error": (
                    f"Approval '{approval_id}' "
                    "previously failed."
                ),
            }

        if (
            approval.status
            == "executing"
        ):
            return (
                execution_unresolved_result(
                    approval
                )
            )

        return {
            "ok":
                False,

            "approval":
                approval_to_dict(
                    approval
                ),

            "error": (
                f"Approval '{approval_id}' "
                "could not be claimed."
            ),
        }