import asyncio

import pytest

from agent.approval_store import (
    ApprovalStore,
)

from agent.approvals import (
    ApprovalManager,
)


TOOL_NAME = (
    "ticket_create"
)


def tool_lookup(
    name: str,
):

    if (
        name
        != TOOL_NAME
    ):
        return None

    return {
        "requires_approval":
            True,

        "risk":
            "medium",
    }


class FakeMCP:

    def __init__(
        self,
        result: dict,
    ) -> None:

        self.result = (
            result
        )

        self.calls = 0

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:

        assert (
            tool_name
            == TOOL_NAME
        )

        assert (
            arguments[
                "project_key"
            ]
            == "KAN"
        )

        self.calls += 1

        return dict(
            self.result
        )


def build_manager(
    tmp_path,
    result: dict,
):

    store = (
        ApprovalStore(
            tmp_path
            / "approvals.db"
        )
    )

    store.initialize()

    mcp = (
        FakeMCP(
            result
        )
    )

    manager = (
        ApprovalManager(
            store=store,
            mcp=mcp,
            tool_lookup=(
                tool_lookup
            ),
        )
    )

    approval = (
        manager
        .create_approval(
            TOOL_NAME,
            {
                "project_key":
                    "KAN",

                "summary":
                    "VPN login failure",
            },
            risk="medium",
        )
    )

    return (
        store,
        mcp,
        manager,
        approval,
    )


@pytest.mark.parametrize(
    "provider_status",
    [
        "outcome_unknown",

        # Compatibility with existing older mutation adapters.
        "unknown",
    ],
)
def test_ambiguous_provider_outcome_remains_executing_and_is_not_retried(
    tmp_path,
    provider_status,
):

    provider_result = {
        "ok":
            False,

        "status":
            provider_status,

        "operation":
            "create_ticket",

        "mutation_performed":
            None,

        "verification_ok":
            False,

        "error":
            "Remote mutation outcome is ambiguous.",
    }

    (
        store,
        mcp,
        manager,
        approval,
    ) = (
        build_manager(
            tmp_path,
            provider_result,
        )
    )

    first = (
        asyncio.run(
            manager
            .approve_approval(
                approval[
                    "id"
                ]
            )
        )
    )

    assert first["ok"] is False

    assert (
        first[
            "replayed"
        ]
        is False
    )

    assert (
        first[
            "result"
        ]
        == provider_result
    )

    assert (
        first[
            "approval"
        ][
            "status"
        ]
        == "executing"
    )

    assert mcp.calls == 1

    persisted = (
        store.get(
            approval[
                "id"
            ]
        )
    )

    assert persisted is not None

    assert (
        persisted.status
        == "executing"
    )

    assert (
        persisted.result
        == provider_result
    )

    # ========================================================
    # RETRY / REPLAY MUST NOT EXECUTE MCP AGAIN.
    # ========================================================

    second = (
        asyncio.run(
            manager
            .approve_approval(
                approval[
                    "id"
                ]
            )
        )
    )

    assert second["ok"] is False

    assert (
        second[
            "replayed"
        ]
        is True
    )

    assert (
        second[
            "approval"
        ][
            "status"
        ]
        == "executing"
    )

    assert (
        second[
            "result"
        ]
        == provider_result
    )

    assert mcp.calls == 1


def test_known_provider_failure_becomes_terminal_failed(
    tmp_path,
):

    provider_result = {
        "ok":
            False,

        "status":
            "denied",

        "operation":
            "create_ticket",

        "mutation_performed":
            False,

        "verification_ok":
            False,
    }

    (
        store,
        mcp,
        manager,
        approval,
    ) = (
        build_manager(
            tmp_path,
            provider_result,
        )
    )

    result = (
        asyncio.run(
            manager
            .approve_approval(
                approval[
                    "id"
                ]
            )
        )
    )

    assert result["ok"] is False

    assert (
        result[
            "approval"
        ][
            "status"
        ]
        == "failed"
    )

    persisted = (
        store.get(
            approval[
                "id"
            ]
        )
    )

    assert persisted is not None
    assert persisted.status == "failed"
    assert persisted.result == provider_result

    assert mcp.calls == 1


def test_verified_provider_success_becomes_terminal_approved(
    tmp_path,
):

    provider_result = {
        "ok":
            True,

        "status":
            "success",

        "operation":
            "create_ticket",

        "mutation_performed":
            True,

        "verification_ok":
            True,

        "ticket_key":
            "KAN-4",
    }

    (
        store,
        mcp,
        manager,
        approval,
    ) = (
        build_manager(
            tmp_path,
            provider_result,
        )
    )

    result = (
        asyncio.run(
            manager
            .approve_approval(
                approval[
                    "id"
                ]
            )
        )
    )

    assert result["ok"] is True

    assert (
        result[
            "approval"
        ][
            "status"
        ]
        == "approved"
    )

    persisted = (
        store.get(
            approval[
                "id"
            ]
        )
    )

    assert persisted is not None
    assert persisted.status == "approved"
    assert persisted.result == provider_result

    assert mcp.calls == 1


@pytest.mark.parametrize("receipt", [
    {},
    {"result": "acknowledged"},
    {"ok": "false", "status": "error"},
    {"ok": 1, "status": "success"},
    {"ok": True},
    {"ok": False, "status": None},
    {"ok": False, "status": []},
    {"ok": False, "status": "unrecognized"},
    {"ok": True, "status": "denied"},
    {"ok": False, "status": "success"},
    {"ok": True, "status": "outcome_unknown"},
    {"ok": False, "status": "timeout"},
    {"ok": False, "status": "cancelled"},
])
def test_malformed_or_contradictory_receipt_preserves_evidence_without_retry(tmp_path, receipt):
    store, mcp, manager, approval = build_manager(tmp_path, receipt)
    first = asyncio.run(manager.approve_approval(approval["id"]))
    assert first["ok"] is False
    assert first["approval"]["status"] == "executing"
    assert first["result"]["status"] == "outcome_unknown"
    assert first["result"]["error_code"] == "tool_receipt_invalid"
    assert first["result"]["provider_receipt"] == receipt
    assert store.get(approval["id"]).result == first["result"]
    second = asyncio.run(manager.approve_approval(approval["id"]))
    assert second["replayed"] is True
    assert second["result"] == first["result"]
    assert mcp.calls == 1


@pytest.mark.parametrize("status", [
    "denied", "error", "conflict", "incomplete", "rate_limited",
    "not_found", "unsupported", "failed",
])
def test_valid_negative_outcome_remains_a_known_failure(tmp_path, status):
    receipt = {"ok": False, "status": status, "changed": False, "verification_ok": False}
    store, mcp, manager, approval = build_manager(tmp_path, receipt)
    result = asyncio.run(manager.approve_approval(approval["id"]))
    assert result["approval"]["status"] == "failed"
    assert result["result"] == receipt
    assert store.get(approval["id"]).result == receipt
    assert mcp.calls == 1


@pytest.mark.parametrize("status", ["success", "executed"])
@pytest.mark.parametrize("verification", [False, 0, 1, "false", "true", [], {}])
def test_explicit_unverified_or_malformed_success_stays_unresolved(tmp_path, status, verification):
    receipt = {"ok": True, "status": status, "verification_ok": verification}
    store, mcp, manager, approval = build_manager(tmp_path, receipt)
    first = asyncio.run(manager.approve_approval(approval["id"]))
    assert first["ok"] is False
    assert first["approval"]["status"] == "executing"
    assert first["result"]["status"] == "outcome_unknown"
    assert first["result"]["provider_receipt"] == receipt
    assert store.get(approval["id"]).result == first["result"]
    second = asyncio.run(manager.approve_approval(approval["id"]))
    assert second["replayed"] is True
    assert second["result"] == first["result"]
    assert mcp.calls == 1


@pytest.mark.parametrize("status", ["success", "executed"])
@pytest.mark.parametrize("evidence", [{}, {"verification_ok": None}, {"verification_ok": True}])
def test_legacy_absent_or_unknown_and_positive_verification_remain_compatible(tmp_path, status, evidence):
    receipt = {"ok": True, "status": status, **evidence}
    store, mcp, manager, approval = build_manager(tmp_path, receipt)
    result = asyncio.run(manager.approve_approval(approval["id"]))
    assert result["ok"] is True
    assert result["approval"]["status"] == "approved"
    assert result["result"] == receipt
    assert store.get(approval["id"]).result == receipt
    assert mcp.calls == 1
