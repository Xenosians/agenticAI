from __future__ import annotations

import asyncio
import json

from pathlib import (
    Path,
)

from learning.evidence.account_exercise import (
    ACCOUNT_SPECIALIST_NAME,
    AccountEvidenceMcpGuard,
    build_account_evidence_observation,
    build_account_evidence_session_report,
    create_account_evidence_approval,
    write_account_evidence_session_report,
)

from subagents.core.definitions.types import (
    AgentResult,
    HubResult,
)


class FakeMcp:
    def __init__(
        self,
    ) -> None:

        self.calls: list[
            tuple[
                str,
                dict,
            ]
        ] = []

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:

        self.calls.append(
            (
                tool_name,
                arguments,
            )
        )

        return {
            "ok":
                True,

            "status":
                "success",

            "user_id":
                arguments.get(
                    "user_id"
                ),

            "enabled":
                True,

            "locked":
                False,
        }


def test_account_evidence_mcp_guard_allows_only_status(
) -> None:

    async def scenario(
    ) -> None:

        delegate = (
            FakeMcp()
        )

        guard = (
            AccountEvidenceMcpGuard(
                delegate
            )
        )

        status_result = (
            await guard.call_tool(
                "account_status",
                {
                    "user_id":
                        "jdoe",
                },
            )
        )

        unlock_result = (
            await guard.call_tool(
                "unlock_user",
                {
                    "user_id":
                        "jdoe",
                },
            )
        )

        reset_result = (
            await guard.call_tool(
                "reset_password",
                {
                    "user_id":
                        "jdoe",
                },
            )
        )

        assert (
            status_result[
                "ok"
            ]
            is True
        )

        assert (
            unlock_result[
                "ok"
            ]
            is False
        )

        assert (
            unlock_result[
                "status"
            ]
            == "denied"
        )

        assert (
            reset_result[
                "ok"
            ]
            is False
        )

        assert (
            delegate.calls
            == [
                (
                    "account_status",
                    {
                        "user_id":
                            "jdoe",
                    },
                )
            ]
        )

    asyncio.run(
        scenario()
    )


def test_account_evidence_approval_is_non_persistent_placeholder(
) -> None:

    approval = (
        create_account_evidence_approval(
            "unlock_user",
            {
                "user_id":
                    "jdoe",
            },
            risk="medium",
        )
    )

    assert (
        approval[
            "id"
        ]
        .startswith(
            "account-evidence-approval-"
        )
    )

    assert (
        approval[
            "status"
        ]
        == "evidence_only"
    )


def test_build_account_evidence_observation_preserves_provenance(
) -> None:

    result = (
        HubResult(
            status=(
                "approval_required"
            ),

            user_request=(
                "is jdoe unlocked?"
            ),

            routes=[
                ACCOUNT_SPECIALIST_NAME
            ],

            results=[
                AgentResult(
                    task_id=(
                        "task-1"
                    ),

                    agent_name=(
                        ACCOUNT_SPECIALIST_NAME
                    ),

                    status=(
                        "approval_required"
                    ),

                    outcome_code=(
                        "approval_required"
                    ),

                    task_instructions=(
                        "Check the account state."
                    ),

                    execution_provenance={
                        "schema":
                            (
                                "specialist-"
                                "execution-provenance.v1"
                            ),

                        "provenance_complete":
                            True,

                        "model_key":
                            (
                                "qwen2.5-"
                                "0.5b-funccall"
                            ),

                        "model_artifact_sha256":
                            "abc123",
                    },

                    proposed_tool=(
                        "unlock_user"
                    ),

                    proposed_arguments={
                        "user_id":
                            "jdoe",
                    },

                    approval_id=(
                        "evidence-approval"
                    ),
                )
            ],

            answer=(
                "Approval required."
            ),
        )
    )

    observation = (
        build_account_evidence_observation(
            request_index=1,

            job_id=(
                "job-1"
            ),

            result=(
                result
            ),

            trajectory_payload={
                "trajectory_id":
                    "trajectory-1",
            },
        )
    )

    assert (
        observation.account_routed
        is True
    )

    assert (
        observation.account_result_count
        == 1
    )

    specialist = (
        observation
        .account_results[
            0
        ]
    )

    assert (
        specialist.proposed_tool
        == "unlock_user"
    )

    assert (
        specialist.provenance_present
        is True
    )

    assert (
        specialist.provenance_complete
        is True
    )

    assert (
        specialist.model_key
        == "qwen2.5-0.5b-funccall"
    )


def test_session_report_is_immutable_file(
    tmp_path: Path,
) -> None:

    result = (
        HubResult(
            status="success",

            user_request=(
                "Is jdoe locked?"
            ),

            routes=[
                ACCOUNT_SPECIALIST_NAME
            ],

            results=[
                AgentResult(
                    task_id="task-2",

                    agent_name=(
                        ACCOUNT_SPECIALIST_NAME
                    ),

                    status="success",

                    outcome_code="success",

                    proposed_tool=(
                        "account_status"
                    ),

                    proposed_arguments={
                        "user_id":
                            "jdoe",
                    },

                    execution_provenance={
                        "schema":
                            (
                                "specialist-"
                                "execution-provenance.v1"
                            ),

                        "provenance_complete":
                            True,

                        "model_key":
                            (
                                "qwen2.5-"
                                "0.5b-funccall"
                            ),
                    },

                    tool_result={
                        "ok":
                            True,

                        "enabled":
                            True,

                        "locked":
                            False,
                    },
                )
            ],

            answer=(
                "jdoe is not locked."
            ),
        )
    )

    observation = (
        build_account_evidence_observation(
            request_index=1,

            job_id="job-2",

            result=result,

            trajectory_payload={
                "trajectory_id":
                    "trajectory-2",
            },
        )
    )

    report = (
        build_account_evidence_session_report(
            session_id=(
                "session-1"
            ),

            observations=[
                observation
            ],
        )
    )

    target = (
        write_account_evidence_session_report(
            report=report,
            root=tmp_path,
        )
    )

    payload = (
        json.loads(
            target.read_text(
                encoding="utf-8"
            )
        )
    )

    assert (
        payload[
            "schema"
        ]
        == "account-evidence-session.v1"
    )

    assert (
        payload[
            "request_count"
        ]
        == 1
    )

    assert (
        payload[
            "account_routed_count"
        ]
        == 1
    )

    try:

        write_account_evidence_session_report(
            report=report,
            root=tmp_path,
        )

    except FileExistsError:
        pass

    else:

        raise AssertionError(
            "Evidence report unexpectedly "
            "overwrote an existing session."
        )
