from __future__ import annotations

import json
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.evidence.sanitizer import (
    sanitize_value,
)

from subagents.core.definitions.types import (
    HubResult,
)


ACCOUNT_SPECIALIST_NAME = (
    "account-specialist"
)


SAFE_ACCOUNT_EVIDENCE_EXECUTION_TOOLS = {
    "account_status",
}


# ============================================================
# TIME
# ============================================================


def _utc_now(
) -> str:

    return (
        datetime
        .now(
            timezone.utc
        )
        .isoformat()
    )


# ============================================================
# SAFETY BOUNDARIES
# ============================================================


def create_account_evidence_approval(
    tool_name: str,
    arguments: dict,
    *,
    risk: str,
) -> dict:
    """
    Create a non-persistent approval placeholder for an evidence run.

    Evidence collection must never create a real approval that can
    later be executed accidentally.

    ToolGateway only requires the returned object to provide an id.
    """

    return {
        "id": (
            "account-evidence-approval-"
            f"{uuid.uuid4().hex}"
        ),

        "tool":
            tool_name,

        "arguments":
            arguments,

        "risk":
            risk,

        "status":
            "evidence_only",
    }


class AccountEvidenceMcpGuard:
    """
    Fail-closed wrapper around the real MCP runtime.

    Read-only account_status is allowed to execute normally.

    Account mutations must never physically reach MCP from this
    evidence harness, even if a trusted tool policy were accidentally
    changed to auto-approve them in the future.
    """

    def __init__(
        self,
        delegate,
    ) -> None:

        self.delegate = (
            delegate
        )

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:

        if (
            tool_name
            not in
            SAFE_ACCOUNT_EVIDENCE_EXECUTION_TOOLS
        ):

            return {
                "ok":
                    False,

                "status":
                    "denied",

                "error": (
                    "Account evidence harness blocked "
                    "physical execution of mutating tool "
                    f"'{tool_name}'."
                ),
            }

        return (
            await
            self.delegate
            .call_tool(
                tool_name,
                arguments,
            )
        )


# ============================================================
# DERIVED REPORT TYPES
# ============================================================


class AccountSpecialistObservation(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    task_id: str

    status: str

    outcome_code: (
        str
        | None
    ) = None

    task_instructions: (
        str
        | None
    ) = None

    proposed_tool: (
        str
        | None
    ) = None

    proposed_arguments: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    approval_id: (
        str
        | None
    ) = None

    error: (
        str
        | None
    ) = None

    tool_result: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    provenance_present: bool = False

    provenance_complete: bool = False

    provenance_schema: (
        str
        | None
    ) = None

    model_key: (
        str
        | None
    ) = None

    model_artifact_sha256: (
        str
        | None
    ) = None


class AccountEvidenceObservation(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    request_index: int

    job_id: str

    trajectory_id: str

    user_request: str

    hub_status: str

    routes: list[
        str
    ] = Field(
        default_factory=list
    )

    account_routed: bool

    account_result_count: int

    account_results: list[
        AccountSpecialistObservation
    ] = Field(
        default_factory=list
    )

    final_answer: (
        str
        | None
    ) = None


class AccountEvidenceSessionReport(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "account-evidence-session.v1"
        ),
        alias="schema",
    )

    session_id: str

    created_at: str

    request_count: int

    account_routed_count: int

    trajectory_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    observations: list[
        AccountEvidenceObservation
    ] = Field(
        default_factory=list
    )


# ============================================================
# OBSERVATION BUILDING
# ============================================================


def _specialist_observation(
    result,
) -> AccountSpecialistObservation:

    provenance = (
        result.execution_provenance
    )

    if not isinstance(
        provenance,
        dict,
    ):

        provenance = {}

    return (
        AccountSpecialistObservation(
            task_id=(
                result.task_id
            ),

            status=(
                result.status
            ),

            outcome_code=(
                result.outcome_code
            ),

            task_instructions=(
                result.task_instructions
            ),

            proposed_tool=(
                result.proposed_tool
            ),

            proposed_arguments=(
                result.proposed_arguments
            ),

            approval_id=(
                result.approval_id
            ),

            error=(
                result.error
            ),

            tool_result=(
                result.tool_result
            ),

            provenance_present=(
                bool(
                    provenance
                )
            ),

            provenance_complete=(
                provenance.get(
                    "provenance_complete"
                )
                is True
            ),

            provenance_schema=(
                provenance.get(
                    "schema"
                )
                or provenance.get(
                    "schema_name"
                )
            ),

            model_key=(
                provenance.get(
                    "model_key"
                )
            ),

            model_artifact_sha256=(
                provenance.get(
                    "model_artifact_sha256"
                )
            ),
        )
    )


def build_account_evidence_observation(
    *,
    request_index: int,
    job_id: str,
    result: HubResult,
    trajectory_payload: dict,
) -> AccountEvidenceObservation:

    trajectory_id = (
        trajectory_payload.get(
            "trajectory_id"
        )
    )

    if not isinstance(
        trajectory_id,
        str,
    ) or not trajectory_id.strip():

        raise ValueError(
            "Recorded trajectory payload is missing "
            "trajectory_id."
        )

    account_results = [
        _specialist_observation(
            item
        )

        for item
        in result.results

        if (
            item.agent_name
            == ACCOUNT_SPECIALIST_NAME
        )
    ]

    return (
        AccountEvidenceObservation(
            request_index=(
                request_index
            ),

            job_id=(
                job_id
            ),

            trajectory_id=(
                trajectory_id
            ),

            user_request=(
                result.user_request
            ),

            hub_status=(
                result.status
            ),

            routes=(
                list(
                    result.routes
                )
            ),

            account_routed=(
                ACCOUNT_SPECIALIST_NAME
                in result.routes
            ),

            account_result_count=(
                len(
                    account_results
                )
            ),

            account_results=(
                account_results
            ),

            final_answer=(
                result.answer
            ),
        )
    )


def build_account_evidence_session_report(
    *,
    session_id: str,
    observations: list[
        AccountEvidenceObservation
    ],
) -> AccountEvidenceSessionReport:

    return (
        AccountEvidenceSessionReport(
            session_id=(
                session_id
            ),

            created_at=(
                _utc_now()
            ),

            request_count=(
                len(
                    observations
                )
            ),

            account_routed_count=(
                sum(
                    1

                    for observation
                    in observations

                    if observation.account_routed
                )
            ),

            trajectory_ids=[
                observation.trajectory_id

                for observation
                in observations
            ],

            observations=(
                observations
            ),
        )
    )


# ============================================================
# IMMUTABLE SESSION REPORT
# ============================================================


def write_account_evidence_session_report(
    *,
    report: AccountEvidenceSessionReport,
    root: Path,
) -> Path:

    resolved_root = (
        root
        .expanduser()
        .resolve()
    )

    resolved_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    target = (
        resolved_root
        / (
            f"{report.session_id}.json"
        )
    )

    payload = (
        sanitize_value(
            report.model_dump(
                mode="json",
                by_alias=True,
            )
        )
    )

    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )

    with target.open(
        "x",
        encoding="utf-8",
    ) as handle:

        handle.write(
            serialized
        )

        handle.flush()

    return target
