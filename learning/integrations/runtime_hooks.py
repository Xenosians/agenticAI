from __future__ import annotations

from dataclasses import (
    asdict,
    is_dataclass,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
)

from learning.context import (
    LearningContextRecorder,
    StructuredContextRecord,
    record_git_context,
    record_human_context,
    record_jira_context,
    record_shell_context,
    record_task_context,
    record_tool_context,
)

from learning.paths import (
    RUNTIME_LEARNING_ROOT,
)


DEFAULT_CONTEXT_RECORDS_PATH = (
    RUNTIME_LEARNING_ROOT
    / "continual"
    / "context-records.jsonl"
)


# ============================================================
# HISTORICAL RUNTIME OUTCOME CLASSIFICATION
# ============================================================


# Historical trajectories created before exact runtime-decision
# capture may contain only outcome_code.
#
# These outcomes happened before ToolGateway could be treated as
# the source of the final decision.
#
# This table is observability metadata only.
#
# It is NOT execution policy.
PRE_GATEWAY_OUTCOME_CODES = {
    "agent_definition_error",
    "model_generation_error",
    "tool_parse_error",
    "invalid_tool_call_count",
    "semantic_guard_exception",
}


# ============================================================
# GENERIC NORMALIZATION
# ============================================================


def _plain(
    value: Any,
) -> Any:

    if value is None:
        return None

    if hasattr(
        value,
        "model_dump",
    ):

        return (
            value.model_dump(
                mode="json",
            )
        )

    if is_dataclass(
        value
    ):

        return (
            asdict(
                value
            )
        )

    if isinstance(
        value,
        dict,
    ):

        return {
            key:
                _plain(
                    item
                )

            for (
                key,
                item,
            )
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        return [
            _plain(
                item
            )

            for item
            in value
        ]

    return value


def _string(
    value: Any,
) -> str | None:

    if not isinstance(
        value,
        str,
    ):

        return None

    normalized = (
        value.strip()
    )

    if not normalized:

        return None

    return normalized


def _dict(
    value: Any,
) -> dict[
    str,
    Any,
] | None:

    if not isinstance(
        value,
        dict,
    ):

        return None

    return (
        dict(
            value
        )
    )


def _dict_list(
    value: Any,
) -> list[
    dict[
        str,
        Any,
    ]
]:

    if not isinstance(
        value,
        list,
    ):

        return []

    return [
        dict(
            item
        )

        for item
        in value

        if isinstance(
            item,
            dict,
        )
    ]


def _string_list(
    value: Any,
) -> list[str]:

    if not isinstance(
        value,
        list,
    ):

        return []

    result: list[str] = []

    for item in value:

        normalized = (
            _string(
                item
            )
        )

        if normalized is None:

            continue

        result.append(
            normalized
        )

    return result


def _append_record(
    records: list[
        StructuredContextRecord
    ],
    record: (
        StructuredContextRecord
        | None
    ),
) -> None:

    if record is not None:

        records.append(
            record
        )


# ============================================================
# DOMAIN EXTRACTION
# ============================================================


def _jira_ticket(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    ticket = (
        result.get(
            "ticket"
        )
    )

    if isinstance(
        ticket,
        dict,
    ):

        return (
            dict(
                ticket
            )
        )

    return {}


def _git_repository(
    *,
    result: dict[
        str,
        Any,
    ],
    arguments: dict[
        str,
        Any,
    ],
) -> str | None:

    return (
        _string(
            result.get(
                "repository"
            )
        )
        or
        _string(
            arguments.get(
                "repository"
            )
        )
    )


def _record_git_result(
    *,
    recorder: LearningContextRecorder,
    tool_name: str,
    arguments: dict[
        str,
        Any,
    ],
    result: dict[
        str,
        Any,
    ],
    task_goal: str | None,
    previous_results: list[
        dict[
            str,
            Any,
        ]
    ],
    trajectory_id: str,
    task_id: str | None,
    job_id: str | None,
) -> StructuredContextRecord | None:

    repository = (
        _git_repository(
            result=(
                result
            ),

            arguments=(
                arguments
            ),
        )
    )

    if repository is None:

        return None

    branch = (
        _string(
            result.get(
                "branch"
            )
        )
        or
        _string(
            result.get(
                "current_branch"
            )
        )
    )

    clean = (
        result.get(
            "clean"
        )
    )

    if not isinstance(
        clean,
        bool,
    ):

        clean = None

    ahead = (
        result.get(
            "ahead"
        )
    )

    if (
        not isinstance(
            ahead,
            int,
        )
        or isinstance(
            ahead,
            bool,
        )
    ):

        ahead = None

    behind = (
        result.get(
            "behind"
        )
    )

    if (
        not isinstance(
            behind,
            int,
        )
        or isinstance(
            behind,
            bool,
        )
    ):

        behind = None

    diff_summary = (
        _string(
            result.get(
                "diff"
            )
        )
    )

    return (
        record_git_context(
            recorder,

            repository=(
                repository
            ),

            branch=(
                branch
            ),

            clean=(
                clean
            ),

            ahead=(
                ahead
            ),

            behind=(
                behind
            ),

            status_changes=(
                _dict_list(
                    result.get(
                        "changes"
                    )
                )
            ),

            recent_commits=(
                _dict_list(
                    result.get(
                        "commits"
                    )
                )
            ),

            diff_summary=(
                diff_summary
            ),

            changed_files=(
                _string_list(
                    result.get(
                        "files"
                    )
                )
            ),

            remotes=(
                _string_list(
                    result.get(
                        "remotes"
                    )
                )
            ),

            previous_results=(
                previous_results
            ),

            task_goal=(
                task_goal
            ),

            trajectory_id=(
                trajectory_id
            ),

            task_id=(
                task_id
            ),

            job_id=(
                job_id
            ),

            source_tool=(
                tool_name
            ),
        )
    )


def _record_jira_result(
    *,
    recorder: LearningContextRecorder,
    tool_name: str,
    arguments: dict[
        str,
        Any,
    ],
    result: dict[
        str,
        Any,
    ],
    original_request: str,
    trajectory_id: str,
    task_id: str | None,
    job_id: str | None,
) -> StructuredContextRecord | None:

    ticket = (
        _jira_ticket(
            result
        )
    )

    ticket_key = (
        _string(
            arguments.get(
                "ticket_key"
            )
        )
        or
        _string(
            result.get(
                "ticket_key"
            )
        )
        or
        _string(
            ticket.get(
                "key"
            )
        )
    )

    project_key = (
        _string(
            arguments.get(
                "project_key"
            )
        )
        or
        _string(
            result.get(
                "project_key"
            )
        )
        or
        _string(
            ticket.get(
                "project_key"
            )
        )
    )

    # Project-agnostic searches still retain generic task/tool
    # evidence.
    #
    # Do not invent a Jira identity merely to create a domain
    # snapshot.
    if (
        ticket_key is None
        and project_key is None
    ):

        return None

    comments = (
        _dict_list(
            result.get(
                "comments"
            )
        )
    )

    relationships = (
        _dict_list(
            result.get(
                "relationships"
            )
        )
    )

    workflow_state: dict[
        str,
        Any,
    ] = {}

    provider_status = (
        _string(
            result.get(
                "status"
            )
        )
    )

    if provider_status is not None:

        workflow_state[
            "provider_status"
        ] = (
            provider_status
        )

    history = (
        _dict_list(
            result.get(
                "history"
            )
        )
    )

    if history:

        workflow_state[
            "history"
        ] = (
            history
        )

    return (
        record_jira_context(
            recorder,

            project_key=(
                project_key
            ),

            ticket_key=(
                ticket_key
            ),

            summary=(
                _string(
                    ticket.get(
                        "summary"
                    )
                )
                or
                _string(
                    result.get(
                        "summary"
                    )
                )
            ),

            status=(
                _string(
                    ticket.get(
                        "status"
                    )
                )
            ),

            assignee=(
                _string(
                    ticket.get(
                        "assignee"
                    )
                )
            ),

            comments=(
                comments
            ),

            relationships=(
                relationships
            ),

            prior_agent_actions=[],

            workflow_state=(
                workflow_state
            ),

            user_request=(
                original_request
            ),

            trajectory_id=(
                trajectory_id
            ),

            task_id=(
                task_id
            ),

            job_id=(
                job_id
            ),

            source_tool=(
                tool_name
            ),
        )
    )


def _record_shell_result(
    *,
    recorder: LearningContextRecorder,
    tool_name: str,
    arguments: dict[
        str,
        Any,
    ],
    result: dict[
        str,
        Any,
    ],
    task_goal: str | None,
    trajectory_id: str,
    task_id: str | None,
    job_id: str | None,
) -> StructuredContextRecord | None:

    # Git receives its own structured domain snapshot.
    if (
        tool_name.startswith(
            "workspace_git_"
        )
    ):

        return None

    if not (
        tool_name.startswith(
            "workspace_"
        )
        or tool_name
        == "process_exec"
    ):

        return None

    path = (
        _string(
            result.get(
                "project_path"
            )
        )
        or
        _string(
            result.get(
                "path"
            )
        )
        or
        _string(
            arguments.get(
                "relative_path"
            )
        )
        or
        _string(
            arguments.get(
                "cwd"
            )
        )
    )

    project_type = (
        _string(
            result.get(
                "project_type"
            )
        )
    )

    files = (
        _string_list(
            result.get(
                "files"
            )
        )
    )

    stdout = (
        _string(
            result.get(
                "stdout"
            )
        )
    )

    stderr = (
        _string(
            result.get(
                "stderr"
            )
        )
    )

    exit_code = (
        result.get(
            "exit_code"
        )
    )

    if (
        not isinstance(
            exit_code,
            int,
        )
        or isinstance(
            exit_code,
            bool,
        )
    ):

        exit_code = None

    tests: list[
        dict[
            str,
            Any,
        ]
    ] = []

    builds: list[
        dict[
            str,
            Any,
        ]
    ] = []

    if (
        tool_name
        == "workspace_run_tests"
    ):

        tests.append(
            dict(
                result
            )
        )

    if (
        tool_name
        == "workspace_run_build"
    ):

        builds.append(
            dict(
                result
            )
        )

    return (
        record_shell_context(
            recorder,

            subject=(
                f"tool:{tool_name}"
            ),

            workspace=(
                path
            ),

            repository=None,

            files=(
                files
            ),

            language=(
                project_type
            ),

            toolchain=[],

            dependencies=[],

            previous_stdout=(
                stdout
            ),

            previous_stderr=(
                stderr
            ),

            exit_code=(
                exit_code
            ),

            tests=(
                tests
            ),

            builds=(
                builds
            ),

            previous_commands=[],

            current_task=(
                task_goal
            ),

            trajectory_id=(
                trajectory_id
            ),

            task_id=(
                task_id
            ),

            job_id=(
                job_id
            ),

            source_tool=(
                tool_name
            ),
        )
    )


# ============================================================
# RUNTIME HOOKS
# ============================================================


class ContinualLearningRuntimeHooks:
    """
    Additive evidence hooks for the existing runtime.

    The hooks only persist sanitized contextual evidence.

    They NEVER:
        - authorize tools;
        - alter SemanticIntent;
        - override SemanticGuard;
        - override ToolGateway;
        - approve mutations;
        - execute providers;
        - mutate model weights.

    New trajectories may contain exact live SemanticGuard and
    ToolGateway decision evidence.

    Historical trajectories may contain only stable outcome_code
    fields.

    Exact evidence always takes precedence.

    Historical reconstruction remains only for backward
    compatibility.
    """

    def __init__(
        self,
        *,
        path: Path = (
            DEFAULT_CONTEXT_RECORDS_PATH
        ),
        enabled: bool = True,
    ) -> None:

        self.recorder = (
            LearningContextRecorder(
                path=(
                    path
                ),

                enabled=(
                    enabled
                ),
            )
        )

    # ========================================================
    # LOW-LEVEL HOOKS
    # ========================================================

    def record_hub_context(
        self,
        *,
        original_request: str,
        routing_context: list[
            Any
        ] | None = None,
        semantic_contract: (
            Any
            | None
        ) = None,
        specialist_instructions: (
            str
            | None
        ) = None,
        previous_trusted_results: list[
            Any
        ] | None = None,
        trajectory_id: (
            str
            | None
        ) = None,
        task_id: (
            str
            | None
        ) = None,
        job_id: (
            str
            | None
        ) = None,
    ) -> (
        StructuredContextRecord
        | None
    ):

        routing_items: list[
            dict[
                str,
                Any,
            ]
        ] = []

        for item in (
            routing_context
            or []
        ):

            plain = (
                _plain(
                    item
                )
            )

            if isinstance(
                plain,
                dict,
            ):

                routing_items.append(
                    plain
                )

        trusted_results: list[
            dict[
                str,
                Any,
            ]
        ] = []

        for item in (
            previous_trusted_results
            or []
        ):

            plain = (
                _plain(
                    item
                )
            )

            if isinstance(
                plain,
                dict,
            ):

                trusted_results.append(
                    plain
                )

        return (
            record_task_context(
                self.recorder,

                subject=(
                    "hub-routing-context"
                ),

                original_request=(
                    original_request
                ),

                routing_context=(
                    routing_items
                ),

                specialist_instructions=(
                    specialist_instructions
                ),

                semantic_contract=(
                    _plain(
                        semantic_contract
                    )
                    or {}
                ),

                previous_trusted_results=(
                    trusted_results
                ),

                trajectory_id=(
                    trajectory_id
                ),

                task_id=(
                    task_id
                ),

                job_id=(
                    job_id
                ),
            )
        )

    def record_specialist_proposal(
        self,
        *,
        proposed_tool: (
            str
            | None
        ),
        proposed_arguments: (
            dict[
                str,
                Any,
            ]
            | None
        ) = None,
        proposed_tool_calls: (
            list[
                dict[
                    str,
                    Any,
                ]
            ]
            | None
        ) = None,
        raw_model_output: (
            str
            | None
        ) = None,
        trajectory_id: (
            str
            | None
        ) = None,
        task_id: (
            str
            | None
        ) = None,
        job_id: (
            str
            | None
        ) = None,
    ) -> (
        StructuredContextRecord
        | None
    ):

        calls = [
            dict(
                item
            )

            for item
            in (
                proposed_tool_calls
                or []
            )

            if isinstance(
                item,
                dict,
            )
        ]

        if (
            proposed_tool
            and not calls
        ):

            calls = [
                {
                    "name":
                        proposed_tool,

                    "arguments":
                        (
                            proposed_arguments
                            or {}
                        ),
                }
            ]

        if raw_model_output is not None:

            calls = [
                *calls,

                {
                    "raw_model_output":
                        raw_model_output,
                },
            ]

        return (
            record_tool_context(
                self.recorder,

                subject=(
                    "specialist-proposal"
                ),

                proposed_calls=(
                    calls
                ),

                trajectory_id=(
                    trajectory_id
                ),

                task_id=(
                    task_id
                ),

                job_id=(
                    job_id
                ),

                source_tool=(
                    proposed_tool
                ),
            )
        )

    def record_semantic_guard_outcome(
        self,
        *,
        allowed: bool,
        code: (
            str
            | None
        ) = None,
        detail: (
            str
            | None
        ) = None,
        trajectory_id: (
            str
            | None
        ) = None,
        task_id: (
            str
            | None
        ) = None,
        job_id: (
            str
            | None
        ) = None,
    ) -> (
        StructuredContextRecord
        | None
    ):

        return (
            record_tool_context(
                self.recorder,

                subject=(
                    "semantic-guard-outcome"
                ),

                guard_outcomes=[
                    {
                        "allowed":
                            allowed,

                        "code":
                            code,

                        "detail":
                            detail,
                    }
                ],

                trajectory_id=(
                    trajectory_id
                ),

                task_id=(
                    task_id
                ),

                job_id=(
                    job_id
                ),
            )
        )

    def record_gateway_outcome(
        self,
        *,
        tool_name: str,
        gateway_result: dict[
            str,
            Any,
        ],
        trusted_provider_result: (
            dict[
                str,
                Any,
            ]
            | None
        ) = None,
        trajectory_id: (
            str
            | None
        ) = None,
        task_id: (
            str
            | None
        ) = None,
        job_id: (
            str
            | None
        ) = None,
    ) -> (
        StructuredContextRecord
        | None
    ):

        approval = None

        if gateway_result.get(
            "approval_id"
        ):

            approval = {
                "approval_id":
                    gateway_result.get(
                        "approval_id"
                    ),

                "status":
                    gateway_result.get(
                        "status"
                    ),
            }

        return (
            record_tool_context(
                self.recorder,

                subject=(
                    f"gateway:{tool_name}"
                ),

                gateway_outcomes=[
                    dict(
                        gateway_result
                    )
                ],

                approval_outcomes=(
                    [
                        approval
                    ]
                    if approval
                    else []
                ),

                trusted_provider_results=(
                    [
                        trusted_provider_result
                    ]
                    if trusted_provider_result
                    is not None
                    else []
                ),

                trajectory_id=(
                    trajectory_id
                ),

                task_id=(
                    task_id
                ),

                job_id=(
                    job_id
                ),

                source_tool=(
                    tool_name
                ),
            )
        )

    # ========================================================
    # APPROVED MUTATION EXECUTION EVIDENCE
    # ========================================================

    def record_approval_execution(
        self,
        *,
        approval: dict[
            str,
            Any,
        ],
        provider_result: (
            dict[
                str,
                Any,
            ]
            | None
        ),
        lineage: dict[
            str,
            Any,
        ],
    ) -> list[
        StructuredContextRecord
    ]:
        """
        Record the ACTUAL execution outcome of one previously
        approved mutation.

        The original model proposal already exists in a durable
        trajectory.

        This method adds post-approval execution evidence linked to
        that exact trajectory/task/job.

        It never grants authority and never executes providers.
        """

        if not isinstance(
            approval,
            dict,
        ):

            return []

        if not isinstance(
            lineage,
            dict,
        ):

            return []

        approval_id = (
            _string(
                approval.get(
                    "id"
                )
            )
        )

        tool_name = (
            _string(
                approval.get(
                    "tool"
                )
            )
        )

        trajectory_id = (
            _string(
                lineage.get(
                    "trajectory_id"
                )
            )
        )

        task_id = (
            _string(
                lineage.get(
                    "task_id"
                )
            )
        )

        job_id = (
            _string(
                lineage.get(
                    "job_id"
                )
            )
        )

        original_request = (
            _string(
                lineage.get(
                    "user_request"
                )
            )
        )

        if (
            approval_id is None
            or tool_name is None
            or trajectory_id is None
            or task_id is None
            or job_id is None
            or original_request is None
        ):

            # Strict lineage: no orphan execution evidence.
            return []

        arguments = (
            _dict(
                approval.get(
                    "arguments"
                )
            )
            or {}
        )

        approval_status = (
            _string(
                approval.get(
                    "status"
                )
            )
        )

        risk = (
            _string(
                approval.get(
                    "risk"
                )
            )
        )

        approval_outcome: dict[
            str,
            Any,
        ] = {
            "approval_id":
                approval_id,

            "status":
                approval_status,

            "risk":
                risk,

            "replayed":
                False,
        }

        if (
            provider_result
            is not None
        ):

            provider_ok = (
                provider_result.get(
                    "ok"
                )
            )

            provider_status = (
                _string(
                    provider_result.get(
                        "status"
                    )
                )
            )

            if isinstance(
                provider_ok,
                bool,
            ):

                approval_outcome[
                    "provider_ok"
                ] = (
                    provider_ok
                )

            if provider_status is not None:

                approval_outcome[
                    "provider_status"
                ] = (
                    provider_status
                )

        records: list[
            StructuredContextRecord
        ] = []

        # ----------------------------------------------------
        # Generic actual approval-execution evidence.
        #
        # This is NOT a second gateway decision.
        # ----------------------------------------------------

        _append_record(
            records,

            record_tool_context(
                self.recorder,

                subject=(
                    f"approval-execution:{tool_name}"
                ),

                approval_outcomes=[
                    approval_outcome,
                ],

                trusted_provider_results=(
                    [
                        dict(
                            provider_result
                        )
                    ]
                    if provider_result
                    is not None
                    else []
                ),

                trajectory_id=(
                    trajectory_id
                ),

                task_id=(
                    task_id
                ),

                job_id=(
                    job_id
                ),

                source_tool=(
                    tool_name
                ),
            ),
        )

        # ----------------------------------------------------
        # Domain-specific trusted-provider context.
        # ----------------------------------------------------

        if provider_result is None:

            return records

        task_goal = (
            _string(
                lineage.get(
                    "task_instructions"
                )
            )
        )

        if (
            tool_name.startswith(
                "workspace_git_"
            )
        ):

            _append_record(
                records,

                _record_git_result(
                    recorder=(
                        self.recorder
                    ),

                    tool_name=(
                        tool_name
                    ),

                    arguments=(
                        arguments
                    ),

                    result=(
                        provider_result
                    ),

                    task_goal=(
                        task_goal
                    ),

                    previous_results=[],

                    trajectory_id=(
                        trajectory_id
                    ),

                    task_id=(
                        task_id
                    ),

                    job_id=(
                        job_id
                    ),
                ),
            )

        elif (
            tool_name.startswith(
                "ticket_"
            )
        ):

            _append_record(
                records,

                _record_jira_result(
                    recorder=(
                        self.recorder
                    ),

                    tool_name=(
                        tool_name
                    ),

                    arguments=(
                        arguments
                    ),

                    result=(
                        provider_result
                    ),

                    original_request=(
                        original_request
                    ),

                    trajectory_id=(
                        trajectory_id
                    ),

                    task_id=(
                        task_id
                    ),

                    job_id=(
                        job_id
                    ),
                ),
            )

        else:

            _append_record(
                records,

                _record_shell_result(
                    recorder=(
                        self.recorder
                    ),

                    tool_name=(
                        tool_name
                    ),

                    arguments=(
                        arguments
                    ),

                    result=(
                        provider_result
                    ),

                    task_goal=(
                        task_goal
                    ),

                    trajectory_id=(
                        trajectory_id
                    ),

                    task_id=(
                        task_id
                    ),

                    job_id=(
                        job_id
                    ),
                ),
            )

        return records


    def record_git(
        self,
        **kwargs,
    ) -> (
        StructuredContextRecord
        | None
    ):

        return (
            record_git_context(
                self.recorder,
                **kwargs,
            )
        )

    def record_jira(
        self,
        **kwargs,
    ) -> (
        StructuredContextRecord
        | None
    ):

        return (
            record_jira_context(
                self.recorder,
                **kwargs,
            )
        )

    def record_shell(
        self,
        **kwargs,
    ) -> (
        StructuredContextRecord
        | None
    ):

        return (
            record_shell_context(
                self.recorder,
                **kwargs,
            )
        )

    def record_human(
        self,
        **kwargs,
    ) -> (
        StructuredContextRecord
        | None
    ):

        return (
            record_human_context(
                self.recorder,
                **kwargs,
            )
        )

    # ========================================================
    # EXACT DECISION RECORDING
    # ========================================================

    def _record_exact_semantic_guard(
        self,
        *,
        decision: dict[
            str,
            Any,
        ],
        trajectory_id: str,
        task_id: str | None,
        job_id: str | None,
    ) -> (
        StructuredContextRecord
        | None
    ):

        allowed = (
            decision.get(
                "allowed"
            )
        )

        decision_code = (
            _string(
                decision.get(
                    "decision_code"
                )
            )
        )

        error = (
            _string(
                decision.get(
                    "error"
                )
            )
        )

        if not isinstance(
            allowed,
            bool,
        ):

            return None

        if decision_code is None:

            return None

        return (
            self.record_semantic_guard_outcome(
                allowed=(
                    allowed
                ),

                code=(
                    decision_code
                ),

                detail=(
                    error
                ),

                trajectory_id=(
                    trajectory_id
                ),

                task_id=(
                    task_id
                ),

                job_id=(
                    job_id
                ),
            )
        )

    # ========================================================
    # DURABLE TRAJECTORY -> CONTEXT BRIDGE
    # ========================================================

    def record_trajectory_context(
        self,
        trajectory: dict[
            str,
            Any,
        ],
    ) -> list[
        StructuredContextRecord
    ]:
        """
        Convert one completed durable trajectory into structured
        contextual evidence.

        New runtime:

            exact SemanticGuard decision
            exact ToolGateway decision

        Historical runtime:

            stable outcome_code reconstruction

        Exact evidence ALWAYS wins.

        This method does not reinterpret authorization and does not
        execute anything.
        """

        if not isinstance(
            trajectory,
            dict,
        ):

            raise ValueError(
                "trajectory context source "
                "must be an object."
            )

        trajectory_id = (
            _string(
                trajectory.get(
                    "trajectory_id"
                )
            )
        )

        if trajectory_id is None:

            raise ValueError(
                "trajectory context source "
                "is missing trajectory_id."
            )

        job_id = (
            _string(
                trajectory.get(
                    "job_id"
                )
            )
        )

        original_request = (
            _string(
                trajectory.get(
                    "user_request"
                )
            )
        )

        if original_request is None:

            raise ValueError(
                "trajectory context source "
                "is missing user_request."
            )

        raw_steps = (
            trajectory.get(
                "steps"
            )
        )

        if not isinstance(
            raw_steps,
            list,
        ):

            raise ValueError(
                "trajectory context source "
                "contains invalid steps."
            )

        records: list[
            StructuredContextRecord
        ] = []

        previous_trusted_results: list[
            dict[
                str,
                Any,
            ]
        ] = []

        for (
            step_index,
            raw_step,
        ) in enumerate(
            raw_steps
        ):

            if not isinstance(
                raw_step,
                dict,
            ):

                continue

            step = (
                dict(
                    raw_step
                )
            )

            task_id = (
                _string(
                    step.get(
                        "task_id"
                    )
                )
            )

            agent_name = (
                _string(
                    step.get(
                        "agent"
                    )
                )
            )

            instructions = (
                _string(
                    step.get(
                        "task_instructions"
                    )
                )
            )

            semantic_intent = (
                _dict(
                    step.get(
                        "semantic_intent"
                    )
                )
                or {}
            )

            proposed_tool = (
                _string(
                    step.get(
                        "proposed_tool"
                    )
                )
            )

            proposed_arguments = (
                _dict(
                    step.get(
                        "proposed_arguments"
                    )
                )
                or {}
            )

            proposed_tool_calls = (
                _dict_list(
                    step.get(
                        "proposed_tool_calls"
                    )
                )
            )

            raw_model_output = (
                _string(
                    step.get(
                        "raw_model_output"
                    )
                )
            )

            outcome_code = (
                _string(
                    step.get(
                        "outcome_code"
                    )
                )
            )

            status = (
                _string(
                    step.get(
                        "status"
                    )
                )
            )

            error = (
                _string(
                    step.get(
                        "error"
                    )
                )
            )

            approval_id = (
                _string(
                    step.get(
                        "approval_id"
                    )
                )
            )

            tool_result = (
                _dict(
                    step.get(
                        "tool_result"
                    )
                )
            )

            # NEW 5L.2 EVIDENCE
            exact_semantic_guard = (
                _dict(
                    step.get(
                        "semantic_guard_decision"
                    )
                )
            )

            exact_gateway = (
                _dict(
                    step.get(
                        "gateway_decision"
                    )
                )
            )

            # ------------------------------------------------
            # TASK / HUB CONTEXT
            # ------------------------------------------------

            _append_record(
                records,

                self.record_hub_context(
                    original_request=(
                        original_request
                    ),

                    routing_context=[
                        {
                            "route_index":
                                step_index,

                            "agent":
                                agent_name,
                        }
                    ],

                    semantic_contract=(
                        semantic_intent
                    ),

                    specialist_instructions=(
                        instructions
                    ),

                    previous_trusted_results=(
                        previous_trusted_results
                    ),

                    trajectory_id=(
                        trajectory_id
                    ),

                    task_id=(
                        task_id
                    ),

                    job_id=(
                        job_id
                    ),
                ),
            )

            # ------------------------------------------------
            # SPECIALIST PROPOSAL
            # ------------------------------------------------

            if (
                proposed_tool is not None
                or proposed_tool_calls
                or raw_model_output is not None
            ):

                _append_record(
                    records,

                    self.record_specialist_proposal(
                        proposed_tool=(
                            proposed_tool
                        ),

                        proposed_arguments=(
                            proposed_arguments
                        ),

                        proposed_tool_calls=(
                            proposed_tool_calls
                        ),

                        raw_model_output=(
                            raw_model_output
                        ),

                        trajectory_id=(
                            trajectory_id
                        ),

                        task_id=(
                            task_id
                        ),

                        job_id=(
                            job_id
                        ),
                    ),
                )

            # ====================================================
            # SEMANTIC GUARD
            #
            # Exact live evidence wins.
            #
            # Historical fallback is used only when the exact
            # decision field is absent.
            # ====================================================

            if (
                exact_semantic_guard
                is not None
            ):

                _append_record(
                    records,

                    self._record_exact_semantic_guard(
                        decision=(
                            exact_semantic_guard
                        ),

                        trajectory_id=(
                            trajectory_id
                        ),

                        task_id=(
                            task_id
                        ),

                        job_id=(
                            job_id
                        ),
                    ),
                )

            elif (
                outcome_code is not None
                and outcome_code.startswith(
                    "semantic_"
                )
            ):

                # Historical negative-only reconstruction.
                _append_record(
                    records,

                    self.record_semantic_guard_outcome(
                        allowed=False,

                        code=(
                            outcome_code
                        ),

                        detail=(
                            error
                        ),

                        trajectory_id=(
                            trajectory_id
                        ),

                        task_id=(
                            task_id
                        ),

                        job_id=(
                            job_id
                        ),
                    ),
                )

            # ====================================================
            # TOOL GATEWAY
            #
            # Exact live gateway decision wins.
            #
            # Historical reconstruction remains for older
            # trajectories.
            # ====================================================

            if (
                exact_gateway
                is not None
                and proposed_tool
                is not None
            ):

                _append_record(
                    records,

                    self.record_gateway_outcome(
                        tool_name=(
                            proposed_tool
                        ),

                        gateway_result=(
                            exact_gateway
                        ),

                        trusted_provider_result=(
                            tool_result
                        ),

                        trajectory_id=(
                            trajectory_id
                        ),

                        task_id=(
                            task_id
                        ),

                        job_id=(
                            job_id
                        ),
                    ),
                )

            elif (
                exact_gateway
                is None
                and proposed_tool
                is not None
                and (
                    outcome_code
                    not in PRE_GATEWAY_OUTCOME_CODES
                )
                and not (
                    outcome_code is not None
                    and outcome_code.startswith(
                        "semantic_"
                    )
                )
            ):

                # --------------------------------------------
                # HISTORICAL FALLBACK
                # --------------------------------------------

                gateway_evidence: dict[
                    str,
                    Any,
                ] = {}

                if status is not None:

                    gateway_evidence[
                        "status"
                    ] = (
                        status
                    )

                if outcome_code is not None:

                    gateway_evidence[
                        "decision_code"
                    ] = (
                        outcome_code
                    )

                if approval_id is not None:

                    gateway_evidence[
                        "approval_id"
                    ] = (
                        approval_id
                    )

                if status == "success":

                    gateway_evidence[
                        "ok"
                    ] = True

                elif status == "error":

                    gateway_evidence[
                        "ok"
                    ] = False

                _append_record(
                    records,

                    self.record_gateway_outcome(
                        tool_name=(
                            proposed_tool
                        ),

                        gateway_result=(
                            gateway_evidence
                        ),

                        trusted_provider_result=(
                            tool_result
                        ),

                        trajectory_id=(
                            trajectory_id
                        ),

                        task_id=(
                            task_id
                        ),

                        job_id=(
                            job_id
                        ),
                    ),
                )

            # ====================================================
            # DOMAIN-SPECIFIC TRUSTED CONTEXT
            # ====================================================

            if (
                proposed_tool is not None
                and tool_result is not None
            ):

                if proposed_tool.startswith(
                    "workspace_git_"
                ):

                    _append_record(
                        records,

                        _record_git_result(
                            recorder=(
                                self.recorder
                            ),

                            tool_name=(
                                proposed_tool
                            ),

                            arguments=(
                                proposed_arguments
                            ),

                            result=(
                                tool_result
                            ),

                            task_goal=(
                                instructions
                            ),

                            previous_results=(
                                previous_trusted_results
                            ),

                            trajectory_id=(
                                trajectory_id
                            ),

                            task_id=(
                                task_id
                            ),

                            job_id=(
                                job_id
                            ),
                        ),
                    )

                elif proposed_tool.startswith(
                    "ticket_"
                ):

                    _append_record(
                        records,

                        _record_jira_result(
                            recorder=(
                                self.recorder
                            ),

                            tool_name=(
                                proposed_tool
                            ),

                            arguments=(
                                proposed_arguments
                            ),

                            result=(
                                tool_result
                            ),

                            original_request=(
                                original_request
                            ),

                            trajectory_id=(
                                trajectory_id
                            ),

                            task_id=(
                                task_id
                            ),

                            job_id=(
                                job_id
                            ),
                        ),
                    )

                else:

                    _append_record(
                        records,

                        _record_shell_result(
                            recorder=(
                                self.recorder
                            ),

                            tool_name=(
                                proposed_tool
                            ),

                            arguments=(
                                proposed_arguments
                            ),

                            result=(
                                tool_result
                            ),

                            task_goal=(
                                instructions
                            ),

                            trajectory_id=(
                                trajectory_id
                            ),

                            task_id=(
                                task_id
                            ),

                            job_id=(
                                job_id
                            ),
                        ),
                    )

            # ====================================================
            # CROSS-STEP TRUSTED RESULT MEMORY
            #
            # Only actual provider results enter subsequent task
            # context.
            # ====================================================

            if tool_result is not None:

                previous_trusted_results.append(
                    {
                        "task_id":
                            task_id,

                        "agent":
                            agent_name,

                        "tool":
                            proposed_tool,

                        "result":
                            tool_result,
                    }
                )

        return (
            records
        )


# Backward-compatible generated-addon alias.
ContinualLearningRuntimeHook = (
    ContinualLearningRuntimeHooks
)