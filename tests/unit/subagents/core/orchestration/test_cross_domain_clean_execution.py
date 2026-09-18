import asyncio

from subagents.core.definitions.types import (
    AgentResult,
    SemanticIntent,
    SpecialistRequest,
)

from subagents.core.orchestration.orchestrator import (
    Orchestrator,
)


REQUEST = (
    "jdoe cannot connect to VPN. "
    "Check the account, verify VPN access, "
    "find relevant guidance, and tell me what "
    "should happen next. Do not change anything."
)


class FakeRouter:

    def __init__(
        self,
    ) -> None:

        self.requests = []

        self.delegations = [
            SpecialistRequest(
                agent_name=(
                    "account-specialist"
                ),

                instructions=(
                    "Check jdoe's current account state "
                    "without changing it."
                ),

                semantic_intent=(
                    SemanticIntent(
                        summary=(
                            "Check jdoe's account status "
                            "without changing it."
                        ),

                        effect="read",

                        allowed_tools=[
                            "account_status",
                        ],

                        forbidden_tools=[
                            "unlock_user",
                            "reset_password",
                            "enable_user",
                            "disable_user",
                        ],

                        allowed_arguments={
                            "user_id": [
                                "jdoe",
                            ],
                        },

                        forbidden_arguments={},

                        max_tool_calls=1,

                        clarification_required=False,
                    )
                ),
            ),

            SpecialistRequest(
                agent_name=(
                    "access-specialist"
                ),

                instructions=(
                    "Check whether jdoe currently has "
                    "VPN access without changing access."
                ),

                semantic_intent=(
                    SemanticIntent(
                        summary=(
                            "Check jdoe's VPN access "
                            "without changing it."
                        ),

                        effect="read",

                        allowed_tools=[
                            "check_access",
                        ],

                        forbidden_tools=[
                            "grant_access",
                            "revoke_access",
                        ],

                        allowed_arguments={
                            "user_id": [
                                "jdoe",
                            ],

                            "resource": [
                                "VPN",
                            ],
                        },

                        forbidden_arguments={},

                        max_tool_calls=1,

                        clarification_required=False,
                    )
                ),
            ),

            SpecialistRequest(
                agent_name=(
                    "knowledge-specialist"
                ),

                instructions=(
                    "Find relevant trusted guidance for "
                    "VPN connectivity troubleshooting."
                ),

                semantic_intent=(
                    SemanticIntent(
                        summary=(
                            "Search trusted knowledge for "
                            "VPN connectivity guidance."
                        ),

                        effect="read",

                        allowed_tools=[
                            "knowledge_search",
                        ],

                        forbidden_tools=[],

                        allowed_arguments={},

                        forbidden_arguments={},

                        max_tool_calls=1,

                        clarification_required=False,
                    )
                ),
            ),
        ]

    async def route(
        self,
        user_request: str,
    ):

        self.requests.append(
            user_request
        )

        return (
            self.delegations
        )


class FakeRuntime:

    def __init__(
        self,
    ) -> None:

        self.tasks = []

    async def run(
        self,
        task,
    ) -> AgentResult:

        self.tasks.append(
            task
        )

        if (
            task.agent_name
            == "account-specialist"
        ):

            return (
                AgentResult(
                    task_id=(
                        task.task_id
                    ),

                    agent_name=(
                        task.agent_name
                    ),

                    status="success",

                    proposed_tool=(
                        "account_status"
                    ),

                    proposed_arguments={
                        "user_id":
                            "jdoe",
                    },

                    outcome_code=(
                        "success"
                    ),

                    tool_result={
                        "ok":
                            True,

                        "user_id":
                            "jdoe",

                        "enabled":
                            True,

                        "locked":
                            False,
                    },

                    answer=(
                        "jdoe is enabled and is not locked."
                    ),
                )
            )

        if (
            task.agent_name
            == "access-specialist"
        ):

            return (
                AgentResult(
                    task_id=(
                        task.task_id
                    ),

                    agent_name=(
                        task.agent_name
                    ),

                    status="success",

                    proposed_tool=(
                        "check_access"
                    ),

                    proposed_arguments={
                        "user_id":
                            "jdoe",

                        "resource":
                            "VPN",
                    },

                    outcome_code=(
                        "success"
                    ),

                    tool_result={
                        "ok":
                            True,

                        "user_id":
                            "jdoe",

                        "resource":
                            "vpn",

                        "has_access":
                            True,
                    },

                    answer=(
                        "jdoe currently has VPN access."
                    ),
                )
            )

        if (
            task.agent_name
            == "knowledge-specialist"
        ):

            return (
                AgentResult(
                    task_id=(
                        task.task_id
                    ),

                    agent_name=(
                        task.agent_name
                    ),

                    status="success",

                    proposed_tool=(
                        "knowledge_search"
                    ),

                    proposed_arguments={
                        "query":
                            "VPN connectivity troubleshooting",
                    },

                    outcome_code=(
                        "success"
                    ),

                    tool_result={
                        "ok":
                            True,

                        "status":
                            "success",

                        "hits": [
                            {
                                "document_id":
                                    "KB-001",

                                "kind":
                                    "knowledge",

                                "title":
                                    "VPN access troubleshooting",
                            },

                            {
                                "document_id":
                                    "RB-001",

                                "kind":
                                    "runbook",

                                "title":
                                    "VPN access restoration triage",
                            },
                        ],
                    },

                    answer=(
                        "Relevant VPN troubleshooting "
                        "guidance was found."
                    ),
                )
            )

        raise AssertionError(
            "Unexpected specialist: "
            f"{task.agent_name}"
        )


class FakePrimaryAssistant:

    def __init__(
        self,
    ) -> None:

        self.respond_calls = []
        self.synthesis_calls = []

    async def respond(
        self,
        user_request: str,
    ) -> str:

        self.respond_calls.append(
            user_request
        )

        return (
            "Unexpected direct response."
        )

    async def synthesize(
        self,
        user_request: str,
        results,
    ) -> str:

        self.synthesis_calls.append(
            {
                "user_request":
                    user_request,

                "results":
                    results,
            }
        )

        return (
            "jdoe's account is enabled and unlocked, "
            "VPN access is already present, and trusted "
            "VPN troubleshooting guidance is available. "
            "No state-changing action was performed."
        )


def build_orchestrator():

    router = (
        FakeRouter()
    )

    runtime = (
        FakeRuntime()
    )

    primary = (
        FakePrimaryAssistant()
    )

    orchestrator = (
        Orchestrator(
            router=router,

            runtime=runtime,

            primary_assistant=primary,

            require_semantic_intent=True,
        )
    )

    return (
        orchestrator,
        router,
        runtime,
        primary,
    )


def test_cross_domain_read_only_request_runs_all_specialists():

    (
        orchestrator,
        router,
        runtime,
        primary,
    ) = (
        build_orchestrator()
    )

    result = (
        asyncio.run(
            orchestrator.run(
                REQUEST
            )
        )
    )

    assert (
        result.status
        == "success"
    )

    assert (
        result.routes
        == [
            "account-specialist",
            "access-specialist",
            "knowledge-specialist",
        ]
    )

    assert (
        len(
            result.results
        )
        == 3
    )

    assert (
        len(
            runtime.tasks
        )
        == 3
    )

    assert (
        router.requests
        == [
            REQUEST,
        ]
    )


def test_cross_domain_tasks_preserve_read_only_semantics():

    (
        orchestrator,
        _router,
        runtime,
        _primary,
    ) = (
        build_orchestrator()
    )

    asyncio.run(
        orchestrator.run(
            REQUEST
        )
    )

    assert (
        len(
            runtime.tasks
        )
        == 3
    )

    for task in (
        runtime.tasks
    ):

        assert (
            task.semantic_intent
            is not None
        )

        assert (
            task.semantic_intent.effect
            == "read"
        )

        assert (
            task.semantic_intent.max_tool_calls
            == 1
        )

        assert (
            task.semantic_intent
            .clarification_required
            is False
        )


def test_cross_domain_read_only_flow_contains_no_mutation_proposal():

    (
        orchestrator,
        _router,
        _runtime,
        _primary,
    ) = (
        build_orchestrator()
    )

    result = (
        asyncio.run(
            orchestrator.run(
                REQUEST
            )
        )
    )

    proposed_tools = {
        item.proposed_tool

        for item
        in result.results
    }

    assert (
        proposed_tools
        == {
            "account_status",
            "check_access",
            "knowledge_search",
        }
    )

    mutation_tools = {
        "unlock_user",
        "reset_password",
        "enable_user",
        "disable_user",
        "grant_access",
        "revoke_access",
        "asset_assign",
        "asset_unassign",
        "ticket_add_comment",
        "ticket_create",
        "ticket_assign",
        "ticket_transition",
    }

    assert (
        proposed_tools
        .isdisjoint(
            mutation_tools
        )
    )


def test_cross_domain_success_synthesizes_once():

    (
        orchestrator,
        _router,
        _runtime,
        primary,
    ) = (
        build_orchestrator()
    )

    result = (
        asyncio.run(
            orchestrator.run(
                REQUEST
            )
        )
    )

    assert (
        len(
            primary.synthesis_calls
        )
        == 1
    )

    assert (
        primary.respond_calls
        == []
    )

    synthesis = (
        primary.synthesis_calls[
            0
        ]
    )

    assert (
        synthesis[
            "user_request"
        ]
        == REQUEST
    )

    assert (
        len(
            synthesis[
                "results"
            ]
        )
        == 3
    )

    assert (
        "No state-changing action was performed."
        in result.answer
    )


def test_runbook_retrieval_does_not_create_execution_authority():

    (
        orchestrator,
        _router,
        runtime,
        _primary,
    ) = (
        build_orchestrator()
    )

    result = (
        asyncio.run(
            orchestrator.run(
                REQUEST
            )
        )
    )

    assert (
        all(
            task.semantic_intent is not None
            and task.semantic_intent.effect
            == "read"

            for task
            in runtime.tasks
        )
    )

    assert (
        not any(
            item.status
            == "approval_required"

            for item
            in result.results
        )
    )

    assert (
        not any(
            item.proposed_tool
            in {
                "grant_access",
                "revoke_access",
                "unlock_user",
                "enable_user",
                "disable_user",
                "reset_password",
            }

            for item
            in result.results
        )
    )
