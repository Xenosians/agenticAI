from pathlib import (
    Path,
)

from services.knowledge import (
    KnowledgeSearchQuery,
    MockKnowledgeService,
)

from subagents.core.definitions.loader import (
    load_agent_directory,
)

from subagents.core.definitions.types import (
    SemanticIntent,
)

from subagents.core.orchestration.semantic_guard import (
    SemanticGuard,
)

from tools.knowledge.mcp import (
    register_knowledge_tools,
)

from tools.registry import (
    get_tool,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[4]
)

AGENTS_DIR = (
    PROJECT_ROOT
    / "subagents"
    / "agents"
)


class FakeMCPServer:

    def __init__(
        self,
    ) -> None:

        self.functions = {}

    def tool(
        self,
    ):

        def decorator(
            function,
        ):

            self.functions[
                function.__name__
            ] = (
                function
            )

            return (
                function
            )

        return decorator


def knowledge_agent():

    agents = {
        agent.name:
            agent

        for agent
        in load_agent_directory(
            AGENTS_DIR
        )
    }

    return (
        agents[
            "knowledge-specialist"
        ]
    )


def test_knowledge_specialist_exposes_read_capabilities():

    agent = (
        knowledge_agent()
    )

    assert {
        "knowledge_search",
        "knowledge_get",
        "runbook_get",
    }.issubset(
        set(
            agent.tools
        )
    )


def test_knowledge_catalog_is_read_only():

    expected = {
        "knowledge_search":
            [],

        "knowledge_get":
            [
                "document_id",
            ],

        "runbook_get":
            [
                "runbook_id",
            ],
    }

    for (
        name,
        grounded_arguments,
    ) in expected.items():

        tool = (
            get_tool(
                name
            )
        )

        assert tool is not None

        assert (
            tool[
                "risk"
            ]
            == "read"
        )

        assert (
            tool[
                "requires_approval"
            ]
            is False
        )

        assert (
            tool[
                "grounded_arguments"
            ]
            == grounded_arguments
        )


def test_search_finds_vpn_knowledge_and_runbook():

    service = (
        MockKnowledgeService()
    )

    result = (
        service
        .search(
            KnowledgeSearchQuery(
                query=(
                    "VPN access"
                )
            )
        )
    )

    assert result.ok is True
    assert result.status == "success"

    document_ids = {
        hit.document_id

        for hit
        in result.hits
    }

    assert (
        "KB-001"
        in document_ids
    )

    assert (
        "RB-001"
        in document_ids
    )


def test_search_can_filter_runbooks():

    service = (
        MockKnowledgeService()
    )

    result = (
        service
        .search(
            KnowledgeSearchQuery(
                query=(
                    "account lockout"
                ),

                kind="runbook",
            )
        )
    )

    assert result.ok is True

    assert all(
        hit.kind
        == "runbook"

        for hit
        in result.hits
    )

    document_ids = {
        hit.document_id

        for hit
        in result.hits
    }

    assert (
        "RB-002"
        in document_ids
    )


def test_get_exact_knowledge_article():

    service = (
        MockKnowledgeService()
    )

    result = (
        service
        .get_knowledge(
            "kb-002"
        )
    )

    assert result.ok is True
    assert result.status == "success"

    assert (
        result.document
        is not None
    )

    assert (
        result.document.document_id
        == "KB-002"
    )

    assert (
        result.document.kind
        == "knowledge"
    )


def test_knowledge_get_does_not_return_runbook():

    service = (
        MockKnowledgeService()
    )

    result = (
        service
        .get_knowledge(
            "RB-001"
        )
    )

    assert result.ok is False
    assert result.status == "not_found"


def test_get_exact_runbook():

    service = (
        MockKnowledgeService()
    )

    result = (
        service
        .get_runbook(
            "rb-001"
        )
    )

    assert result.ok is True
    assert result.status == "success"

    assert (
        result.document
        is not None
    )

    assert (
        result.document.document_id
        == "RB-001"
    )

    assert (
        result.document.kind
        == "runbook"
    )


def test_runbook_get_does_not_return_knowledge_article():

    service = (
        MockKnowledgeService()
    )

    result = (
        service
        .get_runbook(
            "KB-001"
        )
    )

    assert result.ok is False
    assert result.status == "not_found"


def test_knowledge_mcp_registration():

    server = (
        FakeMCPServer()
    )

    service = (
        MockKnowledgeService()
    )

    register_knowledge_tools(
        server,
        service,
    )

    assert (
        set(
            server.functions
        )
        == {
            "knowledge_search",
            "knowledge_get",
            "runbook_get",
        }
    )

    result = (
        server.functions[
            "runbook_get"
        ](
            runbook_id=(
                "RB-001"
            )
        )
    )

    assert result.ok is True

    assert (
        result.document
        is not None
    )

    assert (
        result.document.document_id
        == "RB-001"
    )


def test_semantic_guard_allows_exact_runbook():

    agent = (
        knowledge_agent()
    )

    intent = (
        SemanticIntent(
            summary=(
                "Retrieve runbook RB-001."
            ),

            effect="read",

            allowed_tools=[
                "runbook_get",
            ],

            forbidden_tools=[],

            allowed_arguments={
                "runbook_id": [
                    "RB-001",
                ],
            },

            forbidden_arguments={},

            max_tool_calls=1,

            clarification_required=False,
        )
    )

    decision = (
        SemanticGuard()
        .evaluate(
            agent=agent,

            intent=intent,

            tool_name=(
                "runbook_get"
            ),

            arguments={
                "runbook_id":
                    "RB-001",
            },
        )
    )

    assert decision.allowed is True

    assert (
        decision.decision_code
        == "semantic_guard_allowed"
    )


def test_semantic_guard_blocks_wrong_document_id():

    agent = (
        knowledge_agent()
    )

    intent = (
        SemanticIntent(
            summary=(
                "Retrieve KB-001, not KB-002."
            ),

            effect="read",

            allowed_tools=[
                "knowledge_get",
            ],

            forbidden_tools=[],

            allowed_arguments={
                "document_id": [
                    "KB-001",
                ],
            },

            forbidden_arguments={
                "document_id": [
                    "KB-002",
                ],
            },

            max_tool_calls=1,

            clarification_required=False,
        )
    )

    decision = (
        SemanticGuard()
        .evaluate(
            agent=agent,

            intent=intent,

            tool_name=(
                "knowledge_get"
            ),

            arguments={
                "document_id":
                    "KB-002",
            },
        )
    )

    assert decision.allowed is False

    assert (
        decision.decision_code
        == "semantic_argument_forbidden"
    )
