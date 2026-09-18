from pathlib import (
    Path,
)

from services.assets import (
    AssetSearchQuery,
    MockAssetService,
    build_asset_mutation_service,
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

from tools.assets.mcp import (
    register_asset_tools,
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


def asset_agent():

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
            "asset-specialist"
        ]
    )


def test_asset_specialist_exposes_asset_capabilities():

    agent = (
        asset_agent()
    )

    assert {
        "asset_get",
        "asset_search",
        "asset_assign",
        "asset_unassign",
    }.issubset(
        set(
            agent.tools
        )
    )


def test_asset_catalog_policy():

    read_tools = {
        "asset_get":
            [
                "asset_id",
            ],

        "asset_search":
            [
                "text",
                "owner_id",
                "asset_type",
                "status",
            ],
    }

    mutation_tools = {
        "asset_assign":
            [
                "asset_id",
                "user_id",
            ],

        "asset_unassign":
            [
                "asset_id",
            ],
    }

    for (
        name,
        grounded_arguments,
    ) in read_tools.items():

        tool = (
            get_tool(
                name
            )
        )

        assert (
            tool
            is not None
        )

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

    for (
        name,
        grounded_arguments,
    ) in mutation_tools.items():

        tool = (
            get_tool(
                name
            )
        )

        assert (
            tool
            is not None
        )

        assert (
            tool[
                "risk"
            ]
            == "medium"
        )

        assert (
            tool[
                "requires_approval"
            ]
            is True
        )

        assert (
            tool[
                "grounded_arguments"
            ]
            == grounded_arguments
        )


def test_asset_get_returns_exact_asset():

    service = (
        MockAssetService()
    )

    result = (
        service
        .get_asset(
            "lap-001"
        )
    )

    assert result.ok is True
    assert result.status == "success"

    assert (
        result.asset
        is not None
    )

    assert (
        result.asset.asset_id
        == "LAP-001"
    )

    assert (
        result.asset.owner_id
        == "jdoe"
    )


def test_asset_search_by_owner():

    service = (
        MockAssetService()
    )

    result = (
        service
        .search_assets(
            AssetSearchQuery(
                owner_id=(
                    "jdoe"
                )
            )
        )
    )

    assert result.ok is True
    assert result.count == 1

    assert (
        result.assets[
            0
        ].asset_id
        == "LAP-001"
    )


def test_asset_assignment_updates_shared_state():

    service = (
        MockAssetService()
    )

    mutations = (
        build_asset_mutation_service(
            service
        )
    )

    before = (
        service
        .get_asset(
            "LAP-002"
        )
    )

    assert (
        before.asset
        is not None
    )

    assert (
        before.asset.owner_id
        is None
    )

    result = (
        mutations
        .assign_asset(
            "LAP-002",
            "jdoe",
        )
    )

    assert result.ok is True
    assert result.status == "executed"
    assert result.changed is True
    assert result.owner_id == "jdoe"

    after = (
        service
        .get_asset(
            "LAP-002"
        )
    )

    assert (
        after.asset
        is not None
    )

    assert (
        after.asset.owner_id
        == "jdoe"
    )

    assert (
        after.asset.status
        == "in_use"
    )


def test_asset_unassign_updates_shared_state():

    service = (
        MockAssetService()
    )

    mutations = (
        build_asset_mutation_service(
            service
        )
    )

    result = (
        mutations
        .unassign_asset(
            "LAP-001"
        )
    )

    assert result.ok is True
    assert result.status == "executed"
    assert result.changed is True
    assert result.owner_id is None

    after = (
        service
        .get_asset(
            "LAP-001"
        )
    )

    assert (
        after.asset
        is not None
    )

    assert (
        after.asset.owner_id
        is None
    )

    assert (
        after.asset.status
        == "available"
    )


def test_repair_asset_cannot_be_assigned():

    service = (
        MockAssetService()
    )

    mutations = (
        build_asset_mutation_service(
            service
        )
    )

    result = (
        mutations
        .assign_asset(
            "LAP-003",
            "jdoe",
        )
    )

    assert result.ok is False
    assert result.status == "denied"
    assert result.changed is False


def test_asset_mcp_registration_and_execution():

    server = (
        FakeMCPServer()
    )

    service = (
        MockAssetService()
    )

    mutations = (
        build_asset_mutation_service(
            service
        )
    )

    register_asset_tools(
        server,
        service,
        mutations,
    )

    assert (
        set(
            server.functions
        )
        == {
            "asset_get",
            "asset_search",
            "asset_assign",
            "asset_unassign",
        }
    )

    result = (
        server.functions[
            "asset_get"
        ](
            asset_id=(
                "LAP-001"
            )
        )
    )

    assert result.ok is True

    assert (
        result.asset
        is not None
    )

    assert (
        result.asset.asset_id
        == "LAP-001"
    )


def test_semantic_guard_allows_exact_asset_assignment():

    agent = (
        asset_agent()
    )

    intent = (
        SemanticIntent(
            summary=(
                "Assign LAP-002 to jdoe."
            ),

            effect="mutation",

            allowed_tools=[
                "asset_assign",
            ],

            forbidden_tools=[
                "asset_unassign",
            ],

            allowed_arguments={
                "asset_id": [
                    "LAP-002",
                ],

                "user_id": [
                    "jdoe",
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
                "asset_assign"
            ),

            arguments={
                "asset_id":
                    "LAP-002",

                "user_id":
                    "jdoe",
            },
        )
    )

    assert decision.allowed is True

    assert (
        decision.decision_code
        == "semantic_guard_allowed"
    )


def test_semantic_guard_blocks_wrong_asset_owner():

    agent = (
        asset_agent()
    )

    intent = (
        SemanticIntent(
            summary=(
                "Assign LAP-002 to jdoe, not asmith."
            ),

            effect="mutation",

            allowed_tools=[
                "asset_assign",
            ],

            forbidden_tools=[],

            allowed_arguments={
                "asset_id": [
                    "LAP-002",
                ],

                "user_id": [
                    "jdoe",
                ],
            },

            forbidden_arguments={
                "user_id": [
                    "asmith",
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
                "asset_assign"
            ),

            arguments={
                "asset_id":
                    "LAP-002",

                "user_id":
                    "asmith",
            },
        )
    )

    assert decision.allowed is False

    assert (
        decision.decision_code
        == "semantic_argument_forbidden"
    )
