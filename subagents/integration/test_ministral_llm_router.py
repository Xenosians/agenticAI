import pytest

from config import (
    get_settings,
)

from subagents.core.loader import (
    load_agent_directory,
)

from subagents.core.llm_router import (
    LLMRouter,
)

from subagents.core.registry import (
    AgentRegistry,
)

from subagents.llm.ministral_hub import (
    MinistralHubBackend,
)


@pytest.fixture(
    scope="module"
)
def router():
    settings = get_settings()

    registry = (
        AgentRegistry()
    )

    agents_dir = (
        settings.require_path(
            settings.agents_dir,
            "AGENTS_DIR",
        )
    )

    agents = (
        load_agent_directory(
            agents_dir
        )
    )

    registry.register_many(
        agents
    )

    model_path = (
        settings.require_path(
            settings.hub_model_path,
            "HUB_MODEL_PATH",
        )
    )

    backend = (
        MinistralHubBackend(
            model_path=(
                model_path
            ),

            dequantize_fp8=(
                settings
                .hub_dequantize_fp8
            ),

            offload_folder=(
                settings
                .hub_offload_folder
            ),
        )
    )

    return LLMRouter(
        registry=registry,
        backend=backend,
    )


def test_ministral_routes_account(
    router,
):
    routes = router.route(
        "Is jdoe locked?"
    )

    print(
        "ACCOUNT:",
        routes,
    )

    assert routes == [
        "account-specialist"
    ]


def test_ministral_routes_access(
    router,
):
    routes = router.route(
        "Does jdoe have VPN access?"
    )

    print(
        "ACCESS:",
        routes,
    )

    assert routes == [
        "access-specialist"
    ]


def test_ministral_routes_multiple(
    router,
):
    routes = router.route(
        (
            "Check whether jdoe is locked "
            "and whether jdoe has VPN access."
        )
    )

    print(
        "MULTI:",
        routes,
    )

    assert routes == [
        "account-specialist",
        "access-specialist",
    ]


def test_ministral_no_route(
    router,
):
    routes = router.route(
        "Tell me a joke."
    )

    print(
        "NO ROUTE:",
        routes,
    )

    assert (
        routes
        == []
    )