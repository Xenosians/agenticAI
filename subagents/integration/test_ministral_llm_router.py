import asyncio

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

from subagents.llm.inference import (
    InferenceCoordinator,
)

from subagents.llm.model_manager import (
    ModelManager,
)

from subagents.llm.scheduler import (
    GpuScheduler,
)


@pytest.fixture(
    scope="module"
)
def router_runtime():
    settings = (
        get_settings()
    )

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

    model_manager = (
        ModelManager(
            settings=(
                settings
            )
        )
    )

    scheduler = (
        GpuScheduler()
    )

    inference = (
        InferenceCoordinator(
            model_manager=(
                model_manager
            ),
            scheduler=(
                scheduler
            ),
        )
    )

    router = (
        LLMRouter(
            registry=(
                registry
            ),
            inference=(
                inference
            ),
            model_key=(
                settings
                .hub_model_key
            ),
        )
    )

    runner = (
        asyncio.Runner()
    )

    try:
        runner.run(
            inference.warm(
                settings
                .hub_model_key
            )
        )

        yield (
            runner,
            router,
        )

    finally:
        runner.close()


def test_ministral_routes_account(
    router_runtime,
):
    (
        runner,
        router,
    ) = router_runtime

    routes = (
        runner.run(
            router.route(
                "Is jdoe locked?"
            )
        )
    )

    print(
        "ACCOUNT:",
        routes,
    )

    assert routes == [
        "account-specialist"
    ]


def test_ministral_routes_access(
    router_runtime,
):
    (
        runner,
        router,
    ) = router_runtime

    routes = (
        runner.run(
            router.route(
                "Does jdoe have VPN access?"
            )
        )
    )

    print(
        "ACCESS:",
        routes,
    )

    assert routes == [
        "access-specialist"
    ]


def test_ministral_routes_multiple(
    router_runtime,
):
    (
        runner,
        router,
    ) = router_runtime

    routes = (
        runner.run(
            router.route(
                (
                    "Check whether jdoe is locked "
                    "and whether jdoe has VPN access."
                )
            )
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
    router_runtime,
):
    (
        runner,
        router,
    ) = router_runtime

    routes = (
        runner.run(
            router.route(
                "Tell me a joke."
            )
        )
    )

    print(
        "NO ROUTE:",
        routes,
    )

    assert (
        routes
        == []
    )