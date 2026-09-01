from pathlib import Path

import pytest

from subagents.core.llm_router import LLMRouter
from subagents.core.loader import load_agent_directory
from subagents.core.registry import AgentRegistry
from subagents.llm.qwen_hub import QwenHubBackend


HUB_MODEL_PATH = Path(
    "/mnt/c/project/agenticaiPersonal/Models/Qwen3-0.6B"
)


@pytest.fixture(scope="module")
def router():
    agent_registry = AgentRegistry()

    agents = load_agent_directory(
        "subagents/agents"
    )

    agent_registry.register_many(
        agents
    )

    backend = QwenHubBackend(
        HUB_MODEL_PATH
    )

    return LLMRouter(
        registry=agent_registry,
        backend=backend,
    )


def test_real_hub_routes_account_request(router):
    routes = router.route(
        "Is jdoe locked?"
    )

    print(
        "\nREQUEST: Is jdoe locked?"
        f"\nROUTES: {routes}"
    )

    assert routes == [
        "account-specialist"
    ]


def test_real_hub_routes_access_request(router):
    routes = router.route(
        "Does jdoe have VPN access?"
    )

    print(
        "\nREQUEST: Does jdoe have VPN access?"
        f"\nROUTES: {routes}"
    )

    assert routes == [
        "access-specialist"
    ]


def test_real_hub_routes_multi_specialist_request(
    router,
):
    routes = router.route(
        "Check whether jdoe is locked "
        "and whether jdoe has VPN access."
    )

    print(
        "\nREQUEST: Check account and VPN access"
        f"\nROUTES: {routes}"
    )

    assert routes == [
        "account-specialist",
        "access-specialist",
    ]


def test_real_hub_returns_no_route(router):
    routes = router.route(
        "Tell me a joke."
    )

    print(
        "\nREQUEST: Tell me a joke."
        f"\nROUTES: {routes}"
    )

    assert routes == []