from pathlib import Path

import pytest

from subagents.core.loader import load_agent_directory
from subagents.core.llm_router import LLMRouter
from subagents.core.registry import AgentRegistry
from subagents.llm.ministral_hub import MinistralHubBackend


MODEL_PATH = Path(
    "/mnt/c/project/agenticaiPersonal/Models/"
    "Ministral-3-3B-Instruct-2512"
)

AGENTS_DIR = Path(
    "subagents/agents"
)


@pytest.fixture(scope="module")
def router():
    registry = AgentRegistry()

    agents = load_agent_directory(
        AGENTS_DIR
    )

    registry.register_many(agents)

    backend = MinistralHubBackend(
        MODEL_PATH
    )

    return LLMRouter(
        registry=registry,
        backend=backend,
    )


def test_ministral_routes_account(router):
    routes = router.route(
        "Is jdoe locked?"
    )

    print("ACCOUNT:", routes)

    assert routes == [
        "account-specialist"
    ]


def test_ministral_routes_access(router):
    routes = router.route(
        "Does jdoe have VPN access?"
    )

    print("ACCESS:", routes)

    assert routes == [
        "access-specialist"
    ]


def test_ministral_routes_multiple(router):
    routes = router.route(
        (
            "Check whether jdoe is locked "
            "and whether jdoe has VPN access."
        )
    )

    print("MULTI:", routes)

    assert routes == [
        "account-specialist",
        "access-specialist",
    ]


def test_ministral_no_route(router):
    routes = router.route(
        "Tell me a joke."
    )

    print("NO ROUTE:", routes)

    assert routes == []