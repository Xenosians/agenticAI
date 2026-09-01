from subagents.core.llm_router import LLMRouter
from subagents.core.types import AgentDefinition
from subagents.core.registry import AgentRegistry
from subagents.llm.base import LLMBackend


class FakeHubBackend(LLMBackend):
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.last_messages = None

    def generate(
        self,
        messages,
        max_new_tokens=128,
    ):
        self.last_messages = messages
        return self.response


def build_registry():
    registry = AgentRegistry()

    registry.register(
        AgentDefinition(
            name="account-specialist",
            description="Handles user accounts.",
        )
    )

    registry.register(
        AgentDefinition(
            name="access-specialist",
            description="Handles access and permissions.",
        )
    )

    return registry


def test_llm_router_selects_account_agent():
    backend = FakeHubBackend(
        '{"agents": ["account-specialist"]}'
    )

    router = LLMRouter(
        registry=build_registry(),
        backend=backend,
    )

    routes = router.route(
        "Is jdoe locked?"
    )

    assert routes == [
        "account-specialist"
    ]


def test_llm_router_supports_multiple_agents():
    backend = FakeHubBackend(
        (
            '{"agents": ['
            '"account-specialist", '
            '"access-specialist"'
            ']}'
        )
    )

    router = LLMRouter(
        registry=build_registry(),
        backend=backend,
    )

    routes = router.route(
        "Check jdoe account and VPN access."
    )

    assert routes == [
        "account-specialist",
        "access-specialist",
    ]


def test_llm_router_rejects_unknown_agent():
    backend = FakeHubBackend(
        (
            '{"agents": ['
            '"account-specialist", '
            '"fake-specialist"'
            ']}'
        )
    )

    router = LLMRouter(
        registry=build_registry(),
        backend=backend,
    )

    routes = router.route(
        "Check jdoe."
    )

    assert routes == [
        "account-specialist"
    ]


def test_llm_router_rejects_invalid_json():
    backend = FakeHubBackend(
        "I think account-specialist should do it."
    )

    router = LLMRouter(
        registry=build_registry(),
        backend=backend,
    )

    assert router.route(
        "Check jdoe."
    ) == []


def test_llm_router_handles_no_match():
    backend = FakeHubBackend(
        '{"agents": []}'
    )

    router = LLMRouter(
        registry=build_registry(),
        backend=backend,
    )

    assert router.route(
        "Tell me a joke."
    ) == []