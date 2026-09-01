from subagents.core.loader import load_agent_directory
from subagents.core.registry import AgentRegistry
from subagents.core.router import Router


def build_router() -> Router:
    registry = AgentRegistry()

    agents = load_agent_directory(
        "subagents/agents"
    )

    registry.register_many(agents)

    return Router(registry)


def test_routes_account_request():
    router = build_router()

    routes = router.route(
        "Is jdoe locked?"
    )

    assert routes == [
        "account-specialist"
    ]


def test_routes_access_request():
    router = build_router()

    routes = router.route(
        "Does jdoe have VPN access?"
    )

    assert routes == [
        "access-specialist"
    ]


def test_routes_to_multiple_specialists():
    router = build_router()

    routes = router.route(
        "Check whether jdoe is locked and whether he has VPN access."
    )

    assert routes == [
        "account-specialist",
        "access-specialist",
    ]


def test_unknown_request_returns_no_route():
    router = build_router()

    routes = router.route(
        "Hello, how are you?"
    )

    assert routes == []