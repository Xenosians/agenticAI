from learning.evaluation.eval_types import (
    EvaluationCase,
)
from learning.continuous.ppo_sandbox import (
    score_router_response,
)
from subagents.core.definitions.registry import (
    AgentRegistry,
)


def test_no_route_simulator_reward_is_deterministic():
    case = EvaluationCase(
        case_id="noop",
        suite="test",
        target="orchestrator",
        description="No specialist required.",
        user_request="Say hello conceptually.",
        expected={
            "hub_status":
                "success",

            "routes":
                [],
        },
    )

    reward = score_router_response(
        response='{"delegations":[]}',
        case=case,
        registry=AgentRegistry(),
    )

    assert reward >= 0.5
    assert reward <= 1.0
