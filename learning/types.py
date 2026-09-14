from __future__ import annotations

from typing import (
    Any,
)

from pydantic import (
    BaseModel,
    Field,
)


class TrajectoryStep(
    BaseModel
):
    task_id: str

    agent: str
    status: str

    proposed_tool: (
        str | None
    ) = None

    proposed_arguments: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    tool_result: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    approval_id: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


class TrajectorySignals(
    BaseModel
):
    delegated: bool

    route_count: int = 0
    specialist_count: int = 0

    specialist_success_count: int = 0
    specialist_error_count: int = 0
    approval_required_count: int = 0

    tool_proposed_count: int = 0
    tool_success_count: int = 0

    overall_success: bool = False
    had_error: bool = False
    waiting_approval: bool = False


class ExecutionReward(
    BaseModel
):
    """
    Deterministic runtime reward.

    IMPORTANT:

    This is NOT yet a semantic-quality reward.

    A tool may execute successfully while still being the wrong
    tool for the user's intent.

    Therefore raw execution rewards must not automatically become
    reinforcement-learning labels.
    """

    schema: str = (
        "execution-reward.v1"
    )

    components: dict[
        str,
        float,
    ] = Field(
        default_factory=dict
    )

    total: float = 0.0

    quality_eligible: bool = False


class LearningTrajectory(
    BaseModel
):
    schema: str = (
        "trajectory.v1"
    )

    trajectory_id: str

    observed_at: str

    job_id: str
    attempt: int

    user_request: str

    hub_model: str

    hub_status: str

    routes: list[
        str
    ] = Field(
        default_factory=list
    )

    steps: list[
        TrajectoryStep
    ] = Field(
        default_factory=list
    )

    final_answer: (
        str | None
    ) = None

    signals: TrajectorySignals

    execution_reward: (
        ExecutionReward
    )

    # Raw trajectories are evidence, not automatically approved
    # training examples.
    dataset_eligible: bool = False