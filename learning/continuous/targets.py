from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from subagents.core.definitions.loader import (
    load_agent_directory,
)


TrainingPath = Literal[
    "phase5_hub",
    "shared_hub_adapter_corpus",
    "shared_adapter_unwired",
    "separate_model_unwired",
]


class ContinuousTargetCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_component: str
    model_key: str
    training_path: TrainingPath

    candidate_training_supported: bool = False
    behavior_training_supported: bool = False
    corpus_adaptation_supported: bool = False
    auto_promotion_supported: bool = False

    required_evaluation: list[str] = Field(
        default_factory=list
    )
    blocked_reasons: list[str] = Field(
        default_factory=list
    )


def build_continuous_target_coverage(
    *,
    hub_model_key: str,
    agent_directory: Path,
) -> list[ContinuousTargetCoverage]:
    """
    Describe what the current continuous trainer can actually train.

    This is descriptive only. It does not widen model, tool, approval,
    grounding, or execution authority.
    """
    result = [
        ContinuousTargetCoverage(
            target_component="hub",
            model_key=hub_model_key,
            training_path="phase5_hub",
            candidate_training_supported=True,
            behavior_training_supported=True,
            corpus_adaptation_supported=True,
            auto_promotion_supported=True,
            required_evaluation=[
                "hub-intelligence",
                "gateway-safety",
            ],
        )
    ]

    agents = load_agent_directory(agent_directory)

    for agent in agents:
        if (
            agent.name == "developer-specialist"
            and agent.model == hub_model_key
        ):
            result.append(
                ContinuousTargetCoverage(
                    target_component="developer-specialist",
                    model_key=agent.model,
                    training_path="shared_hub_adapter_corpus",
                    candidate_training_supported=True,
                    behavior_training_supported=False,
                    corpus_adaptation_supported=True,
                    auto_promotion_supported=False,
                    required_evaluation=[
                        "developer-heldout",
                        "gateway-safety",
                    ],
                    blocked_reasons=[
                        (
                            "Developer code corpus can adapt the shared "
                            "hub-main adapter, but developer tool-behavior "
                            "materialization is not wired into Phase-5."
                        ),
                        (
                            "Role-specific developer held-out evaluation "
                            "is not wired into the automatic promotion gate."
                        ),
                    ],
                )
            )
            continue

        if agent.model == hub_model_key:
            result.append(
                ContinuousTargetCoverage(
                    target_component=agent.name,
                    model_key=agent.model,
                    training_path="shared_adapter_unwired",
                    candidate_training_supported=False,
                    behavior_training_supported=False,
                    corpus_adaptation_supported=False,
                    auto_promotion_supported=False,
                    required_evaluation=[
                        agent.name + "-heldout",
                        "gateway-safety",
                    ],
                    blocked_reasons=[
                        (
                            "This role shares the Hub model, so a Hub "
                            "adapter candidate can change its behavior, "
                            "but target-specific continuous training is "
                            "not wired."
                        ),
                        (
                            "Role-specific held-out evaluation is not "
                            "wired into the automatic promotion gate."
                        ),
                    ],
                )
            )
            continue

        result.append(
            ContinuousTargetCoverage(
                target_component=agent.name,
                model_key=agent.model,
                training_path="separate_model_unwired",
                candidate_training_supported=False,
                behavior_training_supported=False,
                corpus_adaptation_supported=False,
                auto_promotion_supported=False,
                required_evaluation=[
                    agent.name + "-heldout",
                    "gateway-safety",
                ],
                blocked_reasons=[
                    (
                        "The role uses a separate logical model and has "
                        "no continuous target-specific trainer wired into "
                        "this service."
                    )
                ],
            )
        )

    return result
