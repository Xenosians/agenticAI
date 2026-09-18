from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from learning.curriculum.domains import DomainAssessment


class DifficultyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int
    band: str
    reasons: list[str] = Field(default_factory=list)
    failure_clusters: list[str] = Field(default_factory=list)


def assess_difficulty(
    trajectory,
    *,
    domain: DomainAssessment,
    correction_count: int = 0,
    context_kind_count: int = 0,
) -> DifficultyAssessment:
    score = 0
    reasons: list[str] = []
    clusters: list[str] = []

    steps = list(getattr(trajectory, "steps", []) or [])
    signals = getattr(trajectory, "signals", None)
    quality = getattr(trajectory, "quality", None)

    if domain.cross_domain:
        score += 2
        reasons.append("cross-domain")
        clusters.append("cross_domain")

    if len(steps) > 1 or int(getattr(signals, "route_count", 0) or 0) > 1:
        score += 1
        reasons.append("multi-specialist")

    if correction_count > 0:
        score += 1
        reasons.append("human-corrected")

    if int(getattr(signals, "approval_required_count", 0) or 0) > 0:
        score += 1
        reasons.append("approval-boundary")

    if int(getattr(signals, "specialist_error_count", 0) or 0) > 0:
        score += 1
        reasons.append("specialist-failure")
        clusters.append("specialist_failure")

    if context_kind_count >= 3:
        score += 1
        reasons.append("rich-context")

    for step in steps:
        code = getattr(step, "outcome_code", None)
        if code == "grounding_failed" or code == "semantic_argument_not_allowed":
            score += 2
            reasons.append("grounding-drift")
            clusters.append("grounding")
        elif code in {"unknown_tool", "agent_tool_not_allowed"}:
            score += 2
            reasons.append("tool-identity-failure")
            clusters.append("tool_identity")
        elif code == "invalid_tool_call_count":
            score += 2
            reasons.append("tool-protocol-failure")
            clusters.append("tool_protocol")
        elif code == "conditional_source_failed":
            score += 2
            reasons.append("conditional-workflow-failure")
            clusters.append("conditional_workflow")

    if quality is not None:
        if getattr(quality, "grounding_valid", None) is False:
            clusters.append("grounding")
        for failure in getattr(quality, "failure_types", []) or []:
            if failure:
                clusters.append(str(failure))

    unique_reasons = list(dict.fromkeys(reasons))
    unique_clusters = list(dict.fromkeys(clusters))

    if score <= 1:
        band = "easy"
    elif score <= 3:
        band = "medium"
    else:
        band = "hard"

    return DifficultyAssessment(
        score=score,
        band=band,
        reasons=unique_reasons,
        failure_clusters=unique_clusters,
    )
