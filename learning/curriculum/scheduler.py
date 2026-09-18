from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel, ConfigDict, Field

from learning.context.assembler import assemble_context_bundle
from learning.context.models import StructuredContextRecord
from learning.curriculum.difficulty import assess_difficulty
from learning.curriculum.domains import classify_trajectory_domains
from learning.curriculum.sampling import CurriculumCandidate, CurriculumSelection, select_curriculum, stable_novelty_key


class CurriculumPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[CurriculumCandidate] = Field(default_factory=list)
    selection: CurriculumSelection
    tier_counts: dict[str, int] = Field(default_factory=dict)


def build_curriculum_plan(
    *,
    trajectories: list,
    signals: list,
    corrections: list,
    context_records: list[StructuredContextRecord],
    max_records: int = 256,
    max_domain_fraction: float = 0.60,
    seed: str = "curriculum-v1",
) -> CurriculumPlan:
    signal_index = {item.trajectory_id: item for item in signals}
    corrections_by_trajectory: dict[str, list] = defaultdict(list)
    for correction in corrections:
        corrections_by_trajectory[correction.trajectory_id].append(correction)

    candidates: list[CurriculumCandidate] = []
    for trajectory in trajectories:
        signal = signal_index.get(trajectory.trajectory_id)
        if signal is None:
            continue

        bundle = assemble_context_bundle(
            context_records,
            trajectory_id=trajectory.trajectory_id,
            max_records=64,
            max_payload_chars=24_000,
        )
        domain = classify_trajectory_domains(trajectory)
        difficulty = assess_difficulty(
            trajectory,
            domain=domain,
            correction_count=len(corrections_by_trajectory.get(trajectory.trajectory_id, [])),
            context_kind_count=len(bundle.kind_counts),
        )
        request = str(getattr(trajectory, "user_request", ""))
        tool_shape = ",".join(
            str(getattr(step, "proposed_tool", "") or "")
            for step in getattr(trajectory, "steps", []) or []
        )
        candidates.append(
            CurriculumCandidate(
                trajectory_id=trajectory.trajectory_id,
                tier=signal.tier,
                primary_domain=domain.primary_domain,
                domains=domain.domains,
                difficulty=difficulty.band,
                difficulty_score=difficulty.score,
                novelty_key=stable_novelty_key(request.casefold().strip(), tool_shape, domain.primary_domain),
                failure_clusters=difficulty.failure_clusters,
            )
        )

    selection = select_curriculum(
        candidates,
        max_records=max_records,
        max_domain_fraction=max_domain_fraction,
        seed=seed,
    )
    return CurriculumPlan(
        candidates=candidates,
        selection=selection,
        tier_counts=dict(Counter(item.tier for item in candidates)),
    )
