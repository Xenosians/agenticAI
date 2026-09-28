from __future__ import annotations

from dataclasses import dataclass

from learning.continual.checkpoints import (
    AdapterCheckpointStore,
)
from learning.continual.storage import (
    fingerprint_directory,
)
from learning.curation.promotion_gate import (
    PromotionGateStore,
    build_model_promotion_decision,
)
from learning.evaluation.eval_reports import (
    EvaluationReportArtifact,
    EvaluationReportStore,
)
from learning.evaluation.eval_suite import (
    load_evaluation_cases,
)
from learning.evaluation.gateway_evaluation import (
    GatewayEvaluationRunner,
)
from learning.evaluation.live_evaluation import (
    LiveOrchestratorEvaluationRunner,
)
from learning.paths import (
    EVALUATIONS_ROOT,
    PROMOTIONS_ROOT,
    REPOSITORY_ROOT,
)
from subagents.hub import (
    build_hub,
)
from subagents.llm.runtime.inference import (
    InferenceCoordinator,
)
from subagents.llm.runtime.model_manager import (
    ModelManager,
)


@dataclass(
    frozen=True
)
class EvaluationBundle:
    baseline_intelligence: (
        EvaluationReportArtifact
    )
    candidate_intelligence: (
        EvaluationReportArtifact
    )
    baseline_safety: (
        EvaluationReportArtifact
    )
    candidate_safety: (
        EvaluationReportArtifact
    )
    decision: object


def _cases(
    suite: str,
):
    path = (
        REPOSITORY_ROOT
        / "learning"
        / "evaluation"
        / "evals"
        / f"{suite}.jsonl"
    )

    values = (
        load_evaluation_cases(
            path
        )
    )

    return (
        [
            case
            for case in values
            if case.target
            == "orchestrator"
        ],
        [
            case
            for case in values
            if case.target
            == "tool_gateway"
        ],
    )


async def ensure_baseline_reports(
    *,
    runtime,
    suite: str,
    state_store,
) -> tuple[
    EvaluationReportArtifact,
    EvaluationReportArtifact,
]:
    """
    Baseline is keyed to the currently active production checkpoint.
    It is reused until production activation changes.
    """
    checkpoint_store = (
        AdapterCheckpointStore()
    )

    active = (
        checkpoint_store.active()
    )

    active_id = (
        active.checkpoint_id
        if active is not None
        else "base"
    )

    profile = (
        runtime.settings
        .require_model_profile(
            runtime.settings
            .hub_model_key
        )
    )

    base_fingerprint = (
        fingerprint_directory(
            profile.model_path
        )
        if profile.model_path
        is not None
        else "no-local-model"
    )

    cache_key = (
        "baseline:"
        + suite
        + ":"
        + active_id
        + ":"
        + base_fingerprint[
            :16
        ]
    )

    cached = (
        state_store
        .get_json(
            cache_key
        )
    )

    report_store = (
        EvaluationReportStore(
            root=(
                EVALUATIONS_ROOT
            )
        )
    )

    if isinstance(
        cached,
        dict,
    ):
        try:
            return (
                report_store.load(
                    suite=suite,
                    target="orchestrator",
                    report_id=(
                        cached[
                            "intelligence"
                        ]
                    ),
                ),
                report_store.load(
                    suite=suite,
                    target="tool_gateway",
                    report_id=(
                        cached[
                            "safety"
                        ]
                    ),
                ),
            )
        except Exception:
            pass

    (
        orchestrator_cases,
        gateway_cases,
    ) = _cases(
        suite
    )

    intelligence_report = (
        await
        LiveOrchestratorEvaluationRunner(
            hub=runtime.hub,
            hub_model=(
                runtime
                .settings
                .hub_model_key
            ),
        )
        .run(
            orchestrator_cases
        )
    )

    intelligence_artifact = (
        report_store
        .save(
            report=(
                intelligence_report
            ),
            label=(
                "continuous-baseline-"
                + active_id
            ),
        )
    )

    safety_report = (
        await
        GatewayEvaluationRunner()
        .run(
            gateway_cases
        )
    )

    safety_artifact = (
        report_store
        .save(
            report=(
                safety_report
            ),
            label=(
                "continuous-baseline-safety-"
                + active_id
            ),
        )
    )

    state_store.set_json(
        cache_key,
        {
            "intelligence":
                intelligence_artifact
                .report_id,

            "safety":
                safety_artifact
                .report_id,
        },
    )

    return (
        intelligence_artifact,
        safety_artifact,
    )


async def evaluate_candidate(
    *,
    runtime,
    suite: str,
    checkpoint_id: str,
    baseline_intelligence: EvaluationReportArtifact,
    baseline_safety: EvaluationReportArtifact,
    label: str,
) -> EvaluationBundle:
    (
        orchestrator_cases,
        gateway_cases,
    ) = _cases(
        suite
    )

    candidate_manager = (
        ModelManager(
            settings=(
                runtime.settings
            ),
            checkpoint_id_override=(
                checkpoint_id
            ),
            allow_unpromoted_checkpoint_override=True,
        )
    )

    candidate_inference = (
        InferenceCoordinator(
            model_manager=(
                candidate_manager
            ),
            scheduler=(
                runtime
                .gpu_scheduler
            ),
        )
    )

    candidate_hub = (
        build_hub(
            settings=(
                runtime.settings
            ),
            model_manager=(
                candidate_manager
            ),
            inference=(
                candidate_inference
            ),
            tool_gateway=(
                runtime
                .tool_gateway
            ),
        )
    )

    report_store = (
        EvaluationReportStore(
            root=(
                EVALUATIONS_ROOT
            )
        )
    )

    try:
        await candidate_inference.warm(
            runtime
            .settings
            .hub_model_key
        )

        candidate_report = (
            await
            LiveOrchestratorEvaluationRunner(
                hub=(
                    candidate_hub
                ),
                hub_model=(
                    runtime
                    .settings
                    .hub_model_key
                ),
            )
            .run(
                orchestrator_cases
            )
        )

        candidate_intelligence = (
            report_store
            .save(
                report=(
                    candidate_report
                ),
                label=(
                    label
                ),
            )
        )

    finally:
        candidate_manager.unload_all()

    candidate_safety_report = (
        await
        GatewayEvaluationRunner()
        .run(
            gateway_cases
        )
    )

    candidate_safety = (
        report_store
        .save(
            report=(
                candidate_safety_report
            ),
            label=(
                label
                + "-safety"
            ),
        )
    )

    decision = (
        build_model_promotion_decision(
            baseline_intelligence=(
                baseline_intelligence
            ),
            candidate_intelligence=(
                candidate_intelligence
            ),
            baseline_safety=(
                baseline_safety
            ),
            candidate_safety=(
                candidate_safety
            ),
            label=(
                label
            ),
        )
    )

    PromotionGateStore(
        root=(
            PROMOTIONS_ROOT
        )
    ).save(
        decision
    )

    return (
        EvaluationBundle(
            baseline_intelligence=(
                baseline_intelligence
            ),
            candidate_intelligence=(
                candidate_intelligence
            ),
            baseline_safety=(
                baseline_safety
            ),
            candidate_safety=(
                candidate_safety
            ),
            decision=(
                decision
            ),
        )
    )
