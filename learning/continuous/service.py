from __future__ import annotations

import asyncio
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path

from config import Settings

from learning.continual.checkpoints import (
    AdapterCheckpointStore,
)
from learning.continual.controller import (
    ContinualLearningController,
)
from learning.continuous.config import (
    ContinuousLearningSettings,
)
from learning.continuous.canary import (
    CanaryState,
    model_attributable_failure,
    observe_canary_event,
)
from learning.continuous.corpus import (
    next_progressive_pages,
    register_sources,
)
from learning.continuous.dataset_overlay import (
    augment_materialization_with_corpus,
    find_latest_ready_materialization,
)
from learning.continuous.evaluation import (
    ensure_baseline_reports,
    evaluate_candidate,
)
from learning.continuous.events import (
    learning_event_from_trajectory,
)
from learning.continuous.ppo_sandbox import (
    SandboxPPOSettings,
    run_sandbox_sequence_ppo,
)
from learning.continuous.store import (
    ContinuousLearningStore,
)
from learning.continuous.trainer import (
    seed_adapter_directory,
    train_continuous_candidate,
    update_recipe,
)
from learning.continuous.types import (
    ContinuousCyclePlan,
    ContinuousCycleResult,
)
from learning.paths import (
    RUNTIME_LEARNING_ROOT,
)
from learning.training.membership import (
    build_training_membership_plan,
)
from learning.training.phase5_materializer import (
    materialize_phase5_training,
)


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


class ContinuousLearningService:
    """
    Long-running Phase-5.6 control plane.

    Each user trajectory is queued immediately.

    Optimization is intentionally micro-batched instead of one optimizer
    run per request. This preserves the user's "always learning" behavior
    while preventing one noisy request from rewriting the model.

    Single-GPU mode:
        inference has priority;
        training starts only after an idle window;
        one short training micro-cycle holds the GPU scheduler.

    Multi-GPU production can move this service into a separate worker while
    keeping the same durable store/artifact contracts.
    """

    TRAINING_PRIORITY = 1000

    def __init__(
        self,
        *,
        runtime,
        settings: ContinuousLearningSettings | None = None,
    ) -> None:
        self.runtime = runtime

        self.settings = (
            settings
            or ContinuousLearningSettings
            .from_env()
        )

        self.store = (
            ContinuousLearningStore(
                self.settings
                .state_db_path
            )
        )

        self.store.initialize()

        self._wakeup = asyncio.Event()
        self._stop = asyncio.Event()
        self._last_activity = (
            time.monotonic()
        )

        register_sources(
            store=self.store,
            settings=self.settings,
        )

    @property
    def enabled(
        self,
    ) -> bool:
        return self.settings.enabled

    def enqueue_trajectory(
        self,
        trajectory: dict,
    ) -> bool:
        if not self.enabled:
            return False

        event = (
            learning_event_from_trajectory(
                trajectory
            )
        )

        inserted = (
            self.store
            .enqueue_event(
                event
            )
        )

        self._last_activity = (
            time.monotonic()
        )

        self._wakeup.set()

        if inserted:
            print(
                "[CONTINUAL] queued "
                f"event={event.event_id} "
                f"trajectory={event.trajectory_id} "
                f"error={event.had_error}"
            )

            raw_canary = (
                self.store
                .get_json(
                    "canary_state"
                )
            )

            if isinstance(
                raw_canary,
                dict,
            ):
                canary = (
                    CanaryState
                    .model_validate(
                        raw_canary
                    )
                )

                canary = (
                    observe_canary_event(
                        state=canary,
                        had_error=(
                            model_attributable_failure(
                                outcome_codes=(
                                    event.outcome_codes
                                ),
                                failure_types=(
                                    event.failure_types
                                ),
                            )
                        ),
                    )
                )

                self.store.set_json(
                    "canary_state",
                    canary.model_dump(
                        mode="json"
                    ),
                )

                if canary.rollback_requested:
                    self.store.set_json(
                        "rollback_requested",
                        {
                            "checkpoint_id":
                                canary.checkpoint_id,
                        },
                    )

                    self._wakeup.set()

        return inserted

    def request_stop(
        self,
    ) -> None:
        self._stop.set()
        self._wakeup.set()

    def status(
        self,
    ) -> dict:
        counts = (
            self.store
            .pending_event_counts()
        )

        latest = (
            self.store
            .latest_cycle()
        )

        recipe = (
            self.store
            .load_recipe(
                self.settings
                .recipe
            )
        )

        active = (
            AdapterCheckpointStore()
            .active()
        )

        return {
            "enabled":
                self.enabled,

            "pending_events":
                counts,

            "latest_cycle":
                latest,

            "active_checkpoint_id":
                (
                    active
                    .checkpoint_id
                    if active is not None
                    else None
                ),

            "recipe":
                recipe.model_dump(
                    mode="json"
                ),

            "gpu_scheduler_active":
                self.runtime
                .gpu_scheduler
                .active,

            "gpu_queue_depth":
                self.runtime
                .gpu_scheduler
                .queue_depth,
        }

    def _trigger_ready(
        self,
    ) -> bool:
        counts = (
            self.store
            .pending_event_counts()
        )

        if (
            counts[
                "total"
            ]
            >= self.settings
            .min_events_per_cycle
        ):
            return True

        if (
            counts[
                "failures"
            ]
            >= self.settings
            .failure_events_trigger
        ):
            return True

        age = (
            self.store
            .oldest_pending_event_age_seconds()
        )

        return (
            age is not None
            and age
            >= self.settings
            .max_cycle_age_seconds
        )

    def _idle_ready(
        self,
    ) -> bool:
        if not (
            self.settings
            .single_gpu_idle_only
        ):
            return True

        if self.runtime.active_jobs:
            return False

        if (
            self.runtime
            .gpu_scheduler
            .active
        ):
            return False

        if (
            self.runtime
            .gpu_scheduler
            .queue_depth
            > 0
        ):
            return False

        idle = (
            time.monotonic()
            - self._last_activity
        )

        return (
            idle
            >= self.settings
            .idle_seconds_before_training
        )

    def _new_cycle_id(
        self,
    ) -> str:
        return (
            "continuous-cycle-"
            + uuid.uuid4().hex
        )

    def _select_pages(
        self,
        *,
        recipe,
    ):
        # Refresh source discovery each cycle so a newly indexed immutable
        # code-corpus snapshot becomes learnable without restarting the API.
        register_sources(
            store=self.store,
            settings=self.settings,
        )

        # Retry previously prepared-but-not-accepted pages first.
        pages = (
            self.store
            .available_pages(
                limit=(
                    recipe
                    .corpus_pages_per_cycle
                )
            )
        )

        missing = (
            recipe
            .corpus_pages_per_cycle
            - len(
                pages
            )
        )

        if missing > 0:
            pages.extend(
                next_progressive_pages(
                    store=self.store,
                    limit=missing,
                )
            )

        replay = (
            self.store
            .replay_pages(
                limit=(
                    recipe
                    .replay_pages_per_cycle
                )
            )
        )

        seen = {
            page.page_id
            for page in pages
        }

        pages.extend(
            page
            for page in replay
            if page.page_id
            not in seen
        )

        return pages

    def _behavior_materialization(
        self,
    ) -> Path:
        """
        Rebuild governed membership/materialization from current durable
        corrections/reviews/curriculum. If current state is not newly ready,
        fall back to the newest previously-ready materialization so corpus
        study can still continue.
        """
        try:
            # Refresh evidence classification/replay artifacts.
            ContinualLearningController().run()

        except Exception as exc:
            print(
                "[CONTINUAL] evidence cycle warning "
                f"error={exc!r}"
            )

        app_settings = Settings()

        profile = (
            app_settings
            .require_model_profile(
                app_settings
                .hub_model_key
            )
        )

        if profile.model_path is None:
            raise RuntimeError(
                "Hub model has no local model_path."
            )

        try:
            behavior_only_code_root = (
                RUNTIME_LEARNING_ROOT
                / "continuous"
                / "behavior-membership-code-disabled"
            )

            behavior_only_code_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            membership = (
                build_training_membership_plan(
                    curriculum_name="hub",
                    code_corpus_root=(
                        behavior_only_code_root
                    ),
                )
            )

            if (
                membership
                .manifest
                .readiness_ready
            ):
                materialized = (
                    materialize_phase5_training(
                        plan_directory=Path(
                            membership
                            .output_directory
                        ),
                        target_model_key=(
                            app_settings
                            .hub_model_key
                        ),
                        base_model_path=(
                            profile
                            .model_path
                        ),
                    )
                )

                return Path(
                    materialized
                    .output_directory
                )

        except Exception as exc:
            print(
                "[CONTINUAL] current membership not materializable "
                f"error={exc!r}"
            )

        return (
            find_latest_ready_materialization()
        )

    async def _train_under_gpu_lease(
        self,
        *,
        materialization_directory: Path,
        recipe,
        seed_adapter: Path | None,
    ):
        def work(
        ):
            # Release inference residency before the training model is loaded.
            self.runtime.model_manager.unload_all()

            return (
                train_continuous_candidate(
                    materialization_directory=(
                        materialization_directory
                    ),
                    recipe=recipe,
                    seed_adapter=(
                        seed_adapter
                    ),
                )
            )

        return (
            await
            self.runtime
            .gpu_scheduler
            .run(
                priority=(
                    self
                    .TRAINING_PRIORITY
                ),
                work=work,
            )
        )

    async def _ppo_under_gpu_lease(
        self,
        *,
        checkpoint_id: str,
        cycle_id: str,
        recipe,
    ):
        ppo_settings = (
            SandboxPPOSettings(
                ppo_epochs=max(
                    1,
                    recipe.ppo_updates,
                ),
                max_episodes=4,
            )
        )

        def work(
        ):
            self.runtime.model_manager.unload_all()

            return (
                run_sandbox_sequence_ppo(
                    source_checkpoint_id=(
                        checkpoint_id
                    ),
                    cycle_id=(
                        cycle_id
                    ),
                    suite=(
                        self.settings
                        .ppo_suite
                    ),
                    settings=(
                        ppo_settings
                    ),
                )
            )

        return (
            await
            self.runtime
            .gpu_scheduler
            .run(
                priority=(
                    self
                    .TRAINING_PRIORITY
                ),
                work=work,
            )
        )

    async def _apply_requested_rollback(
        self,
    ) -> bool:
        request = (
            self.store
            .get_json(
                "rollback_requested"
            )
        )

        if not isinstance(
            request,
            dict,
        ):
            return False

        try:
            pointer = (
                AdapterCheckpointStore()
                .rollback()
            )

            self.runtime.model_manager.unload_all()

            await self.runtime.inference.warm(
                self.runtime
                .settings
                .hub_model_key
            )

            self.store.set_json(
                "rollback_requested",
                None,
            )

            print(
                "[CONTINUAL] canary rollback applied "
                f"checkpoint={pointer.checkpoint_id}"
            )

            return True

        except Exception as exc:
            print(
                "[CONTINUAL] canary rollback failed "
                f"error={exc!r}"
            )

            return False

    async def run_cycle(
        self,
    ) -> ContinuousCycleResult | None:
        if not self.enabled:
            return None

        if not self._trigger_ready():
            return None

        if not self._idle_ready():
            return None

        cycle_id = (
            self._new_cycle_id()
        )

        recipe = (
            self.store
            .load_recipe(
                self.settings
                .recipe
            )
        )

        events = (
            self.store
            .claim_events(
                cycle_id=cycle_id,
                limit=(
                    self.settings
                    .max_events_per_cycle
                ),
            )
        )

        if not events:
            return None

        pages = (
            self._select_pages(
                recipe=recipe
            )
        )

        self.store.claim_pages(
            cycle_id=cycle_id,
            page_ids=[
                page.page_id
                for page in pages
                if not page.replay
            ],
        )

        seed_checkpoint_id, seed_adapter = (
            seed_adapter_directory()
        )

        plan = ContinuousCyclePlan(
            cycle_id=cycle_id,
            created_at=_utc_now(),
            event_ids=[
                event.event_id
                for event in events
            ],
            corpus_page_ids=[
                page.page_id
                for page in pages
            ],
            seed_checkpoint_id=(
                seed_checkpoint_id
            ),
            stages=[
                "evidence",
                "progressive_corpus",
                "sft",
                "dpo",
                *(
                    [
                        "ppo_simulator",
                    ]
                    if self.settings
                    .ppo_enabled
                    else []
                ),
                "heldout_eval",
                "safety_eval",
                "promotion_gate",
            ],
            recipe=recipe,
        )

        self.store.save_cycle_plan(
            plan
        )

        baseline_intelligence = None
        baseline_safety = None

        try:
            if (
                self.settings
                .auto_evaluate
            ):
                (
                    baseline_intelligence,
                    baseline_safety,
                ) = (
                    await
                    ensure_baseline_reports(
                        runtime=(
                            self.runtime
                        ),
                        suite=(
                            self.settings
                            .eval_suite
                        ),
                        state_store=(
                            self.store
                        ),
                    )
                )

            behavior_materialization = (
                self._behavior_materialization()
            )

            effective_materialization = (
                augment_materialization_with_corpus(
                    base_directory=(
                        behavior_materialization
                    ),
                    cycle_id=(
                        cycle_id
                    ),
                    pages=(
                        pages
                    ),
                )
                if pages
                else behavior_materialization
            )

            if not self.settings.auto_train:
                result = (
                    ContinuousCycleResult(
                        cycle_id=cycle_id,
                        completed_at=_utc_now(),
                        state="rejected",
                        guard_signal="training_disabled",
                        notes=[
                            "CONTINUAL_AUTO_TRAIN is disabled."
                        ],
                    )
                )

                self.store.save_cycle_result(
                    result
                )

                self.store.finalize_events(
                    cycle_id=cycle_id,
                    consumed=True,
                )

                self.store.finalize_pages(
                    cycle_id=cycle_id,
                    consumed=False,
                )

                return result

            training_manifest = (
                await
                self._train_under_gpu_lease(
                    materialization_directory=(
                        effective_materialization
                    ),
                    recipe=recipe,
                    seed_adapter=(
                        seed_adapter
                    ),
                )
            )

            candidate_checkpoint_id = (
                training_manifest
                .registered_checkpoint_id
            )

            (
                next_recipe,
                guard_signal,
            ) = update_recipe(
                manifest=(
                    training_manifest
                ),
                recipe=recipe,
            )

            self.store.save_recipe(
                next_recipe
            )

            cycle_number = (
                self.store
                .cycle_count()
            )

            if (
                self.settings
                .ppo_enabled
                and recipe.ppo_updates
                > 0
                and cycle_number
                % self.settings
                .ppo_every_n_cycles
                == 0
            ):
                ppo_result = (
                    await
                    self._ppo_under_gpu_lease(
                        checkpoint_id=(
                            candidate_checkpoint_id
                        ),
                        cycle_id=(
                            cycle_id
                        ),
                        recipe=recipe,
                    )
                )

                candidate_checkpoint_id = (
                    ppo_result
                    .checkpoint_id
                )

            promotion_eligible = False
            improvement_observed = False
            promotion_decision_id = None
            activated = False

            candidate_intelligence_id = None
            candidate_safety_id = None

            if (
                self.settings
                .auto_evaluate
                and baseline_intelligence
                is not None
                and baseline_safety
                is not None
            ):
                bundle = (
                    await
                    evaluate_candidate(
                        runtime=(
                            self.runtime
                        ),
                        suite=(
                            self.settings
                            .eval_suite
                        ),
                        checkpoint_id=(
                            candidate_checkpoint_id
                        ),
                        baseline_intelligence=(
                            baseline_intelligence
                        ),
                        baseline_safety=(
                            baseline_safety
                        ),
                        label=(
                            "continuous-"
                            + cycle_id
                        ),
                    )
                )

                decision = (
                    bundle
                    .decision
                )

                candidate_intelligence_id = (
                    bundle
                    .candidate_intelligence
                    .report_id
                )

                candidate_safety_id = (
                    bundle
                    .candidate_safety
                    .report_id
                )

                promotion_eligible = bool(
                    decision
                    .promotion_eligible
                )

                improvement_observed = bool(
                    decision
                    .improvement_observed
                )

                promotion_decision_id = (
                    decision
                    .decision_id
                )

                auto_eligible = (
                    promotion_eligible
                    and (
                        improvement_observed
                        or not self.settings
                        .require_improvement_for_auto_promote
                    )
                )

                if (
                    self.settings
                    .auto_promote
                    and auto_eligible
                ):
                    pointer = (
                        AdapterCheckpointStore()
                        .promote(
                            checkpoint_id=(
                                candidate_checkpoint_id
                            ),
                            suite=(
                                self.settings
                                .eval_suite
                            ),
                            decision_id=(
                                promotion_decision_id
                            ),
                        )
                    )

                    self.store.set_json(
                        "canary_state",
                        CanaryState(
                            checkpoint_id=(
                                pointer
                                .checkpoint_id
                            ),
                            previous_checkpoint_id=(
                                pointer
                                .previous_checkpoint_id
                            ),
                        )
                        .model_dump(
                            mode="json"
                        ),
                    )

                    activated = True

            accepted = (
                promotion_eligible
                if self.settings
                .auto_evaluate
                else True
            )

            result = (
                ContinuousCycleResult(
                    cycle_id=cycle_id,
                    completed_at=_utc_now(),
                    state=(
                        "accepted"
                        if accepted
                        else "rejected"
                    ),
                    candidate_checkpoint_id=(
                        candidate_checkpoint_id
                    ),
                    baseline_intelligence_report_id=(
                        baseline_intelligence
                        .report_id
                        if baseline_intelligence
                        is not None
                        else None
                    ),
                    candidate_intelligence_report_id=(
                        candidate_intelligence_id
                    ),
                    baseline_safety_report_id=(
                        baseline_safety
                        .report_id
                        if baseline_safety
                        is not None
                        else None
                    ),
                    candidate_safety_report_id=(
                        candidate_safety_id
                    ),
                    promotion_decision_id=(
                        promotion_decision_id
                    ),
                    promotion_eligible=(
                        promotion_eligible
                    ),
                    improvement_observed=(
                        improvement_observed
                    ),
                    activated=(
                        activated
                    ),
                    guard_signal=(
                        guard_signal
                    ),
                    notes=[
                        (
                            "Candidate was automatically activated."
                            if activated
                            else (
                                "Candidate retained for review/next cycle; "
                                "production pointer was not changed."
                            )
                        )
                    ],
                )
            )

            self.store.save_cycle_result(
                result
            )

            self.store.finalize_events(
                cycle_id=cycle_id,
                consumed=True,
            )

            self.store.finalize_pages(
                cycle_id=cycle_id,
                consumed=(
                    promotion_eligible
                ),
            )

            return result

        except Exception as exc:
            result = (
                ContinuousCycleResult(
                    cycle_id=cycle_id,
                    completed_at=_utc_now(),
                    state="failed",
                    guard_signal="exception",
                    notes=[
                        repr(
                            exc
                        )
                    ],
                )
            )

            self.store.save_cycle_result(
                result
            )

            # Runtime trajectories remain durable in their original ledgers.
            # Requeue the controller events/pages for a future bounded retry.
            self.store.finalize_events(
                cycle_id=cycle_id,
                consumed=False,
            )

            self.store.finalize_pages(
                cycle_id=cycle_id,
                consumed=False,
            )

            print(
                "[CONTINUAL] cycle failed "
                f"cycle={cycle_id} "
                f"error={exc!r}"
            )

            return result

        finally:
            # Restore the production serving model. If the candidate was
            # promoted, ModelManager will pick up the new active pointer on
            # this fresh load.
            try:
                await self.runtime.inference.warm(
                    self.runtime
                    .settings
                    .hub_model_key
                )
            except Exception as exc:
                print(
                    "[CONTINUAL] production model rewarm failed "
                    f"error={exc!r}"
                )

    async def run_forever(
        self,
    ) -> None:
        if not self.enabled:
            print(
                "[CONTINUAL] service disabled"
            )
            return

        print(
            "[CONTINUAL] service started "
            f"min_events={self.settings.min_events_per_cycle} "
            "corpus_pages="
            f"{self.settings.recipe.corpus_pages_per_cycle} "
            f"ppo={self.settings.ppo_enabled} "
            f"auto_promote={self.settings.auto_promote}"
        )

        try:
            while not (
                self._stop
                .is_set()
            ):
                try:
                    rolled_back = (
                        await
                        self._apply_requested_rollback()
                    )

                    if not rolled_back:
                        await self.run_cycle()

                except asyncio.CancelledError:
                    raise

                except Exception as exc:
                    print(
                        "[CONTINUAL] controller tick failed "
                        f"error={exc!r}"
                    )

                try:
                    await asyncio.wait_for(
                        self._wakeup.wait(),
                        timeout=(
                            self.settings
                            .poll_seconds
                        ),
                    )

                except asyncio.TimeoutError:
                    pass

                finally:
                    self._wakeup.clear()

        except asyncio.CancelledError:
            print(
                "[CONTINUAL] service stopped"
            )
            raise
