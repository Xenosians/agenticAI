"""Stage-aware inference dispatch with a single observation boundary."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Protocol, TypeVar

from subagents.llm.runtime.model_manager import ModelManager
from subagents.llm.runtime.observability import (
    InferenceMetric,
    MemorySampler,
    MetricSink,
    capture_cuda_memory,
    emit_inference_metric,
)
from subagents.llm.runtime.scheduler import GpuScheduler, InferencePriority
from subagents.llm.runtime.stages import InferenceStage, StagePolicy, resolve_stage_policy


_LOGGER = logging.getLogger(__name__)
_Observation = TypeVar("_Observation")


def _observe(label: str, callback: Callable[[], _Observation]) -> _Observation | None:
    """Telemetry must not change an inference result, error, or cancellation."""
    try:
        return callback()
    except (Exception, asyncio.CancelledError) as exc:
        # Callbacks are synchronous, so CancelledError here belongs to the
        # observer, not to an await in the actual inference operation.
        _LOGGER.warning(
            "Inference observation failed: observation=%s error_type=%s",
            label, type(exc).__name__,
        )
        return None


class InferenceEngine(Protocol):
    async def generate(
        self,
        *,
        model_key: str,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        priority: int,
        stage: InferenceStage | str,
        repair_of: InferenceStage | str | None = None,
    ) -> str:
        ...


def _stage_request(profile: dict, messages: list[dict], max_new_tokens: int, policy: StagePolicy):
    """Use the existing budget resolver, then only lower its effective limits."""
    from services.extensions.budget import (
        ExternalInferenceBudgetError,
        default_external_request_budget,
        prepare_external_request,
    )

    default = default_external_request_budget(profile, backend=profile["backend"])
    # Resolve profile/environment precedence and validate configuration using
    # the shared resolver. One output token avoids reserving unused capacity.
    configured = prepare_external_request(profile, [], 1, default_request_budget=default)
    bounded_profile = dict(profile)
    bounded_profile["request_token_budget"] = min(
        default,
        configured.request_token_budget,
        policy.request_token_ceiling or configured.request_token_budget,
    )
    try:
        requested = int(max_new_tokens)
    except (TypeError, ValueError):
        raise ExternalInferenceBudgetError("Invalid output token request") from None
    bounded = prepare_external_request(
        bounded_profile,
        messages,
        min(requested, policy.output_token_ceiling or requested),
        default_request_budget=default,
    )
    return bounded_profile, bounded


class InferenceCoordinator:
    """Apply trusted stage policy before adapter dispatch or GPU admission.

    Every attempt emits content-free dispatch accounting. Local work retains
    actual scheduler, generation, and CUDA measurements; remote work never
    pretends that those GPU measurements exist.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        scheduler: GpuScheduler,
        memory_sampler: MemorySampler | None = None,
        metric_sink: MetricSink | None = None,
    ) -> None:
        self.model_manager = model_manager
        self.scheduler = scheduler
        self.memory_sampler = memory_sampler if memory_sampler is not None else capture_cuda_memory
        self.metric_sink = metric_sink if metric_sink is not None else emit_inference_metric

    async def generate(
        self,
        *,
        model_key: str,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        priority: int,
        stage: InferenceStage | str,
        repair_of: InferenceStage | str | None = None,
    ) -> str:
        return await self._execute(
            operation="generate", model_key=model_key, messages=messages,
            max_new_tokens=max_new_tokens, priority=priority, stage=stage, repair_of=repair_of,
        )

    async def warm(
        self,
        model_key: str,
        *,
        stage: InferenceStage | str,
        repair_of: InferenceStage | str | None = None,
    ) -> None:
        await self._execute(
            operation="warm", model_key=model_key, messages=[], max_new_tokens=0,
            priority=InferencePriority.STARTUP, stage=stage, repair_of=repair_of,
        )

    async def _execute(
        self, *, operation, model_key, messages, max_new_tokens, priority, stage, repair_of,
    ):
        from services.extensions.budget import ExternalInferenceBudgetError
        from services.extensions.context import selected_profile, with_retrieval

        started = time.perf_counter()
        stage_label, repair_label, adapter, outcome = "INVALID", None, "unresolved", "error"
        loaded_before = loaded_after = None
        memory_before = memory_after = None
        observe_local = False
        scheduled = generation = None
        error_type = None
        request_budget = output_cap = estimated_input = None
        input_message_count = len(messages)
        dropped_messages = 0
        try:
            policy = resolve_stage_policy(stage, repair_of)
            stage_label = InferenceStage(stage).value
            repair_label = InferenceStage(repair_of).value if repair_of is not None else None
            adapter = "local"
            profile = selected_profile(stage=stage, repair_of=repair_of)
            if operation == "generate":
                messages = with_retrieval(messages, stage=stage, repair_of=repair_of)
                input_message_count = len(messages)
            if profile:
                backend = profile["backend"]
                if backend not in {"http", "opencode", "pi", "local", "uploaded"}:
                    raise ValueError("Unsupported selected model backend")
                adapter = backend
                if backend in {"http", "opencode", "pi"}:
                    if operation == "warm":
                        outcome = "skipped"  # No local weights for remote inference.
                        return None
                    profile, budget = _stage_request(profile, messages, max_new_tokens, policy)
                    request_budget = budget.request_token_budget
                    output_cap = budget.max_output_tokens
                    estimated_input = budget.estimated_input_tokens
                    dropped_messages = budget.dropped_messages
                    if backend == "http":
                        from services.extensions.transport import remote_generate
                        from services.extensions.response_format import http_response_format_for_stage
                        response_format = http_response_format_for_stage(
                            profile, stage=stage, repair_of=repair_of,
                        )
                        format_options = {"response_format": response_format} if response_format is not None else {}
                        result = await remote_generate(
                            profile, budget.messages, budget.max_output_tokens, **format_options,
                        )
                    else:
                        from services.extensions.harness import harness_generate
                        result = await harness_generate(profile, budget.messages, budget.max_output_tokens)
                    outcome = "success"
                    return result
                if backend == "local":
                    model_key = profile["model"]
                else:
                    model_key = self.model_manager.register_user_model(profile)

            if not self.model_manager.exists(model_key):
                raise KeyError(f"Model '{model_key}' is not registered.")
            observe_local = True
            loaded_before = _observe("loaded_before", lambda: self.model_manager.is_loaded(model_key))
            memory_before = _observe("memory_before", self.memory_sampler)
            if operation == "generate":
                work = lambda: self.model_manager.generate_observed(
                    model_key=model_key, messages=messages, max_new_tokens=max_new_tokens,
                )
            else:
                work = lambda: self.model_manager.load(model_key)
            scheduled = await self.scheduler.run_observed(priority=priority, work=work)
            if operation == "generate":
                generation = scheduled.value
            outcome = "success"
            return generation.text if generation is not None else None
        except asyncio.CancelledError:
            outcome, error_type = "cancelled", "CancelledError"
            raise
        except Exception as exc:
            error_type = type(exc).__name__
            if isinstance(exc, ExternalInferenceBudgetError):
                outcome = "budget_rejected"
                request_budget = exc.request_token_budget
                output_cap = exc.output_tokens
                estimated_input = exc.mandatory_tokens
            raise
        finally:
            if observe_local:
                memory_after = _observe("memory_after", self.memory_sampler)
                loaded_after = _observe("loaded_after", lambda: self.model_manager.is_loaded(model_key))
            execution = scheduled.metrics.execution_seconds if scheduled is not None else None
            generated = generation.generated_tokens if generation is not None else None
            tokens_per_second = (
                generated / execution
                if generated is not None and generated >= 0 and execution is not None and execution > 0
                else None
            )
            metric = InferenceMetric(
                operation=operation, model_key=model_key, priority=int(priority),
                loaded_before=loaded_before, loaded_after=loaded_after,
                queue_wait_seconds=scheduled.metrics.queue_wait_seconds if scheduled is not None else None,
                execution_seconds=execution, total_seconds=time.perf_counter() - started,
                generated_tokens=generated, tokens_per_second=tokens_per_second,
                first_token_seconds=generation.first_token_seconds if generation is not None else None,
                memory_before=memory_before, memory_after=memory_after,
                stage=stage_label, repair_of=repair_label, adapter=adapter, outcome=outcome,
                error_type=error_type, input_message_count=input_message_count,
                estimated_input_tokens=estimated_input, request_token_budget=request_budget,
                output_token_cap=output_cap, dropped_messages=dropped_messages,
            )
            _observe("metric_sink", lambda: self.metric_sink(metric))
