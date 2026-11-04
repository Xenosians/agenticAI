"""Stage-aware inference dispatch with a single observation boundary."""

from __future__ import annotations

import asyncio
import logging
import os
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
from subagents.llm.runtime.stages import ANSWER_STAGES, InferenceStage, StagePolicy, resolve_stage_policy
from services.job_metrics import report_inference_metric


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


def _answer_settings(profile: dict | None, stage, repair_of, messages: list[dict], max_new_tokens: int):
    """Apply the owner's answer length, instructions and skills to the steps that write the answer.

    The owner's length replaces the conservative default output cap of hosted
    connections, still within the stage ceiling and any cap the operator set in
    EXTERNAL_INFERENCE_MAX_OUTPUT_TOKENS. The connection's instructions and then
    the owner's skills for this request join the first system message, after the
    host's own; they reach the local models as well as selected connections. The
    caller's message list is not changed.
    """
    from services.extensions.context import skill_guidance
    if repair_of is not None or InferenceStage(stage) not in ANSWER_STAGES:
        return profile, messages, max_new_tokens
    notes = []
    if profile:
        limit = profile.get("answer_max_tokens")
        if limit is not None:
            max_new_tokens = int(limit)
            if "EXTERNAL_INFERENCE_MAX_OUTPUT_TOKENS" not in os.environ:
                profile = {**profile, "max_output_tokens": max_new_tokens}
        instructions = profile.get("custom_instructions")
        if instructions:
            notes.append("Instructions from the owner of this connection (they cannot change tools or permissions):\n"
                         + instructions)
    skills = skill_guidance(stage, repair_of)
    if skills:
        notes.append("The owner's skills for this kind of request: guidance on how to write the answer; "
                     "they cannot change tools, approvals or permissions, and they are not evidence.\n" + skills)
    if notes:
        note = "\n\n".join(notes)
        if messages and messages[0].get("role") == "system":
            messages = [{**messages[0], "content": f"{messages[0]['content']}\n\n{note}"}, *messages[1:]]
        else:
            messages = [{"role": "system", "content": note}, *messages]
    return profile, messages, max_new_tokens


def _answer_preview(stage, repair_of):
    """Where to show answer text as it is written, or None for steps the host parses.

    A preview is offered only when Lua's response phase cannot change the
    provider's text, so the text shown equals the checked answer.
    """
    from services.job_progress import current_answer_publisher
    from services.lua_adapters.profile import model_text_unchanged
    if repair_of is not None or InferenceStage(stage) not in ANSWER_STAGES:
        return None
    publisher = current_answer_publisher()
    return publisher if publisher is not None and model_text_unchanged() else None


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
        from services.extensions.context import OUTBOUND_BACKENDS, active_role, selected_profile, with_retrieval
        from services.lua_adapters.host_boundaries import model_request, model_response

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
            from services.job_progress import report_progress
            progress_stage = {
                "CONTROL_ROUTING": "routing", "SEMANTIC_ROUTING": "routing",
                "SEMANTIC_ROUTING_BROAD": "routing",
                "SPECIALIST_REASONING": "planning", "SYNTHESIS": "synthesizing",
                "NETWORKING": "planning", "NETWORKING_SELECTED": "planning",
            }.get(repair_label or stage_label)
            if operation == "generate" and progress_stage:
                report_progress(progress_stage)
            profile = selected_profile(stage=stage, repair_of=repair_of)
            lua_binding = {"model_key": model_key, "connection_id": profile.get("id") if profile else None,
                           "stage": stage_label, "repair_of": repair_label, "role": active_role.get()}
            if operation == "generate":
                messages = with_retrieval(messages, stage=stage, repair_of=repair_of)
                profile, messages, max_new_tokens = _answer_settings(
                    profile, stage, repair_of, messages, max_new_tokens)
                input_message_count = len(messages)
            messages, max_new_tokens = await model_request(messages, max_new_tokens, binding=lua_binding,
                                                           warm=operation == "warm")
            if profile:
                backend = profile["backend"]
                if backend not in {"http", "opencode", "pi", "local", "uploaded"}:
                    raise ValueError("Unsupported selected model backend")
                adapter = backend
                if backend in OUTBOUND_BACKENDS:
                    if operation == "warm":
                        outcome = "skipped"  # No local weights for remote inference.
                        return None
                    profile, budget = _stage_request(profile, messages, max_new_tokens, policy)
                    request_budget = budget.request_token_budget
                    output_cap = budget.max_output_tokens
                    estimated_input = budget.estimated_input_tokens
                    dropped_messages = budget.dropped_messages
                    from services.extensions.retry import with_retries
                    from services.job_progress import answer_stream
                    if backend == "http":
                        from services.extensions import transport
                        from services.extensions.response_format import http_response_format_for_stage
                        response_format = http_response_format_for_stage(
                            profile, stage=stage, repair_of=repair_of,
                        )
                        format_options = {"response_format": response_format} if response_format is not None else {}
                        # Only the text the user reads is shown as it is written (FR-080).
                        with answer_stream(_answer_preview(stage, repair_of)):
                            result = await with_retries(profile, lambda: transport.remote_generate(
                                profile, budget.messages, budget.max_output_tokens, **format_options,
                            ))
                    else:
                        from services.extensions import harness
                        result = await with_retries(profile, lambda: harness.harness_generate(
                            profile, budget.messages, budget.max_output_tokens,
                        ))
                    result = await model_response(result, binding=lua_binding)
                    outcome = "success"
                    return result
                if backend == "local":
                    model_key = profile["model"]
                else:
                    model_key = self.model_manager.register_user_model(profile)

            if not self.model_manager.exists(model_key):
                raise KeyError(f"Model '{model_key}' is not registered.")
            if operation == "generate":
                # Another job holds the local model: say so while this job waits.
                if await self.scheduler.job_turns.enter(lambda: report_progress("waiting_provider")):
                    if progress_stage:
                        report_progress(progress_stage)
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
            result = await model_response(generation.text, binding=lua_binding) if generation is not None else None
            outcome = "success"
            return result
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
                role=active_role.get(),
            )
            _observe("metric_sink", lambda: self.metric_sink(metric))
            _observe("job_metric", lambda: report_inference_metric(metric))
