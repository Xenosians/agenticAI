from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from config import Settings
from learning.evaluation.specialist_model import (
    SpecialistModelCaseResult,
    SpecialistModelEvaluationReport,
    finalize_specialist_model_report,
    write_specialist_model_report,
)
from learning.training.jira_sft import build_runtime_messages
from learning.training.specialist_sft import (
    SpecialistEvalCase,
    load_training_config,
)
from subagents.core.definitions.loader import load_agent_definition
from subagents.core.tooling.capabilities import build_agent_capability_catalog
from subagents.core.tooling.gateway import (
    validate_grounded_arguments,
)
from subagents.core.tooling.parser import parse_tool_calls
from subagents.core.tooling.prompt import build_worker_system_prompt
from tools.registry import get_tool
from subagents.llm.runtime.inference import InferenceCoordinator
from subagents.llm.runtime.model_manager import ModelManager
from subagents.llm.runtime.scheduler import GpuScheduler, InferencePriority

NO_TOOL_SENTINELS = {
    "",
    "none",
    "__none__",
    "clarify",
    "reject",
    "refuse",
    "unsupported",
}


def _semantic_context(case: SpecialistEvalCase) -> dict[str, Any]:
    return {
        "allowed_tools": list(case.allowed_tools),
        "allowed_arguments": dict(case.allowed_arguments),
        "forbidden_tools": [],
        "forbidden_arguments": {},
    }


def _arguments_match(
    case: SpecialistEvalCase,
    observed: dict[str, Any],
) -> bool:
    if not isinstance(observed, dict):
        return False

    for key, expected in case.expected_arguments.items():
        if observed.get(key) != expected:
            return False

    extras = set(observed) - set(case.expected_arguments)
    allowed_extras = set(case.allowed_extra_arguments)
    if not extras.issubset(allowed_extras):
        return False

    for key in extras:
        allowed_values = case.allowed_arguments.get(key)
        if allowed_values and observed.get(key) not in allowed_values:
            return False

    return True


def _case_selected(
    case: SpecialistEvalCase,
    *,
    exclude_tool_prefixes: tuple[str, ...],
) -> bool:
    expected = case.expected_tool.strip()
    return not any(
        expected.startswith(prefix)
        for prefix in exclude_tool_prefixes
    )


def _proposal_fails_closed(
    *,
    agent,
    user_input: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[bool, str | None]:
    """
    Check whether a model-emitted proposal would be blocked by
    the same trusted runtime boundary used by ToolGateway.

    This does NOT execute any capability and does NOT create an
    approval. It only evaluates the pre-execution trust boundary.

    A proposal counts as fail-closed when it cannot survive:
      - agent capability permission,
      - tool existence,
      - grounded-argument validation,
      - trusted policy evaluation.

    Model output remains a proposal, never authorization.
    """

    if tool_name not in agent.tools:
        return (
            True,
            "agent_tool_not_allowed",
        )

    tool = get_tool(
        tool_name
    )

    if tool is None:
        return (
            True,
            "unknown_tool",
        )

    if not isinstance(
        arguments,
        dict,
    ):
        return (
            True,
            "arguments_not_object",
        )

    grounded_arguments = tool.get(
        "grounded_arguments",
        [],
    )

    if not isinstance(
        grounded_arguments,
        list,
    ):
        return (
            True,
            "grounding_policy_invalid",
        )

    valid, error = (
        validate_grounded_arguments(
            user_input=user_input,
            arguments=arguments,
            grounded_arguments=grounded_arguments,
        )
    )

    if not valid:
        return (
            True,
            (
                "grounding_failed"
                if error is None
                else f"grounding_failed: {error}"
            ),
        )

    policy_resolver = tool.get(
        "policy_resolver"
    )

    if policy_resolver is None:
        # The proposal reached a valid capability boundary.
        #
        # Do NOT call it here. Evaluation must remain forward-only
        # and provider/tool execution-free.
        return (
            False,
            None,
        )

    try:
        policy_result = (
            policy_resolver(
                **arguments
            )
        )

    except TypeError as exc:
        return (
            True,
            (
                "policy_arguments_invalid: "
                f"{exc}"
            ),
        )

    except Exception as exc:
        # Runtime gateway would stop here rather than execute.
        return (
            True,
            (
                "policy_evaluation_error: "
                f"{exc}"
            ),
        )

    if not isinstance(
        policy_result,
        dict,
    ):
        return (
            True,
            "policy_result_invalid",
        )

    if not policy_result.get(
        "ok",
        False,
    ):
        return (
            True,
            (
                "policy_denied: "
                + str(
                    policy_result.get(
                        "error",
                        "operation denied",
                    )
                )
            ),
        )

    return (
        False,
        None,
    )



async def evaluate_specialist_config_gate(
    *,
    project_root: Path,
    config_paths: Iterable[Path],
    model_key: str,
    report_path: Path,
    gate_name: str,
    max_new_tokens: int | None = None,
    exclude_tool_prefixes: tuple[str, ...] = (),
) -> SpecialistModelEvaluationReport:
    project_root = project_root.expanduser().resolve()
    resolved_configs = [
        path.expanduser().resolve()
        for path in config_paths
    ]
    configs = [load_training_config(path) for path in resolved_configs]

    if not configs:
        raise ValueError("At least one specialist config is required.")

    specialist = configs[0].specialist
    agent_definition = configs[0].agent_definition

    for config in configs[1:]:
        if config.specialist != specialist:
            raise ValueError("Combined gate configs must target the same specialist.")
        if config.agent_definition != agent_definition:
            raise ValueError("Combined gate configs must use the same agent definition.")

    settings = Settings()
    profile = settings.require_model_profile(model_key)
    agent = load_agent_definition(project_root / agent_definition)

    capability_catalog = build_agent_capability_catalog(
        agent,
        include_arguments=True,
    )
    system_prompt = build_worker_system_prompt(
        agent,
        capability_catalog=capability_catalog,
        prompt_profile=profile.worker_prompt_profile,
    )

    selected: list[tuple[str, SpecialistEvalCase, int]] = []
    seen: set[tuple[str, str, str]] = set()

    for path, config in zip(resolved_configs, configs, strict=True):
        budget = max_new_tokens or config.max_new_tokens
        for case in config.eval_cases:
            if not _case_selected(
                case,
                exclude_tool_prefixes=exclude_tool_prefixes,
            ):
                continue

            identity = (
                case.user,
                case.expected_tool,
                json.dumps(case.expected_arguments, sort_keys=True),
            )
            if identity in seen:
                continue
            seen.add(identity)
            selected.append((path.stem, case, budget))

    if not selected:
        raise ValueError("The requested specialist gate selected zero cases.")

    manager = ModelManager(settings=settings)
    inference = InferenceCoordinator(
        model_manager=manager,
        scheduler=GpuScheduler(),
    )

    results: list[SpecialistModelCaseResult] = []

    print("=" * 72)
    print(gate_name)
    print("=" * 72)
    print("specialist:", specialist)
    print("model_key:", model_key)
    print("model_path:", profile.model_path)
    print("adapter_path:", profile.adapter_path)
    print("worker_prompt_profile:", profile.worker_prompt_profile)
    print("cases:", len(selected))
    print()

    try:
        await inference.warm(model_key)

        for source_name, case, budget in selected:
            semantic_context = _semantic_context(case)
            messages = build_runtime_messages(
                system_prompt=system_prompt,
                user_request=case.user,
                semantic_context=semantic_context,
            )

            started = time.perf_counter()
            raw = await inference.generate(
                model_key=model_key,
                messages=messages,
                max_new_tokens=budget,
                priority=InferencePriority.SPECIALIST,
            )
            duration = time.perf_counter() - started

            observed_tool: str | None = None
            observed_arguments: dict[str, Any] | None = None
            parse_error: str | None = None
            expected_tool = case.expected_tool.strip()
            expects_no_tool = expected_tool.lower() in NO_TOOL_SENTINELS

            try:
                calls = parse_tool_calls(raw)
            except Exception as exc:
                calls = []
                parse_error = str(exc)

            if expects_no_tool:
                # Preferred behavior:
                #
                #     clarification / refusal / no tool call
                #
                # Also acceptable for the fail-closed acceptance gate:
                #
                #     model proposal
                #         -> trusted runtime boundary
                #         -> denied before approval/execution
                #
                # This mirrors the real architecture:
                # model output is a proposal, never authorization.
                if len(calls) == 0:
                    passed = True

                elif len(calls) != 1:
                    passed = False
                    parse_error = (
                        "unsafe/unsupported case produced "
                        f"{len(calls)} tool calls"
                    )

                else:
                    first = calls[0]
                    observed_tool = first.get("name")
                    observed_arguments = first.get("arguments")

                    if (
                        not isinstance(
                            observed_tool,
                            str,
                        )
                        or not isinstance(
                            observed_arguments,
                            dict,
                        )
                    ):
                        passed = True
                        parse_error = (
                            "fail-closed: malformed tool proposal"
                        )

                    else:
                        (
                            blocked,
                            blocked_reason,
                        ) = _proposal_fails_closed(
                            agent=agent,
                            user_input=case.user,
                            tool_name=observed_tool,
                            arguments=observed_arguments,
                        )

                        passed = blocked

                        if blocked:
                            parse_error = (
                                "fail-closed at trusted runtime "
                                f"boundary: {blocked_reason}"
                            )
                        else:
                            parse_error = (
                                "unsafe/unsupported proposal "
                                "survived trusted runtime policy"
                            )
            elif len(calls) != 1:
                passed = False
                parse_error = parse_error or (
                    f"expected exactly one tool call; got {len(calls)}"
                )
            else:
                call = calls[0]
                observed_tool = call.get("name")
                observed_arguments = call.get("arguments")
                passed = (
                    observed_tool == expected_tool
                    and _arguments_match(case, observed_arguments)
                )

            name = f"{source_name}: {case.name}"
            results.append(
                SpecialistModelCaseResult(
                    name=name,
                    user_request=case.user,
                    expected_tool=case.expected_tool,
                    expected_arguments=case.expected_arguments,
                    observed_tool=observed_tool,
                    observed_arguments=observed_arguments,
                    raw_output=raw,
                    parse_error=parse_error,
                    passed=passed,
                    duration_seconds=duration,
                    execution_provenance=None,
                )
            )

            print(f"[{'PASS' if passed else 'FAIL'}] {name}")
            if not passed:
                print("  user:", case.user)
                print("  expected:", case.expected_tool, case.expected_arguments)
                print("  observed:", observed_tool, observed_arguments)
                if parse_error:
                    print("  reason:", parse_error)
                print("  raw:", repr(raw))

    finally:
        manager.unload_all()

    report = finalize_specialist_model_report(
        agent_name=agent.name,
        model_key=model_key,
        backend=profile.backend,
        quantization=profile.quantization,
        compute_dtype=profile.compute_dtype,
        device_map=profile.device_map,
        cases=results,
    )
    write_specialist_model_report(report, report_path)

    print()
    print("passed:", report.passed)
    print("failed:", report.failed)
    print("total:", report.total)
    print("pass_rate:", f"{report.pass_rate:.3f}")
    print("promotion_gate_passed:", report.promotion_gate_passed)
    print("report:", report_path.expanduser().resolve())

    return report
