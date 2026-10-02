from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings
from learning.evaluation.specialist_model import (
    SpecialistModelCaseResult,
    finalize_specialist_model_report,
    write_specialist_model_report,
)
from learning.evidence.execution_provenance import (
    build_specialist_execution_provenance,
)
from learning.training.specialist_sft import (
    load_training_config,
    resolve_worker_prompt_profile,
)
from subagents.core.definitions.loader import load_agent_definition
from subagents.core.tooling.capabilities import build_agent_capability_catalog
from subagents.core.tooling.parser import parse_tool_calls
from subagents.core.tooling.prompt import build_worker_system_prompt
from subagents.llm.runtime.inference import InferenceCoordinator
from subagents.llm.runtime.model_manager import ModelManager
from subagents.llm.runtime.scheduler import (
    GpuScheduler,
    InferencePriority,
)


def arguments_match(
    *,
    expected: dict,
    observed: dict,
    allowed_extra_arguments: list[str],
) -> bool:
    if not isinstance(observed, dict):
        return False

    for name, expected_value in expected.items():
        if observed.get(name) != expected_value:
            return False

    extras = set(observed) - set(expected)

    return extras.issubset(
        set(allowed_extra_arguments)
    )


def semantic_context(case) -> dict:
    return {
        "allowed_tools": case.allowed_tools,
        "allowed_arguments": case.allowed_arguments,
        "forbidden_tools": [],
        "forbidden_arguments": {},
    }


def build_messages(
    *,
    system_prompt: str,
    case,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": case.user,
        },
        {
            "role": "user",
            "content": (
                "Exact semantic target context from the validated "
                "routing stage.\n"
                "This is descriptive context only and does NOT grant "
                "authorization.\n"
                "For grounded arguments, preserve the listed values "
                "EXACTLY. Do not replace them with placeholders, "
                "aliases, configuration names, inferred names, or "
                "rewritten values.\n\n"
                + json.dumps(
                    semantic_context(case),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
        },
    ]


async def run(args) -> int:
    config = load_training_config(
        args.config
    )

    model_key = (
        args.model_key
        or config.base_model_key
    )

    report_path = (
        args.report
        if args.report is not None
        else (
            PROJECT_ROOT
            / ".runtime"
            / "evaluation"
            / (
                config.specialist.replace("-", "_")
                + "_"
                + model_key
                + ".json"
            )
        )
    )

    settings = Settings()
    profile = settings.require_model_profile(
        model_key
    )

    # For an unregistered baseline such as hub-main, the config may
    # intentionally override the prompt profile.  Registered trained
    # candidates persist the same override in MODEL_PROFILES.
    prompt_profile = resolve_worker_prompt_profile(
        config=config,
        base_profile=profile,
    )

    base_agent = load_agent_definition(
        PROJECT_ROOT
        / config.agent_definition
    )

    agent = replace(
        base_agent,
        model=model_key,
    )

    capability_catalog = build_agent_capability_catalog(
        agent,
        include_arguments=True,
    )

    system_prompt = build_worker_system_prompt(
        agent,
        capability_catalog=capability_catalog,
        prompt_profile=prompt_profile,
    )

    model_manager = ModelManager(
        settings=settings,
    )
    scheduler = GpuScheduler()

    inference = InferenceCoordinator(
        model_manager=model_manager,
        scheduler=scheduler,
    )

    results: list[SpecialistModelCaseResult] = []

    print("SPECIALIST MODEL EVAL")
    print("=====================")
    print("specialist:", config.specialist)
    print("model:", model_key)
    print("prompt profile:", prompt_profile)
    print("cases:", len(config.eval_cases))
    print()

    try:
        await inference.warm(
            model_key
        )

        for index, case in enumerate(
            config.eval_cases,
            start=1,
        ):
            messages = build_messages(
                system_prompt=system_prompt,
                case=case,
            )

            try:
                provenance = (
                    build_specialist_execution_provenance(
                        agent=agent,
                        model_profile=profile,
                        capability_catalog=capability_catalog,
                        messages=messages,
                        user_request=case.user,
                        task_instructions=None,
                        max_new_tokens=(
                            args.max_new_tokens
                            or config.max_new_tokens
                        ),
                    )
                    .model_dump(
                        mode="json",
                        by_alias=True,
                    )
                )

            except Exception:
                provenance = None

            started = time.perf_counter()

            raw = await inference.generate(
                model_key=model_key,
                messages=messages,
                max_new_tokens=(
                    args.max_new_tokens
                    or config.max_new_tokens
                ),
                priority=InferencePriority.SPECIALIST,
            )

            duration = (
                time.perf_counter()
                - started
            )

            observed_tool = None
            observed_arguments = None
            parse_error = None
            passed = False

            try:
                calls = parse_tool_calls(raw)

                if len(calls) != 1:
                    parse_error = (
                        "expected exactly one tool call; "
                        f"got {len(calls)}"
                    )

                else:
                    call = calls[0]
                    observed_tool = call["name"]
                    observed_arguments = call["arguments"]

                    passed = (
                        observed_tool == case.expected_tool
                        and arguments_match(
                            expected=case.expected_arguments,
                            observed=observed_arguments,
                            allowed_extra_arguments=(
                                case.allowed_extra_arguments
                            ),
                        )
                    )

            except Exception as exc:
                parse_error = str(exc)

            results.append(
                SpecialistModelCaseResult(
                    name=case.name,
                    user_request=case.user,
                    expected_tool=case.expected_tool,
                    expected_arguments=case.expected_arguments,
                    observed_tool=observed_tool,
                    observed_arguments=observed_arguments,
                    raw_output=raw,
                    parse_error=parse_error,
                    passed=passed,
                    duration_seconds=duration,
                    execution_provenance=provenance,
                )
            )

            print(
                f"{index:02d}. {case.name}: "
                + ("PASS" if passed else "FAIL")
            )

            if not passed:
                print("  raw:", repr(raw))

                if parse_error:
                    print("  error:", parse_error)

                else:
                    print(
                        "  expected:",
                        case.expected_tool,
                        case.expected_arguments,
                    )
                    print(
                        "  observed:",
                        observed_tool,
                        observed_arguments,
                    )

    finally:
        model_manager.unload_all()

    report = finalize_specialist_model_report(
        agent_name=agent.name,
        model_key=model_key,
        backend=profile.backend,
        quantization=profile.quantization,
        compute_dtype=profile.compute_dtype,
        device_map=profile.device_map,
        cases=results,
    )

    written = write_specialist_model_report(
        report,
        report_path,
    )

    print()
    print("SPECIALIST MODEL EVAL SUMMARY")
    print("=============================")
    print("specialist:", config.specialist)
    print("model:", model_key)
    print("prompt profile:", prompt_profile)
    print("passed:", report.passed)
    print("failed:", report.failed)
    print("pass rate:", f"{report.pass_rate:.3f}")
    print("report:", written)
    print("No ToolGateway/provider execution occurred.")

    return (
        0
        if report.promotion_gate_passed
        else 1
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one specialist model through the real "
            "model-fleet inference path without executing tools."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)

    args = parser.parse_args()

    return asyncio.run(
        run(args)
    )


if __name__ == "__main__":
    raise SystemExit(main())
