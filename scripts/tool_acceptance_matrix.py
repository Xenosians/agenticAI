"""Inventory tool acceptance honestly; optionally evaluate bounded worker proposals.

The default command performs no inference or tool execution. ``--generate`` uses
the serving coordinator, but never invokes ToolGateway, approvals, or MCP tools.
Worker cases include a trusted semantic context: passing them does not prove Hub
routing, policy, provider execution, or the browser/durable-job flow.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ADAPTERS = ("local", "http", "pi", "opencode")
SCHEMA = "tool-acceptance-matrix.v1"


def fingerprint(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalized_prompt(value: str) -> str:
    return " ".join(value.casefold().split())


@dataclass(frozen=True)
class ProposalCase:
    case_id: str
    source: str
    source_sha256: str
    agent_name: str
    user: str
    allowed_tools: list[str]
    allowed_arguments: dict[str, list[str]]
    expected_tool: str
    expected_arguments: dict[str, Any]
    allowed_extra_arguments: list[str]
    training_overlap: bool
    review_status: str = "existing_eval"

    @property
    def digest(self) -> str:
        return fingerprint(self.__dict__)


def load_inventory(root: Path = PROJECT_ROOT):
    from mcp_server import register_capability_signatures
    from subagents.core.definitions.loader import load_agent_directory
    from tools.capability_contract import validate_capability_registry
    from tools.registry import TOOLS
    from tools.runtime_schema import runtime_registered_tools

    # Registration only captures callable signatures; no MCP transport starts.
    register_capability_signatures()
    contracts = {item.name: item for item in validate_capability_registry(TOOLS, require_runtime_schema=True)}
    if set(runtime_registered_tools()) != set(contracts):
        raise ValueError("Production catalog and registered executable tools differ")
    agents = {item.name: item for item in load_agent_directory(root / "subagents" / "agents")}
    return contracts, agents


def load_proposal_cases(root: Path = PROJECT_ROOT) -> list[ProposalCase]:
    """Reuse authored eval cases; never derive gold from capability descriptions."""
    configs = []
    training_prompts: set[str] = set()
    for path in sorted((root / "learning" / "config" / "specialists").glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema") != "specialist-training-config.v1":
            continue
        configs.append((path, raw))
        training_prompts.update(normalized_prompt(item["user_request"]) for item in raw.get("examples", []))

    cases = []
    seen = set()
    for path, config in configs:
        for item in config.get("eval_cases", []):
            case_id = f"{path.stem}:{item['name']}"
            if case_id in seen:
                raise ValueError(f"Duplicate proposal case: {case_id}")
            seen.add(case_id)
            cases.append(ProposalCase(
                case_id=case_id,
                source=path.relative_to(root).as_posix(),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                agent_name=config["specialist"],
                user=item["user"],
                allowed_tools=item["allowed_tools"],
                allowed_arguments=item.get("allowed_arguments", {}),
                expected_tool=item["expected_tool"],
                expected_arguments=item["expected_arguments"],
                allowed_extra_arguments=item.get("allowed_extra_arguments", []),
                training_overlap=normalized_prompt(item["user"]) in training_prompts,
            ))
    # This existing suite keeps authored gold in a Python literal. Read it as
    # data instead of importing its training/runtime dependencies or settings.
    jira_path = root / "scripts" / "jira_specialist_model_eval.py"
    if jira_path.exists():
        tree = ast.parse(jira_path.read_text(encoding="utf-8"))
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "CASES" for target in node.targets)]
        if len(assignments) != 1:
            raise ValueError("Jira evaluation must expose exactly one literal CASES assignment")
        for item in ast.literal_eval(assignments[0].value):
            cases.append(ProposalCase(
                case_id=f"jira-specialist:{item['name']}",
                source=jira_path.relative_to(root).as_posix(),
                source_sha256=hashlib.sha256(jira_path.read_bytes()).hexdigest(),
                agent_name="jira-specialist", user=item["user"],
                allowed_tools=item["allowed_tools"], allowed_arguments=item.get("allowed_arguments", {}),
                expected_tool=item["expected_tool"], expected_arguments=item["expected_arguments"],
                allowed_extra_arguments=item.get("allowed_extra_arguments", []),
                training_overlap=normalized_prompt(item["user"]) in training_prompts,
            ))
    acceptance_sources = (
        ("tool_acceptance_reviewed_cases.json", "reviewed", "reviewed"),
        ("tool_acceptance_authored_cases.json", "authored_unreviewed", "authored"),
    )
    for filename, review_status, prefix in acceptance_sources:
        path = root / "config" / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != "tool-proposal-cases.v1"
            or payload.get("review_status") != review_status
        ):
            raise ValueError(
                f"{filename} requires schema tool-proposal-cases.v1 "
                f"and review_status={review_status}"
            )
        for item in payload["cases"]:
            cases.append(ProposalCase(
                case_id=f"{prefix}:{item['id']}",
                source=path.relative_to(root).as_posix(),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                agent_name=item["agent"],
                user=item["user"],
                allowed_tools=item["allowed_tools"],
                allowed_arguments=item.get("allowed_arguments", {}),
                expected_tool=item["expected_tool"],
                expected_arguments=item["expected_arguments"],
                allowed_extra_arguments=item.get("allowed_extra_arguments", []),
                training_overlap=normalized_prompt(item["user"]) in training_prompts,
                review_status=review_status,
            ))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Proposal case IDs must be globally unique")
    return cases


def validate_case(case: ProposalCase, contracts, agents) -> str | None:
    contract = contracts.get(case.expected_tool)
    agent = agents.get(case.agent_name)
    if contract is None:
        return "unknown_expected_tool"
    if agent is None or case.expected_tool not in agent.tools:
        return "invalid_specialist_ownership"
    if not case.user.strip() or case.allowed_tools != [case.expected_tool]:
        return "invalid_semantic_target"
    if set(case.expected_arguments) - set(contract.parameters):
        return "unknown_expected_arguments"
    if set(contract.required_arguments) - set(case.expected_arguments):
        return "missing_expected_arguments"
    if set(case.allowed_extra_arguments) - set(contract.optional_arguments):
        return "invalid_extra_arguments"
    result = evaluate_proposal(
        case, json.dumps([{"name": case.expected_tool, "arguments": case.expected_arguments}]),
        contracts=contracts, agents=agents,
    )
    return None if result["passed"] else f"invalid_gold:{result['decision_code']}"


def evaluate_proposal(case: ProposalCase, raw: str, *, contracts, agents) -> dict:
    from subagents.core.orchestration.intent_contract import parse_semantic_intent
    from subagents.core.orchestration.semantic_guard import SemanticGuard
    from subagents.core.tooling.parser import parse_tool_calls

    def outcome(passed, code):
        return {"passed": passed, "decision_code": code}

    try:
        calls = parse_tool_calls(raw)
    except (TypeError, ValueError):
        return outcome(False, "invalid_tool_json")
    if len(calls) != 1:
        return outcome(False, "invalid_call_count")
    call = calls[0]
    contract = contracts.get(call["name"])
    if contract is None:
        return outcome(False, "unknown_tool")
    if set(call["arguments"]) - set(contract.parameters):
        return outcome(False, "unknown_arguments")
    try:
        intent = parse_semantic_intent({
            "summary": case.user, "effect": contracts[case.expected_tool].effect,
            "allowed_tools": case.allowed_tools, "allowed_arguments": case.allowed_arguments,
            "forbidden_tools": [], "forbidden_arguments": {},
            "max_tool_calls": 1, "clarification_required": False,
        }, agent=agents[case.agent_name], user_request=case.user)
    except ValueError:
        return outcome(False, "invalid_semantic_context")
    decision = SemanticGuard().evaluate(
        agent=agents[case.agent_name], intent=intent,
        tool_name=call["name"], arguments=call["arguments"],
    )
    if not decision.allowed:
        return outcome(False, decision.decision_code)
    if call["name"] != case.expected_tool:
        return outcome(False, "wrong_tool")
    if any(call["arguments"].get(key) != value for key, value in case.expected_arguments.items()):
        return outcome(False, "wrong_arguments")
    if set(call["arguments"]) - set(case.expected_arguments) - set(case.allowed_extra_arguments):
        return outcome(False, "unexpected_extra_arguments")
    return outcome(True, "proposal_matches_gold_and_semantic_guard")


def reference_cases(root: Path = PROJECT_ROOT) -> dict[str, dict[str, list[str]]]:
    """Other suites describe coverage availability, never a completed run."""
    references: dict[str, dict[str, list[str]]] = {}
    for path in sorted((root / "learning" / "evaluation" / "evals").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            tool = raw.get("expected", {}).get("tool")
            if tool:
                category = "orchestrator" if raw.get("target", "orchestrator") == "orchestrator" else "gateway_safety"
                references.setdefault(tool, {}).setdefault(category, []).append(f"{path.stem}:{raw['case_id']}")
    path = root / "config" / "cross_repo_acceptance_cases.json"
    for item in json.loads(path.read_text(encoding="utf-8"))["cases"]:
        if item.get("expected_tool") and item.get("enabled", True):
            references.setdefault(item["expected_tool"], {}).setdefault("durable_job", []).append(item["id"])
    return references


def build_report(*, contracts, agents, cases, references, observations=()) -> dict:
    from tools.registry import get_tool

    valid_cases = {}
    invalid_cases = []
    negative_reference_cases = []
    for case in cases:
        if case.expected_tool == "__none__" and not case.allowed_tools:
            negative_reference_cases.append(case.case_id)
            continue
        error = validate_case(case, contracts, agents)
        if error:
            invalid_cases.append({"case_id": case.case_id, "reason": error})
        else:
            valid_cases[case.case_id] = case

    results = list(observations)
    seen = set()
    for result in results:
        key = (result.get("adapter"), result.get("case_id"))
        case = valid_cases.get(result.get("case_id"))
        if key in seen or key[0] not in ADAPTERS or case is None or result.get("case_sha256") != case.digest:
            raise ValueError("Observation does not identify a unique current valid case and adapter")
        if type(result.get("passed")) is not bool:
            raise ValueError("Observation requires an explicit boolean proposal result")
        seen.add(key)

    rows = []
    for name, contract in sorted(contracts.items()):
        tool_cases = [case for case in valid_cases.values() if case.expected_tool == name]
        existing_eval = [
            case for case in tool_cases
            if not case.training_overlap and case.review_status == "existing_eval"
        ]
        reviewed_acceptance = [
            case for case in tool_cases
            if not case.training_overlap and case.review_status == "reviewed"
        ]
        held_out = existing_eval + reviewed_acceptance
        authored = [
            case for case in tool_cases
            if not case.training_overlap and case.review_status == "authored_unreviewed"
        ]
        case_ids = {case.case_id for case in held_out + authored}
        adapters = {}
        for adapter in ADAPTERS:
            checked = [result for result in results if result["adapter"] == adapter and result["case_id"] in case_ids]
            checked_ids = {result["case_id"] for result in checked}
            proposal = "missing_cases" if not case_ids else "unverified"
            if checked:
                proposal = "failed" if any(not result["passed"] for result in checked) else (
                    ("passed_authored_unreviewed" if authored else "passed") if checked_ids == case_ids else "partial"
                )
            adapters[adapter] = {
                "proposal": proposal,
                "checked_cases": len(checked),
                "approval": "unverified" if contract.requires_approval else "not_required",
                "tool_execution": "unverified",
                "durable_app_flow": "unverified",
            }
        owners = sorted(agent.name for agent in agents.values() if name in agent.tools)
        dedicated_networking = owners == ["networking-specialist"]
        for adapter, state in adapters.items():
            checked = [result for result in results if result["adapter"] == adapter
                       and result["case_id"] in case_ids]
            observed_stages = sorted({result["stage"] for result in checked if result.get("stage")})
            state["effective_adapter"] = (
                adapter if not dedicated_networking or "NETWORKING_SELECTED" in observed_stages else "local"
            )
            state["observed_inference_stages"] = observed_stages
            state["observed_networking_modes"] = sorted({
                result["networking_inference"] for result in checked if result.get("networking_inference")
            })
        rows.append({
            **contract.as_dict(), "specialists": owners,
            "argument_schema": get_tool(name).get("parameters", {}),
            "inference_stage": "NETWORKING" if dedicated_networking else "SPECIALIST_REASONING",
            "held_out_proposal_cases": [case.case_id for case in held_out],
            "existing_eval_proposal_cases": [case.case_id for case in existing_eval],
            "reviewed_acceptance_proposal_cases": [
                case.case_id for case in reviewed_acceptance
            ],
            "authored_unreviewed_proposal_cases": [case.case_id for case in authored],
            "training_overlap_cases": [case.case_id for case in tool_cases if case.training_overlap],
            "available_reference_cases": references.get(name, {}),
            "adapters": adapters,
        })

    missing = [
        row["name"]
        for row in rows
        if not (
            row["held_out_proposal_cases"]
            or row["authored_unreviewed_proposal_cases"]
        )
    ]
    missing_existing = [
        row["name"] for row in rows
        if not row["existing_eval_proposal_cases"]
    ]
    missing_reviewed = [
        row["name"] for row in rows
        if not row["held_out_proposal_cases"]
    ]
    authored_unreviewed_count = sum(
        len(row["authored_unreviewed_proposal_cases"])
        for row in rows
    )
    unowned = [row["name"] for row in rows if not row["specialists"]]
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "worker_proposals_with_trusted_semantic_context",
        "limitations": [
            "Proposal success does not prove Hub routing, approval, provider execution, or durable app flow.",
            "Exact prompts found in authored training examples are excluded from held-out acceptance.",
            "Overlap detection checks version-controlled specialist JSON examples; it cannot prove absence from every model's training data.",
            "Reference cases are available coverage, never evidence of a passing run.",
            "New authored cases are unreviewed test targets; fixture availability and a passing generation do not imply reviewed acceptance.",
            "All-tools MVP remains unverified until every adapter and the real app flow pass acceptance.",
        ],
        "registry_sha256": fingerprint([contracts[name].as_dict() for name in sorted(contracts)]),
        "summary": {
            "registered_tools": len(rows),
            "tools_with_valid_proposal_cases": len(rows) - len(missing),
            "tools_missing_valid_proposal_cases": missing,
            "tools_with_existing_held_out_proposal_cases": len(rows) - len(missing_existing),
            "tools_missing_existing_held_out_proposal_cases": missing_existing,
            "tools_with_reviewed_proposal_cases": len(rows) - len(missing_reviewed),
            "tools_missing_reviewed_proposal_cases": missing_reviewed,
            "authored_unreviewed_cases": authored_unreviewed_count,
            "tools_without_specialist": unowned,
            "invalid_cases": invalid_cases,
            "negative_reference_cases": negative_reference_cases,
            "valid_proposal_cases": len(valid_cases),
            "training_overlap_cases_excluded": sum(case.training_overlap for case in valid_cases.values()),
            "observed_proposals": len(results),
            "all_tools_mvp_verified": False,
        },
        "tools": rows,
        "observations": results,
    }


def select_cases(cases, *, requested, limit, contracts, agents):
    if not 1 <= limit <= 100:
        raise ValueError("Case limit must be between 1 and 100")
    eligible = {case.case_id: case for case in cases if not case.training_overlap and not validate_case(case, contracts, agents)}
    if requested:
        if len(requested) != len(set(requested)) or set(requested) - set(eligible):
            raise ValueError("Requested cases must be unique, valid, non-overlapping case IDs from the matrix")
        chosen = [eligible[identity] for identity in requested]
        if len(chosen) > limit:
            raise ValueError("Requested cases exceed the explicit case limit")
        return chosen
    return list(eligible.values())[:limit]


async def generate_observations(*, inference, adapter, model_key, cases, contracts, agents, max_tokens, timeout,
                                prompt_profile_resolver=None):
    from subagents.core.definitions.registry import AgentRegistry
    from subagents.core.definitions.types import AgentTask
    from subagents.core.orchestration.intent_contract import parse_semantic_intent
    from subagents.core.orchestration.runtime import AgentRuntime
    from subagents.llm.runtime.stages import InferenceStage
    from services.extensions.context import selected_profile, specialist_inference_stage

    class CapturingInference:
        """Record only the in-memory proposal; never expose provider exceptions."""

        def __init__(self):
            self.responses = []
            self.error_type = None

        async def generate(self, **kwargs):
            try:
                raw = await inference.generate(**kwargs)
            except Exception as exc:
                self.error_type = type(exc).__name__
                raise RuntimeError(f"Acceptance inference failed ({self.error_type})") from None
            self.responses.append(raw)
            return raw

    class NeverExecutionGateway:
        """The real execution/approval gateway is deliberately absent."""

        def __init__(self):
            self.proposal_reached_boundary = False

        async def execute(self, **kwargs):
            self.proposal_reached_boundary = True
            return {"ok": False, "status": "blocked", "decision_code": "acceptance_proposal_only",
                    "error": "Tool execution is disabled for proposal acceptance."}

    observations = []
    for case in cases:
        agent = agents[case.agent_name]
        stage = specialist_inference_stage(agent.name)
        profile = selected_profile(stage=stage)
        networking_mode = None
        if agent.name == "networking-specialist":
            from services.extensions.options import networking_inference_mode
            networking_mode = networking_inference_mode(profile or {})
        effective_adapter = (
            "local" if stage == InferenceStage.NETWORKING
            else profile["backend"] if profile is not None else adapter
        )
        if adapter == "local" and stage != InferenceStage.NETWORKING:
            agent = replace(agent, model=model_key)
        registry = AgentRegistry()
        registry.register(agent)
        capture, gateway = CapturingInference(), NeverExecutionGateway()
        runtime = AgentRuntime(agent_registry=registry, inference=capture, tool_gateway=gateway,
                               model_profile_resolver=prompt_profile_resolver, max_new_tokens=max_tokens)
        started = time.perf_counter()
        try:
            intent = parse_semantic_intent({
                "summary": case.user, "effect": contracts[case.expected_tool].effect,
                "allowed_tools": case.allowed_tools, "allowed_arguments": case.allowed_arguments,
                "forbidden_tools": [], "forbidden_arguments": {},
                "max_tool_calls": 1, "clarification_required": False,
            }, agent=agent, user_request=case.user)
            runtime_result = await asyncio.wait_for(runtime.run(AgentTask(
                task_id="proposal-acceptance-" + case.digest[:20], agent_name=agent.name,
                user_request=case.user, semantic_intent=intent,
            )), timeout=timeout)
            if capture.error_type:
                result = {"passed": False, "decision_code": "inference_error", "error_type": capture.error_type}
            elif len(capture.responses) != 1:
                result = {"passed": False, "decision_code": "unexpected_generation_count"}
            else:
                # Score the raw proposal separately from runtime argument
                # binding. A corrected runtime target cannot hide model drift.
                result = evaluate_proposal(case, capture.responses[0], contracts=contracts, agents=agents)
                if result["passed"] and not gateway.proposal_reached_boundary:
                    result = {"passed": False, "decision_code": runtime_result.outcome_code or "runtime_rejected_proposal"}
            result["runtime_outcome_code"] = runtime_result.outcome_code
            result["runtime_semantic_guard_passed"] = gateway.proposal_reached_boundary
        except Exception as exc:
            # Exception messages can contain provider content or credentials.
            result = {"passed": False, "decision_code": "inference_error", "error_type": type(exc).__name__}
        duration = time.perf_counter() - started
        observations.append({"adapter": effective_adapter, "requested_adapter": adapter, "stage": stage.value,
                             "networking_inference": networking_mode,
                             "case_id": case.case_id, "case_sha256": case.digest,
                             "review_status": case.review_status,
                             "model_key": agent.model,
                             "tool_executed": False,
                             "duration_seconds": round(duration, 4), **result})
    return observations


async def run(args) -> int:
    contracts, agents = load_inventory()
    cases = load_proposal_cases()
    observations = []
    if args.generate:
        from config import Settings
        from services.extensions.context import execution_profile, retrieval_evidence
        from subagents.llm.runtime.inference import InferenceCoordinator
        from subagents.llm.runtime.model_manager import ModelManager
        from subagents.llm.runtime.scheduler import GpuScheduler

        if not args.adapter:
            raise ValueError("Generation requires an explicit --adapter")
        selected = select_cases(cases, requested=args.case, limit=args.limit, contracts=contracts, agents=agents)
        profile = None
        if args.adapter != "local":
            if not args.owner or not args.model_id:
                raise ValueError("External generation requires --owner and --model-id from its admitted connection")
            from services.extensions.service import extension_store
            profile = extension_store().get(args.owner, args.model_id, secret=True)
            if profile.get("kind") != "model" or profile.get("state") != "ready" or profile.get("backend") != args.adapter:
                raise ValueError("Stored connection does not match the requested admitted adapter")
        elif args.model_id or args.owner:
            raise ValueError("Local baseline uses --model-key; owner profile overrides are not accepted")
        if not 32 <= args.max_tokens <= 1024 or not 1 <= args.timeout <= 300:
            raise ValueError("Generation bounds: 32..1024 output tokens and 1..300 seconds per case")
        manager = ModelManager(settings=Settings())
        coordinator = InferenceCoordinator(model_manager=manager, scheduler=GpuScheduler())
        profile_token = execution_profile.set(profile)
        evidence_token = retrieval_evidence.set([])
        try:
            observations = await generate_observations(
                inference=coordinator, adapter=args.adapter, model_key=args.model_key,
                cases=selected, contracts=contracts, agents=agents,
                max_tokens=args.max_tokens, timeout=args.timeout,
                prompt_profile_resolver=manager.model_profile,
            )
        finally:
            execution_profile.reset(profile_token)
            retrieval_evidence.reset(evidence_token)
            manager.unload_all()

    report = build_report(contracts=contracts, agents=agents, cases=cases,
                          references=reference_cases(), observations=observations)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(json.dumps({"report": str(args.report), **summary}, indent=2))
    if any(not item["passed"] for item in observations):
        return 1
    if args.require_full_coverage and (
        summary["tools_missing_valid_proposal_cases"]
        or summary["invalid_cases"]
        or summary["tools_without_specialist"]
    ):
        return 1
    if args.require_reviewed_coverage and (
        summary["tools_missing_reviewed_proposal_cases"]
        or summary["authored_unreviewed_cases"]
    ):
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / ".runtime" / "tool-acceptance-matrix.json")
    parser.add_argument(
        "--require-full-coverage",
        action="store_true",
        help="Fail if any tool lacks a valid proposal case",
    )
    parser.add_argument(
        "--require-reviewed-coverage",
        action="store_true",
        help="Fail if any tool lacks reviewed proposal coverage or unreviewed authored cases remain",
    )
    parser.add_argument("--generate", action="store_true", help="Opt in to bounded inference; never executes a tool")
    parser.add_argument("--adapter", choices=ADAPTERS)
    parser.add_argument("--model-key", default="hub-main")
    parser.add_argument("--owner")
    parser.add_argument("--model-id")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not args.generate and any((args.adapter, args.owner, args.model_id, args.case)):
        parser.error("Generation options require explicit --generate")
    try:
        return asyncio.run(run(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
