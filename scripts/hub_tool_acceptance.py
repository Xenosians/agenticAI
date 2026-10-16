"""Opt-in Hub and specialist proposal checks; never executes ITSM tools.

Unlike the worker-only matrix, the real Hub must infer the semantic contract
from the request. Authored expected targets are used only after inference for
scoring. Approval, provider execution, and durable app flow need separate tests.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tool_acceptance_matrix import (
    ADAPTERS, evaluate_proposal, load_inventory, load_proposal_cases, select_cases,
)


class ProposalBoundary:
    """Capture guarded proposals without creating approvals or invoking MCP."""

    def __init__(self):
        self.calls = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": False, "status": "denied",
                "decision_code": "acceptance_proposal_only",
                "error": "Execution is disabled for Hub proposal acceptance."}


class ObservedInference:
    def __init__(self, inference):
        self.inference = inference
        self.responses = []

    async def generate(self, **kwargs):
        response = await self.inference.generate(**kwargs)
        self.responses.append((kwargs["stage"].value, response))
        return response


def score_case(case, result, responses, calls, *, contracts, agents):
    """Fail on extra routes/calls, worker drift, or a proposal blocked by guards."""
    if len(result.results) != 1 or result.results[0].agent_name != case.agent_name:
        return {"passed": False, "decision_code": "wrong_specialist_or_route_count"}
    worker = result.results[0]
    generated = [raw for stage, raw in responses
                 if stage in {"SPECIALIST_REASONING", "NETWORKING"}]
    if len(generated) != 1:
        return {"passed": False, "decision_code": "unexpected_worker_generation_count"}
    scored = evaluate_proposal(case, generated[0], contracts=contracts, agents=agents)
    if not scored["passed"]:
        return scored
    if (len(calls) != 1 or worker.proposed_tool != case.expected_tool
            or worker.outcome_code != "acceptance_proposal_only"):
        return {"passed": False, "decision_code": "proposal_did_not_reach_boundary"}
    actual = calls[0]
    if actual.get("tool_name") != case.expected_tool:
        return {"passed": False, "decision_code": "unexpected_boundary_tool"}
    # Raw scoring above cannot be rescued by runtime argument binding.
    return {"passed": True, "decision_code": "hub_and_worker_proposal_passed"}


async def observe_cases(*, hub, inference, boundary, cases, contracts, agents,
                        metrics, timeout, record):
    for case in cases:
        inference.responses.clear()
        boundary.calls.clear()
        metrics.clear()
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(hub.run(case.user), timeout=timeout)
            score = score_case(case, result, inference.responses, boundary.calls,
                               contracts=contracts, agents=agents)
        except Exception as exc:
            score = {"passed": False, "decision_code": "inference_or_routing_failure",
                     "error_type": type(exc).__name__}
        row = {"case_id": case.case_id, "case_sha256": case.digest,
               "review_status": case.review_status, "expected_tool": case.expected_tool,
               "tool_executed": False, "approval_created": False,
               "elapsed_seconds": round(time.perf_counter() - started, 3),
               "attempts": [asdict(metric) for metric in metrics], **score}
        record(row)
        if score.get("error_type") == "TimeoutError":
            # A timed-out local worker may still be finishing off-thread. Do not
            # queue more work or conceal it by rerunning the same case.
            return False
    return True


async def run(args):
    from config import Settings
    from services.extensions.context import execution_profile, retrieval_evidence
    from services.extensions.service import extension_store
    from subagents.hub import build_hub
    from subagents.llm.runtime.inference import InferenceCoordinator
    from subagents.llm.runtime.model_manager import ModelManager
    from subagents.llm.runtime.scheduler import GpuScheduler

    contracts, agents = load_inventory()
    cases = load_proposal_cases()
    if args.all_tools:
        # The corpus can contain more than 100 variants. Enumerate target tools
        # through the same validity/overlap rules rather than truncating coverage.
        from scripts.tool_acceptance_matrix import validate_case
        by_tool = {}
        for case in cases:
            if not case.training_overlap and not validate_case(case, contracts, agents):
                by_tool.setdefault(case.expected_tool, case)
        missing = set(contracts) - by_tool.keys()
        if missing:
            raise ValueError("Missing held-out proposal targets: " + ", ".join(sorted(missing)))
        chosen = [by_tool[name] for name in sorted(by_tool)]
    else:
        chosen = select_cases(cases, requested=args.case, limit=args.limit,
                              contracts=contracts, agents=agents)
    profile = None
    if args.adapter != "local":
        if not args.owner or not args.model_id:
            raise ValueError("External acceptance requires an owning account and admitted connection")
        profile = extension_store().get(args.owner, args.model_id, secret=True)
        if (profile.get("kind") != "model" or profile.get("state") != "ready"
                or profile.get("backend") != args.adapter):
            raise ValueError("Connection is not ready for the requested adapter")
    elif args.owner or args.model_id:
        raise ValueError("Local default uses configured Hub/specialist profiles")

    manager = ModelManager(settings=Settings())
    metrics = []
    coordinator = InferenceCoordinator(model_manager=manager, scheduler=GpuScheduler(),
                                       metric_sink=metrics.append)
    observed = ObservedInference(coordinator)
    boundary = ProposalBoundary()
    hub = build_hub(settings=Settings(), model_manager=manager,
                    inference=observed, tool_gateway=boundary)
    report = {"schema": "hub-tool-acceptance.v1", "requested_adapter": args.adapter,
              "scope": "Hub and worker proposals; excludes real policy/provider execution and durable app flow",
              "all_tools_mvp_verified": False, "complete": False,
              "selected_case_count": len(chosen), "results": []}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def record(row):
        report["results"].append(row)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: row[key] for key in
                          ("case_id", "passed", "decision_code", "elapsed_seconds")}), flush=True)

    profile_token = execution_profile.set(profile)
    evidence_token = retrieval_evidence.set([])
    try:
        report["complete"] = await observe_cases(
            hub=hub, inference=observed, boundary=boundary, cases=chosen,
            contracts=contracts, agents=agents, metrics=metrics,
            timeout=args.timeout, record=record)
    finally:
        execution_profile.reset(profile_token)
        retrieval_evidence.reset(evidence_token)
        manager.unload_all()
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return int(not report["complete"] or not all(row["passed"] for row in report["results"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=ADAPTERS, required=True)
    parser.add_argument("--owner", default=os.environ.get("ITSM_ACCEPTANCE_OWNER"))
    parser.add_argument("--model-id")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--all-tools", action="store_true")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--report", type=Path, default=ROOT / ".runtime/hub-tool-acceptance.json")
    args = parser.parse_args()
    if args.all_tools and args.case:
        parser.error("Choose --all-tools or explicit cases")
    if not 1 <= args.timeout <= 300:
        parser.error("Timeout must be between 1 and 300 seconds")
    try:
        return asyncio.run(run(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
