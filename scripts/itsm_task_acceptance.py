"""Opt-in live ITSM task acceptance through the real AI orchestration graph.

Exercises an admitted external model connection, the trusted guards, real MCP
reads and durable approval proposals. Mutations are never executed by this
runner. Phoenix delivery and browser rendering require their separate checks.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.approval_store import ApprovalStore
from services.extensions.context import execution_profile
from services.extensions.service import extension_store
from subagents.core.orchestration.intent_contract import trusted_tool_effect
from subagents.integration.support import build_hub_integration_context
from tools.registry import get_tool


class ReadOnlyExecution:
    """Additional acceptance-only restriction beyond normal application policy."""

    def __init__(self, mcp):
        self.mcp = mcp
        self.calls = []

    async def call_tool(self, name, arguments):
        tool = get_tool(name)
        if tool is None or trusted_tool_effect(tool) != "read":
            raise RuntimeError("Acceptance runner does not execute mutations")
        result = await self.mcp.call_tool(name, arguments)
        self.calls.append(name)
        return result


def _unexpected_local_inference(**kwargs):
    raise RuntimeError("Selected-adapter acceptance encountered local fallback")


async def check_profile(owner, identity, case):
    profile = extension_store().get(owner, identity, secret=True)
    if (profile.get("kind") != "model" or profile.get("state") != "ready"
            or profile.get("backend") not in {"http", "pi", "opencode"}):
        raise ValueError("Select an admitted external model connection")
    metrics = []
    report = {"case": case, "adapter": profile["backend"], "model": profile["model"],
              "success": False, "mutation_executed": False}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="itsm-task-acceptance-") as directory:
        approval_path = Path(directory) / "approvals.sqlite3"
        runtime = build_hub_integration_context(approval_db_path=approval_path)
        guarded_mcp = ReadOnlyExecution(runtime.mcp)
        runtime.tool_gateway.mcp = guarded_mcp
        runtime.approvals.mcp = guarded_mcp
        runtime.inference.metric_sink = metrics.append
        # A miss is a visible failed acceptance; do not silently load local
        # weights or claim this checks local-model routing quality.
        runtime.model_manager.generate_observed = _unexpected_local_inference
        if case == "read":
            prompt, expected = "List Jira projects", "jira_project_list"
        else:
            target = "itsm_acceptance_" + uuid.uuid4().hex[:12]
            prompt = f"Create a new directory named {target} in the developer workspace."
            expected = "workspace_mkdir"
        token = execution_profile.set(profile)
        try:
            await runtime.mcp.start()
            result = await runtime.hub.run(prompt)
            report["status"] = result.status
            report["workers"] = [{"agent": item.agent_name, "status": item.status,
                "tool": item.proposed_tool, "outcome_code": item.outcome_code,
                "structured_result": item.tool_result is not None} for item in result.results]
            report["answer_generated"] = bool(result.answer)
            if len(result.results) == 1:
                worker = result.results[0]
                correct_tool = worker.proposed_tool == expected
                if case == "read":
                    report["success"] = (correct_tool and result.status == "success"
                        and worker.status == "success" and worker.tool_result is not None
                        and guarded_mcp.calls == [expected] and bool(result.answer))
                elif worker.approval_id:
                    saved = runtime.approval_store.get(worker.approval_id)
                    # Reopen the store to verify persisted state, including any
                    # application-owned snapshot fields, without approving it.
                    reopened = ApprovalStore(approval_path).get(worker.approval_id)
                    persisted = (saved is not None and reopened == saved and saved.status == "pending"
                        and saved.tool == expected
                        and all(saved.arguments.get(key) == value
                                for key, value in (worker.proposed_arguments or {}).items()))
                    report["pending_proposal_survives_reopen"] = persisted
                    report["success"] = (correct_tool and result.status == "approval_required"
                        and worker.status == "approval_required" and persisted
                        and not guarded_mcp.calls)
        except Exception as exc:
            report["error_type"] = type(exc).__name__
            report["error_category"] = getattr(exc, "category", None)
        finally:
            execution_profile.reset(token)
            await runtime.mcp.stop()
        report["executed_read_tools"] = guarded_mcp.calls
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["attempts"] = [asdict(metric) for metric in metrics]
    return report


async def run(args):
    owner = os.environ.get("ITSM_ACCEPTANCE_OWNER", "").strip()
    if not owner:
        raise ValueError("Set ITSM_ACCEPTANCE_OWNER to the connection's account ID")
    cases = ["read", "approval"] if args.case == "all" else [args.case]
    results = []
    report = {"schema": "itsm-task-acceptance.v1", "scope": "AI orchestration; excludes Phoenix/browser",
              "complete": False, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for identity in args.profile:
        for case in cases:
            results.append(await check_profile(owner, identity, case))
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(not all(result["success"] for result in results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True, help="Saved external connection ID (repeatable)")
    parser.add_argument("--case", choices=["read", "approval", "all"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/itsm-task-acceptance.json")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
