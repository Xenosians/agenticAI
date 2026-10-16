"""Opt-in live routing-contract check; never executes a proposed tool.

Use encrypted, owner-scoped connections already admitted by the application.
This exercises candidate selection, dispatch and semantic validation, not the
Phoenix durable-job/approval lifecycle. No account selection is changed.
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.extensions.context import execution_profile
from services.extensions.service import extension_store
from subagents.core.definitions.loader import load_agent_directory
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.orchestration.router import LLMRouter
from subagents.llm.runtime.inference import InferenceCoordinator


async def check_profile(store, owner, identity):
    profile = store.get(owner, identity, secret=True)
    if profile.get("kind") != "model" or profile.get("state") != "ready":
        raise ValueError("Acceptance requires an admitted model connection")
    if profile.get("backend") not in {"http", "pi", "opencode"}:
        raise ValueError("This check requires an external inference adapter")
    registry = AgentRegistry()
    registry.register_many(load_agent_directory(ROOT / "subagents/agents"))
    metrics = []
    # Local fallback is visible as a failed acceptance, never a silent model
    # load. This fixture does not claim to measure local-model routing quality.
    manager = SimpleNamespace(exists=lambda key: False)
    coordinator = InferenceCoordinator(manager, SimpleNamespace(), metric_sink=metrics.append)
    router = LLMRouter(registry, coordinator, "acceptance-control", strict_contract=True, max_new_tokens=512)
    token = execution_profile.set(profile)
    started = time.perf_counter()
    report = {"adapter": profile["backend"], "model": profile["model"],
              "scope": "routing_contract_only", "tool_executed": False,
              "private_dataset_sent": False, "success": False}
    try:
        routes = await router.route_with_repair("List Jira projects")
        tools = [tool for route in routes for tool in route.semantic_intent.allowed_tools]
        report["proposed_capabilities"] = tools
        report["success"] = (
            len(routes) == 1 and tools == ["jira_project_list"]
            and not routes[0].semantic_intent.clarification_required
        )
    except Exception as exc:
        # Provider errors can contain user data. Persist only classification.
        report["error_type"] = type(exc).__name__
    finally:
        execution_profile.reset(token)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["attempts"] = [asdict(metric) for metric in metrics]
    return report


async def run(args):
    owner = os.environ.get("ITSM_ACCEPTANCE_OWNER", "").strip()
    if not owner:
        raise ValueError("Set ITSM_ACCEPTANCE_OWNER to the connection's account ID")
    store = extension_store()
    reports = [await check_profile(store, owner, identity) for identity in args.profile]
    report = {"schema": "compact-routing-acceptance.v1", "results": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(row["success"] for row in reports) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True, help="Saved model connection ID (repeatable)")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/compact-routing-acceptance.json")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
