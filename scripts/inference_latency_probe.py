"""Measure configured model loading and warm inference without executing tools.

This is a runtime diagnostic, not a semantic accuracy or V18-03 acceptance gate.
No generated text or credentials are written to the report.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Settings
from subagents.llm.runtime.inference import InferenceCoordinator
from subagents.llm.runtime.model_manager import ModelManager
from subagents.llm.runtime.observability import capture_cuda_memory
from subagents.llm.runtime.scheduler import GpuScheduler, InferencePriority


async def probe(args: argparse.Namespace, report: dict) -> None:
    settings = Settings()
    report["artifact_cache_root"] = str(settings.model_artifact_cache_root)
    report["runtime_versions"] = {}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes"):
        try:
            report["runtime_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["runtime_versions"][package] = None
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    report["source_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    report["working_tree_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    manager = ModelManager(settings)
    keys = args.model_key or [settings.hub_model_key]
    for key in keys:
        if not manager.exists(key):
            raise ValueError(f"Unknown configured model key: {key}")
    report["residency_policy"] = {
        name: getattr(settings, name)
        for name in (
            "model_max_loaded_models", "model_pin_hub",
            "model_shared_base_enabled", "model_artifact_cache_enabled",
            "model_trim_accelerator_cache_after_generation",
        )
    }
    metrics = []
    engine = InferenceCoordinator(manager, GpuScheduler(), metric_sink=metrics.append)
    try:
        for key in keys:
            profile = manager.model_profile(key)
            entry = {
                "model_key": key,
                "backend": profile.backend,
                "quantization": profile.quantization,
                "configured_device_map": profile.device_map,
                "memory_before_load": asdict(capture_cuda_memory()),
                "runs": [],
            }
            report["models"].append(entry)
            print(f"Loading {key}...", flush=True)
            started = time.perf_counter()
            manager.load(key)
            entry["load_seconds"] = time.perf_counter() - started
            entry["memory_after_load"] = asdict(capture_cuda_memory())
            print(f"Load: {entry['load_seconds']:.3f}s", flush=True)
            for index in range(args.runs):
                await engine.generate(
                    model_key=key,
                    messages=[{"role": "user", "content": args.prompt}],
                    max_new_tokens=args.max_new_tokens,
                    priority=InferencePriority.PRIMARY_RESPONSE,
                )
                measurement = asdict(metrics[-1])
                measurement["run"] = index + 1
                entry["runs"].append(measurement)
                print(json.dumps(measurement), flush=True)
    finally:
        manager.unload_all()


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", action="append", help="Repeat for model-switch measurements")
    parser.add_argument("--runs", type=positive_int, default=2)
    parser.add_argument("--max-new-tokens", type=positive_int, default=64)
    parser.add_argument("--prompt", default="Explain what a network firewall does in two sentences.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / ".runtime" / "diagnostics" / f"inference-{stamp}.json"
    report = {
        "created_at": stamp,
        "scope": "model runtime only; no routing, provider execution, or E2E certification",
        "max_new_tokens": args.max_new_tokens,
        "models": [],
    }
    code = 0
    try:
        asyncio.run(probe(args, report))
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        print(f"Probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        code = 1
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {output}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
