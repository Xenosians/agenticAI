"""Train any specialist with the one portable worker (owner, 18 Nov 2026: one trainer; the dataset and model switch).

    python -m learning.cli.train_specialist <name> --device cpu|cuda [--stages sft,dpo] [--settings JSON]

The name picks ``learning/config/specialists/<name>.json``: its reviewed examples, its base model and its
held-out cases. The job runs on this machine with the same worker Proxmox and the idle cycle use (merge, SFT
LoRA/QLoRA under the fit guard, DPO, PPO for the hub; a stage is kept only when strictly better). The candidate
stays in the run folder: Janus's held-out gate decides, and nothing here registers or serves it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from learning.continual.automation.specialists import (SPECIALIST_WORK_ROOT, model_profiles,
                                                       small_specialist_targets)
from learning.worker_jobs import STAGES, build_specialist_job
from learning.worker_transport import build_worker_package, collect_candidate

REPOSITORY = Path(__file__).resolve().parents[2]
SUMMARY = "Train any specialist with the one portable worker; Janus's held-out gate decides."


def specialist_targets() -> dict:
    """Every configured specialist whose base model has local weights, by name (no size limit by hand)."""
    return {target.name: target for target in small_specialist_targets(profiles=model_profiles(),
                                                                         max_weights_gib=float("inf"))}


KAGGLE_TIMEOUT_SECONDS = 3 * 3600


def kaggle_worker():
    """Kaggle's free GPU with the account's API token (environment ``KAGGLE_API_TOKEN``, never an argument)."""
    from learning.kaggle_worker import connect

    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token:
        raise SystemExit("set KAGGLE_API_TOKEN to use --kaggle")
    return connect(token=token, package=build_worker_package(SPECIALIST_WORK_ROOT / "worker-package"))


def _run_on_kaggle(folder: Path, job_id: str, output: Path) -> int:
    kaggle = kaggle_worker()
    started = time.monotonic()
    try:
        kaggle.send(folder, job_id)
        kaggle.start(job_id)
        while kaggle.running(job_id):
            if time.monotonic() - started > KAGGLE_TIMEOUT_SECONDS:
                kaggle.stop(job_id)
                break
            time.sleep(30)
        kaggle.fetch(job_id, output)
    finally:
        kaggle.clean(job_id, keep_result=True)    # nothing stays on Kaggle
    return 0


def _settings(text: str) -> dict:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("--settings is a JSON object, e.g. {\"sft\": {\"epochs\": 2}}")
    return value


def _stages(text: str) -> tuple[str, ...]:
    stages = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [stage for stage in stages if stage not in STAGES]
    if not stages or unknown:
        raise argparse.ArgumentTypeError(f"stages are a comma list of {', '.join(STAGES)}")
    return stages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=SUMMARY)
    parser.add_argument("specialist", help="a name from learning/config/specialists")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--stages", type=_stages, default=("sft", "dpo"))
    parser.add_argument("--settings", type=_settings, default={},
                        help="job settings merged into the recipe, e.g. {\"sft\": {\"epochs\": 3}}")
    parser.add_argument("--work", type=Path, default=SPECIALIST_WORK_ROOT / "by-hand")
    parser.add_argument("--kaggle", action="store_true",
                        help="run on Kaggle's free GPU; a base pinned to a public commit is fetched there")
    args = parser.parse_args(argv)
    if args.kaggle:
        args.device = "cuda"
    targets = specialist_targets()
    if args.specialist not in targets:
        parser.error(f"no specialist named {args.specialist!r}; known: {', '.join(sorted(targets)) or 'none'}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = f"{args.specialist}-{stamp}"
    work = Path(args.work)
    folder = build_specialist_job(targets[args.specialist], profiles=model_profiles(),
                                  directory=work / "jobs" / job_id, stages=args.stages, stamp=stamp,
                                  device=args.device, settings=args.settings, base_by_reference=args.kaggle)
    output = work / "runs" / job_id
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(Path.home()),
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "TOKENIZERS_PARALLELISM": "false"}
    if args.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    try:
        if args.kaggle:
            code = _run_on_kaggle(folder, job_id, output)
        else:
            code = subprocess.run([sys.executable, "-m", "learning.worker.stages", str(folder / "job.json"),
                                   str(output)], cwd=str(REPOSITORY), env=environment).returncode
    finally:
        shutil.rmtree(folder, ignore_errors=True)       # links to the base weights and the data
    result_file = output / "result.json"
    result: dict = {"outcome": "failed", "error": f"the worker ended with exit code {code} and wrote no result"}
    if result_file.is_file():
        loaded = json.loads(result_file.read_text(encoding="utf-8"))
        result = loaded if isinstance(loaded, dict) else result
    candidate = collect_candidate(output) if result.get("outcome") == "trained" else None
    print(json.dumps({
        "specialist": args.specialist, "outcome": result.get("outcome"), "error": result.get("error"),
        "baseline": {key: (result.get("baseline") or {}).get(key) for key in ("passed", "total")},
        "stages": [{key: stage.get(key) for key in ("stage", "ran", "kept", "reason", "passed", "total")}
                   for stage in result.get("stages") or [] if isinstance(stage, dict)],
        "candidate": None if candidate is None else candidate["adapter_directory"], "run": str(output),
        "next": "Janus's held-out gate judges the candidate; it is never served from here.",
    }, indent=2))
    return 0 if code == 0 and result.get("outcome") in {"trained", "no_improvement"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
