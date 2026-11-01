"""Start, stop, check or build the whole Janus stack with one command.

    python deploy/janus.py up     --env-file /path/deployment.env [--gpu] [--machine-profile] [--private-ca]
    python deploy/janus.py stop   --env-file /path/deployment.env
    python deploy/janus.py status --env-file /path/deployment.env
    python deploy/janus.py build  --env-file /path/deployment.env

`up` validates the configuration first (deploy/validate.py), then starts the
services in dependency order and waits for each one's health check: database,
the idempotent schema task (every statement is IF NOT EXISTS), backend and
frontend, then the AI. It never builds or pulls images; run `build` for that.
`stop` stops every container and keeps all data. `--dry-run` prints the steps.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

DEPLOY = Path(__file__).resolve().parent
OVERLAYS = {"gpu": "compose.gpu.yaml", "machine_profile": "compose.profile.yaml", "private_ca": "compose.trust.yaml"}
ACTIONS = ("up", "stop", "status", "build")


def plan(action: str, env_file: Path, *, gpu: bool = False, machine_profile: bool = False,
         private_ca: bool = False) -> list[list[str]]:
    """The exact commands for one action, in order. Nothing is run here."""
    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action}")
    selected = {"gpu": gpu, "machine_profile": machine_profile, "private_ca": private_ca}
    compose = ["docker", "compose", "--env-file", str(env_file), "-f", str(DEPLOY / "compose.yaml")]
    for name, overlay in OVERLAYS.items():
        if selected[name]:
            compose += ["-f", str(DEPLOY / overlay)]
    validate = [sys.executable, "deploy/validate.py", "--env-file", str(env_file)]
    validate += ["--" + name.replace("_", "-") for name, chosen in selected.items() if chosen]
    up = compose + ["up", "-d", "--no-build", "--pull", "never", "--wait", "--wait-timeout"]
    if action == "up":
        return [
            validate,
            up + ["120", "database"],
            compose + ["--profile", "bootstrap", "run", "--rm", "--no-deps", "--pull", "never", "schema"],
            up + ["180", "backend", "frontend"],
            up + ["600", "ai"],
        ]
    if action == "build":
        return [validate, compose + ["build", "backend", "frontend", "ai"]]
    if action == "stop":
        return [compose + ["stop", "--timeout", "60"]]
    return [compose + ["ps", "--all", "--format", "json"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--machine-profile", action="store_true")
    parser.add_argument("--private-ca", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the steps without running them")
    args = parser.parse_args()
    if not args.env_file.is_file():
        print(f"env file not found: {args.env_file}", file=sys.stderr)
        return 2
    steps = plan(args.action, args.env_file.resolve(), gpu=args.gpu, machine_profile=args.machine_profile,
                 private_ca=args.private_ca)
    for step in steps:
        print(" ".join(step), flush=True)
        if args.dry_run:
            continue
        # Each step must succeed before the next starts; the first failure stops the run.
        completed = subprocess.run(step, cwd=DEPLOY.parent)
        if completed.returncode != 0:
            print(f"stopped: step failed with exit code {completed.returncode}", file=sys.stderr)
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
