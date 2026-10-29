"""Record the exact child identity before importing or starting the trainer.

The registry saves its claim and inputs before launching this process. A parent
restart can then reconcile this receipt without launching a second candidate.
This shim imports no models and receives no application/provider credentials.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

if __package__:
    from .process_protocol import process_identity, write_json
else:
    from process_protocol import process_identity, write_json


def prepare(run_dir: Path) -> bool:
    """Return whether the claimed trainer may run; cancellation never trains."""
    with (run_dir.parent / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = json.loads((run_dir / "run.json").read_text())
        if record["run_id"] != run_dir.name or record["schema"] != "learning-run.v2":
            raise ValueError("Invalid trainer claim")
        for name, key in (("recipe.json", "recipe_sha256"), ("dataset.jsonl", "input_sha256")):
            if hashlib.sha256((run_dir / name).read_bytes()).hexdigest() != record[key]:
                raise ValueError("Trainer input changed after admission")
        if record["state"] not in {"starting", "running", "outcome_unknown", "cancelling"}:
            return False
        receipt = {key: record[key] for key in ("run_id", "request_sha256", "launch_token")}
        receipt.update(pid=os.getpid(), process_identity=process_identity(os.getpid()))
        if receipt["process_identity"] is None:
            raise OSError("Trainer process identity is unavailable")
        write_json(run_dir / "process.json", receipt)
        return not record.get("cancel_requested", False)


def main() -> int:
    run_dir = Path(sys.argv[1]).resolve()
    if not prepare(run_dir):
        return 0
    trainer = Path(__file__).resolve().parents[2] / "scripts" / "run_candidate_training.py"
    os.execv(sys.executable, [sys.executable, str(trainer), str(run_dir)])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
