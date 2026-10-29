"""Lua row transforms: the optional scripting entry point for dataset admission.

An owner may give a dataset a short Lua script that maps each JSON Lines row to
a new row (or ``nil`` to drop it), for example to rename fields into chat
messages. The script runs in a separate locked-down worker process, never in
this service; its output then goes through the same dataset admission as any
upload. A transform can only reshape text the owner supplied: it has no files,
network, database, credentials, clock, randomness or process access, and its
steps, memory, CPU time and output are bounded.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "lua-row-transform.v1"
WORKER = Path(__file__).with_name("transform_worker.py")
MAX_SOURCE_BYTES = 20_000
MAX_INPUT_BYTES = 2_000_000
MAX_ROWS = 10_000
SOURCE_PATTERN = re.compile(r"\A[\t\n\r\x20-\x7e\u00a0-\U0010ffff]*\Z")


@dataclass(frozen=True)
class TransformLimits:
    instructions: int = 20_000_000
    memory_bytes: int = 64 * 1024 * 1024
    output_bytes: int = 2_000_000
    cpu_seconds: int = 10
    address_space_bytes: int = 768 * 1024 * 1024
    wall_seconds: float = 20.0


class TransformError(ValueError):
    """A curated, content-free failure that may cross the public boundary."""

    def __init__(self, code: str, message: str):
        super().__init__(code)
        self.code = code
        self.message = message

    def public_detail(self) -> dict:
        return {"code": self.code, "message": self.message}


def _rows(text: str) -> list[dict]:
    if not isinstance(text, str) or not 0 < len(text.encode()) <= MAX_INPUT_BYTES:
        raise TransformError("transform_input_too_large", "Rows must be 1 to 2,000,000 UTF-8 bytes.")
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise TransformError("transform_input_invalid", f"Line {number} is not JSON.") from None
        if not isinstance(row, dict):
            raise TransformError("transform_input_invalid", f"Line {number} must be a JSON object.")
        rows.append(row)
    if not rows:
        raise TransformError("transform_input_invalid", "There are no rows to transform.")
    if len(rows) > MAX_ROWS:
        raise TransformError("transform_input_too_large", f"At most {MAX_ROWS} rows can be transformed.")
    return rows


def _source(source: str) -> str:
    if not isinstance(source, str) or not source.strip():
        raise TransformError("transform_script_missing", "Write a script that returns function(row).")
    if len(source.encode()) > MAX_SOURCE_BYTES:
        raise TransformError("transform_script_too_large", "Scripts are limited to 20,000 bytes.")
    if not SOURCE_PATTERN.match(source):
        raise TransformError("transform_script_invalid", "Scripts must be plain text.")
    return source


def worker_environment() -> dict:
    """The worker inherits nothing: no credentials, tokens or service settings."""
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1"}


def run_transform(source: str, text: str, limits: TransformLimits = TransformLimits()) -> dict:
    """Run the owner's script over JSON Lines ``text`` in an isolated worker."""
    request = {"source": _source(source), "rows": _rows(text),
               "limits": {"instructions": limits.instructions, "memory_bytes": limits.memory_bytes,
                          "output_bytes": limits.output_bytes, "cpu_seconds": limits.cpu_seconds,
                          "address_space_bytes": limits.address_space_bytes}}
    payload = json.dumps(request, ensure_ascii=False).encode()
    try:
        completed = subprocess.run([sys.executable, "-I", str(WORKER)], input=payload,
                                   capture_output=True, env=worker_environment(), cwd="/",
                                   timeout=limits.wall_seconds, close_fds=True, start_new_session=True,
                                   check=False)
    except subprocess.TimeoutExpired:
        raise TransformError("transform_time_limit", "The transform took longer than allowed.") from None
    except OSError:
        raise TransformError("transform_unavailable", "Lua transforms are not available on this host.") from None
    try:
        answer = json.loads(completed.stdout or b"null")
    except ValueError:
        answer = None
    if not isinstance(answer, dict):
        if completed.returncode < 0 or completed.returncode in {137, 152}:
            raise TransformError("transform_time_limit", "The transform used more CPU time than allowed.")
        raise TransformError("transform_unavailable", "Lua transforms are not available on this host.")
    if "error" in answer:
        code, message = answer.get("error"), answer.get("message")
        if not isinstance(code, str) or not code.startswith("transform_") or not isinstance(message, str):
            raise TransformError("transform_unavailable", "Lua transforms are not available on this host.")
        raise TransformError(code[:80], message[:300])
    rows = answer.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TransformError("transform_unavailable", "Lua transforms are not available on this host.")
    if not rows:
        raise TransformError("transform_output_empty", "The script dropped every row.")
    output = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    if len(output.encode()) > limits.output_bytes:
        raise TransformError("transform_output_too_large", "The transformed rows exceed the dataset size limit.")
    return {"schema": SCHEMA, "text": output, "input_rows": len(request["rows"]), "output_rows": len(rows),
            "elapsed_ms": answer.get("elapsed_ms"), "isolation": answer.get("isolation")}


def transforms_available() -> bool:
    try:
        import lupa.lua54  # noqa: F401
    except ImportError:
        return False
    return os.name == "posix"
