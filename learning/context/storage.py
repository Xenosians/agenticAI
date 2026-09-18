from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from pydantic import BaseModel


_LOCK = threading.Lock()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def append_jsonl(path: Path, value: BaseModel | dict) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json", by_alias=True) if hasattr(value, "model_dump") else value
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with _LOCK:
        with resolved.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def load_jsonl_models(path: Path, model_type) -> list:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return []
    if not resolved.is_file():
        raise ValueError(f"context ledger path is not a file: {resolved}")
    result = []
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            result.append(model_type.model_validate(payload))
        except Exception as exc:
            raise ValueError(f"invalid context record at {resolved}:{line_number}: {exc}") from exc
    return result
