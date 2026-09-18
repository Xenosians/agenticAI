from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, TypeVar

from pydantic import BaseModel

from learning.evidence.sanitizer import sanitize_value


T = TypeVar("T", bound=BaseModel)
_APPEND_LOCK = threading.Lock()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str | None:
    path = path.expanduser().resolve()
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_directory(directory: Path) -> str:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")

    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(
        (candidate for candidate in directory.rglob("*") if candidate.is_file()),
        key=lambda item: str(item.relative_to(directory)),
    ):
        file_count += 1
        relative = str(path.relative_to(directory))
        file_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update((file_hash or "").encode("ascii"))
        digest.update(str(path.stat().st_size).encode("ascii"))

    if file_count == 0:
        raise ValueError(f"Directory is empty: {directory}")
    return digest.hexdigest()


def _payload(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        raw = value.model_dump(mode="json", by_alias=True)
    else:
        raw = value
    sanitized = sanitize_value(raw)
    if not isinstance(sanitized, dict):
        raise ValueError("Serialized artifact must be an object")
    return sanitized


def append_jsonl(path: Path, value: BaseModel | dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_payload(value), ensure_ascii=False, sort_keys=True)

    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def load_jsonl_models(path: Path, model_type: type[T]) -> list[T]:
    path = path.expanduser().resolve()
    if not path.exists():
        return []
    if not path.is_file():
        raise ValueError(f"JSONL path is not a file: {path}")

    result: list[T] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            result.append(model_type.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return result


def immutable_write_json(path: Path, value: BaseModel | dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(value)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def immutable_write_jsonl(path: Path, values: Iterable[BaseModel | dict[str, Any]]) -> str:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_payload(value), ensure_ascii=False, sort_keys=True) for value in values]
    blob = "".join(f"{line}\n" for line in lines)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_bytes(blob.encode("utf-8"))


def atomic_write_json(path: Path, value: BaseModel | dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(value)

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_json_model(path: Path, model_type: type[T]) -> T:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"JSON artifact does not exist: {path}")
    try:
        return model_type.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ValueError(f"Invalid JSON artifact {path}: {exc}") from exc
