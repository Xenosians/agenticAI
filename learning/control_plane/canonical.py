"""Canonical values shared by the control-plane store and its socket contract.

One module decides how a job id, a worker id, a timestamp and a payload digest are formed, so the
store and the wire cannot disagree about what "the same job" means.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
KIND = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
WORKER_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
REASON_CODE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
SENDER = WORKER_ID
MESSAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,63}$")
LEASE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
STAGE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")

MAX_DETAIL_CHARS = 500


class ContractError(ValueError):
    """A value that does not belong on the control-plane or socket boundary."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc(value: datetime | None = None) -> datetime:
    """A timezone-aware UTC datetime; a naive one is refused, not guessed."""
    moment = utc_now() if value is None else value
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise ContractError("a timestamp must carry a timezone")
    return moment.astimezone(timezone.utc)


def iso(value: datetime | None = None) -> str:
    return utc(value).isoformat(timespec="microseconds")


def parse_iso(value: str) -> datetime:
    if not isinstance(value, str):
        raise ContractError("a timestamp must be ISO-8601 text")
    try:
        return utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as error:
        raise ContractError("a timestamp must be ISO-8601 with a timezone") from error


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ContractError("a canonical value must be finite JSON") from error


def json_text(value: Any) -> str:
    """The stored form of a payload or result; the digest covers exactly this text."""
    return _json(value)


def digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def require(pattern: re.Pattern[str], value: Any, label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ContractError(f"{label} has an invalid shape")
    return value


def detail(value: str | None) -> str | None:
    """A bounded human reason. It explains a stable reason code; it is never authority."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractError("detail must be text")
    text = " ".join(value.split())
    return text[:MAX_DETAIL_CHARS] or None
