"""Process-local, cumulative admission for external inference requests.

Limits are connection settings, never guesses about an account's subscription.
HTTP, Pi and OpenCode using the same provider/model/key share reservations.
Reservations charge the conservative input plus output estimate for 60 seconds,
including failed/disconnected requests: their provider outcome can be unknown.
This cannot account for other processes or applications using the same account.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import secrets
import threading
import time
from urllib.parse import urlsplit


LIMIT_FIELDS = {
    "provider_tokens_per_minute": (1, 1_000_000_000),
    "provider_requests_per_minute": (1, 1_000_000),
}
DEFAULT_MAX_WAIT_SECONDS = 65.0
MAX_WAIT_SECONDS = 120.0


class ProviderAdmissionError(ValueError):
    """A content-free admission failure suitable for the public boundary."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def validate_admission_settings(profile: dict) -> dict:
    """Return detached, explicitly supplied settings; reject booleans/coercion."""
    result = {}
    for name, (minimum, maximum) in LIMIT_FIELDS.items():
        if name in profile:
            value = profile[name]
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
            result[name] = value
    if "admission_max_wait_seconds" in profile:
        value = profile["admission_max_wait_seconds"]
        if type(value) not in (int, float) or not 0 <= value <= MAX_WAIT_SECONDS:
            raise ValueError("admission_max_wait_seconds must be from 0 to 120")
        result["admission_max_wait_seconds"] = float(value)
    return result


def provider_model_identity(profile: dict) -> tuple[str, str]:
    """Canonical public identity, independent of the inference adapter."""
    backend = profile.get("backend", "http")
    model = profile.get("model", "")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("An external inference model is required")
    if backend in {"pi", "opencode"}:
        provider, separator, model = model.partition("/")
        if not separator or not provider or not model:
            raise ValueError("Harness models require a provider/model identifier")
        return provider, model
    parts = urlsplit(profile.get("url", ""))
    provider = {
        "api.groq.com": "groq",
        "openrouter.ai": "openrouter",
        "api.anthropic.com": "anthropic",
        "generativelanguage.googleapis.com": "google",
        "api.mistral.ai": "mistral",
        "api.openai.com": "openai",
        "api.x.ai": "xai",
        "api.cerebras.ai": "cerebras",
        "opencode.ai": "opencode",
    }.get(parts.hostname or "", f"{parts.scheme}://{parts.netloc}")
    return provider, model


@dataclass(frozen=True)
class AdmissionReceipt:
    estimated_request_tokens: int
    wait_seconds: float
    limits_configured: bool
    _identity: str = field(repr=False)
    _reservation_id: str = field(repr=False)


@dataclass
class _Reservation:
    identifier: str
    timestamp: float
    tokens: int


@dataclass
class _Bucket:
    reservations: deque = field(default_factory=deque)
    waiters: deque = field(default_factory=deque)
    token_limit: int | None = None
    request_limit: int | None = None
    token_total: int = 0
    last_used: float = 0.0


class ProviderAdmission:
    """Bounded FIFO sliding-window reservations, safe across event loops.

    Identity keys are ephemeral HMACs; neither plaintext credentials nor prompt
    content enter limiter state. Conflicting profiles use the lower limit until
    the shared bucket expires after inactivity. An unconfigured profile cannot
    bypass a limit already registered for the same credential/model.
    """

    def __init__(self, *, window_seconds=60.0, max_waiters=64,
                 max_identities=1024, max_reservations=10_000, clock=time.monotonic):
        self._window = window_seconds
        self._max_waiters = max_waiters
        self._max_identities = max_identities
        self._max_reservations = max_reservations
        self._clock = clock
        self._secret = secrets.token_bytes(32)
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}

    def _identity(self, profile):
        credential = profile.get("api_key", "") or ""
        if not isinstance(credential, str):
            raise ValueError("Provider credential must be text")
        value = json.dumps((*provider_model_identity(profile), credential), separators=(",", ":"))
        return hmac.new(self._secret, value.encode(), hashlib.sha256).hexdigest()

    def _prune(self, bucket, now):
        while bucket.reservations and bucket.reservations[0].timestamp <= now - self._window:
            reservation = bucket.reservations.popleft()
            bucket.token_total -= reservation.tokens

    def charge_usage(self, receipt: AdmissionReceipt, total_tokens: object) -> None:
        """Reconcile a reported total upward; never refund estimated capacity.

        Missing/invalid usage is ignored. Receipts identify only the reservation
        and expire with it; a delayed response cannot alter another reservation.
        """
        if type(total_tokens) is not int or total_tokens <= 0:
            return
        with self._lock:
            bucket = self._buckets.get(receipt._identity)
            if bucket is None:
                return
            self._prune(bucket, self._clock())
            for reservation in bucket.reservations:
                if reservation.identifier == receipt._reservation_id:
                    delta = max(0, total_tokens - reservation.tokens)
                    reservation.tokens += delta
                    bucket.token_total += delta
                    return

    async def admit(self, profile: dict, estimated_request_tokens: int) -> AdmissionReceipt:
        if type(estimated_request_tokens) is not int or estimated_request_tokens <= 0:
            raise ValueError("Estimated request tokens must be a positive integer")
        settings = validate_admission_settings(profile)
        identity = self._identity(profile)
        started = self._clock()
        deadline = started + settings.get("admission_max_wait_seconds", DEFAULT_MAX_WAIT_SECONDS)
        ticket = object()
        with self._lock:
            for key, old in list(self._buckets.items()):
                self._prune(old, started)
                if not old.waiters and not old.reservations and old.last_used <= started - self._window:
                    del self._buckets[key]
            if identity not in self._buckets:
                if len(self._buckets) >= self._max_identities:
                    raise ProviderAdmissionError("provider_admission_capacity_exceeded")
                self._buckets[identity] = _Bucket(last_used=started)
            bucket = self._buckets[identity]
            for attribute, name in (("token_limit", "provider_tokens_per_minute"),
                                    ("request_limit", "provider_requests_per_minute")):
                if name in settings:
                    previous = getattr(bucket, attribute)
                    setattr(bucket, attribute, min(previous, settings[name]) if previous else settings[name])
            if bucket.token_limit is not None and estimated_request_tokens > bucket.token_limit:
                raise ProviderAdmissionError("provider_admission_request_too_large")
            if len(bucket.waiters) >= self._max_waiters:
                raise ProviderAdmissionError("provider_admission_queue_full")
            bucket.waiters.append(ticket)

        has_waited = False
        try:
            while True:
                with self._lock:
                    now = self._clock()
                    self._prune(bucket, now)
                    bucket.last_used = now
                    if has_waited and now >= deadline:
                        raise ProviderAdmissionError("provider_admission_wait_exceeded")
                    # A stricter connection can register while this caller waits.
                    if bucket.token_limit is not None and estimated_request_tokens > bucket.token_limit:
                        raise ProviderAdmissionError("provider_admission_request_too_large")
                    token_room = bucket.token_limit is None or bucket.token_total + estimated_request_tokens <= bucket.token_limit
                    request_room = bucket.request_limit is None or len(bucket.reservations) < bucket.request_limit
                    if bucket.waiters[0] is ticket and token_room and request_room and len(bucket.reservations) < self._max_reservations:
                        reservation = _Reservation(secrets.token_hex(16), now, estimated_request_tokens)
                        bucket.reservations.append(reservation)
                        bucket.token_total += estimated_request_tokens
                        bucket.waiters.popleft()
                        return AdmissionReceipt(estimated_request_tokens, now - started,
                                                bucket.token_limit is not None or bucket.request_limit is not None,
                                                identity, reservation.identifier)
                    if now >= deadline:
                        raise ProviderAdmissionError("provider_admission_wait_exceeded")
                    # Short polling also permits prompt cancellation and cross-loop
                    # FIFO changes without binding process state to an event loop.
                    next_release = bucket.reservations[0].timestamp + self._window if bucket.reservations else now + 0.05
                    delay = max(0.001, min(0.05, deadline - now, next_release - now))
                await asyncio.sleep(delay)
                has_waited = True
        finally:
            with self._lock:
                if ticket in bucket.waiters:
                    bucket.waiters.remove(ticket)


provider_admission = ProviderAdmission()


async def admit_external_request(profile: dict, estimated_request_tokens: int) -> AdmissionReceipt:
    return await provider_admission.admit(profile, estimated_request_tokens)


def record_external_usage(receipt: AdmissionReceipt, total_tokens: object) -> None:
    provider_admission.charge_usage(receipt, total_tokens)
