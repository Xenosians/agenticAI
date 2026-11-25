"""The versioned socket contract between the control plane and a GPU worker.

Every message is a frozen, extra-forbidding Pydantic model with the same envelope: protocol,
version, type, message id, sender, send time and job id. A lease token is the fence on every
message that can change a job. Parsing refuses an unknown type, an extra field, a naive
timestamp, a bad fraction or a digest that does not cover the value it names, before the store
is asked to do anything.

The socket is a notification and trigger path. ``succeeded`` and ``failed`` carry the worker's
observed result, but durable completion is still the store transition the control plane makes
after it validates that message; an acknowledgement is never execution evidence.
"""
from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, ClassVar, Literal, Self, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from learning.control_plane import canonical

PROTOCOL = "janus-learning-control"
VERSION = 1
MAX_MESSAGE_BYTES = 1_048_576

Direction = Literal["to_worker", "to_control"]


def _job_id(value: str) -> str:
    return canonical.require(canonical.JOB_ID, value, "job id")


def _message_id(value: str) -> str:
    return canonical.require(canonical.MESSAGE_ID, value, "message id")


def _sender(value: str) -> str:
    return canonical.require(canonical.SENDER, value, "sender")


def _lease_token(value: str) -> str:
    return canonical.require(canonical.LEASE_TOKEN, value, "lease token")


def _kind(value: str) -> str:
    return canonical.require(canonical.KIND, value, "job kind")


def _stage(value: str) -> str:
    return canonical.require(canonical.STAGE, value, "stage")


def _reason_code(value: str) -> str:
    return canonical.require(canonical.REASON_CODE, value, "reason code")


def _sha256(value: str) -> str:
    return canonical.require(canonical.SHA256, value, "sha256")


def _timestamp(value: str) -> str:
    return canonical.iso(canonical.parse_iso(value))


JobId = Annotated[str, AfterValidator(_job_id)]
MessageId = Annotated[str, AfterValidator(_message_id)]
Sender = Annotated[str, AfterValidator(_sender)]
LeaseToken = Annotated[str, AfterValidator(_lease_token)]
Kind = Annotated[str, AfterValidator(_kind)]
Stage = Annotated[str, AfterValidator(_stage)]
ReasonCode = Annotated[str, AfterValidator(_reason_code)]
Sha256 = Annotated[str, AfterValidator(_sha256)]
UtcStamp = Annotated[str, AfterValidator(_timestamp)]
Attempt = Annotated[int, Field(strict=True, ge=1)]


def _fraction(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("a progress fraction must be a number between 0 and 1")
    return value


Fraction = Annotated[float, Field(ge=0.0, le=1.0), AfterValidator(lambda value: float(value))]


def new_message_id() -> str:
    return uuid.uuid4().hex


class Message(BaseModel):
    """The common envelope. A subclass narrows ``type`` and declares its direction."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    protocol: Literal["janus-learning-control"] = PROTOCOL
    version: Literal[1] = VERSION
    type: str
    message_id: MessageId
    sender: Sender
    sent_at: UtcStamp
    job_id: JobId

    DIRECTION: ClassVar[Direction]

    @classmethod
    def create(cls, *, sender: str, job_id: str, **values: Any) -> Self:
        """Build a message with a fresh id and the current time; callers never reuse the id."""
        values.setdefault("sent_at", canonical.iso())
        return cls(
            message_id=new_message_id(),
            sender=sender,
            job_id=job_id,
            **values,
        )


class LeaseMessage(Message):
    """Control plane to worker: this job, this attempt, this token, this deadline."""

    DIRECTION: ClassVar[Direction] = "to_worker"

    type: Literal["lease"] = "lease"
    worker_id: Sender
    lease_token: LeaseToken
    expires_at: UtcStamp
    attempt: Attempt
    kind: Kind
    payload: dict[str, Any]
    payload_sha256: Sha256 | None = None

    @model_validator(mode="before")
    @classmethod
    def _fill_digest(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("payload_sha256") is None and "payload" in data:
            return {**data, "payload_sha256": canonical.digest(data["payload"])}
        return data

    @model_validator(mode="after")
    def _digest_covers_payload(self) -> Self:
        if canonical.digest(self.payload) != self.payload_sha256:
            raise ValueError("lease payload digest does not cover the payload")
        return self


class StartedMessage(Message):
    """Worker to control plane: execution began. A transport reply never produces this message."""

    DIRECTION: ClassVar[Direction] = "to_control"

    type: Literal["started"] = "started"
    lease_token: LeaseToken
    attempt: Attempt


class ProgressMessage(Message):
    """Worker to control plane: the current stage and how far the attempt has come."""

    DIRECTION: ClassVar[Direction] = "to_control"

    type: Literal["progress"] = "progress"
    lease_token: LeaseToken
    attempt: Attempt
    stage: Stage
    fraction: Fraction
    detail: str | None = Field(default=None, max_length=canonical.MAX_DETAIL_CHARS)

    @field_validator("fraction", mode="before")
    @classmethod
    def _check_fraction(cls, value: Any) -> Any:
        return _fraction(value)


class SucceededMessage(Message):
    """Worker to control plane: the result it observed, with a digest over exactly that result."""

    DIRECTION: ClassVar[Direction] = "to_control"

    type: Literal["succeeded"] = "succeeded"
    lease_token: LeaseToken
    attempt: Attempt
    result: dict[str, Any]
    result_sha256: Sha256 | None = None

    @model_validator(mode="before")
    @classmethod
    def _fill_digest(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("result_sha256") is None and "result" in data:
            return {**data, "result_sha256": canonical.digest(data["result"])}
        return data

    @model_validator(mode="after")
    def _digest_covers_result(self) -> Self:
        if canonical.digest(self.result) != self.result_sha256:
            raise ValueError("result digest does not cover the result")
        return self


class FailedMessage(Message):
    """Worker to control plane: a known failure and whether the worker believes a retry is safe."""

    DIRECTION: ClassVar[Direction] = "to_control"

    type: Literal["failed"] = "failed"
    lease_token: LeaseToken
    attempt: Attempt
    reason_code: ReasonCode
    detail: str | None = Field(default=None, max_length=canonical.MAX_DETAIL_CHARS)
    retryable: StrictBool = False


class HeartbeatMessage(Message):
    """Worker to control plane: keep this lease alive while the attempt is still running."""

    DIRECTION: ClassVar[Direction] = "to_control"

    type: Literal["heartbeat"] = "heartbeat"
    lease_token: LeaseToken
    attempt: Attempt


class RenewedMessage(Message):
    """Control plane to worker: the lease now lasts until ``expires_at``."""

    DIRECTION: ClassVar[Direction] = "to_worker"

    type: Literal["renewed"] = "renewed"
    lease_token: LeaseToken
    attempt: Attempt
    expires_at: UtcStamp


class LeaseLostMessage(Message):
    """Control plane to worker: stop; this token no longer owns the job."""

    DIRECTION: ClassVar[Direction] = "to_worker"

    type: Literal["lease_lost"] = "lease_lost"
    lease_token: LeaseToken
    attempt: Attempt
    reason_code: ReasonCode
    detail: str | None = Field(default=None, max_length=canonical.MAX_DETAIL_CHARS)


class ReconcileRequestMessage(Message):
    """Control plane to worker: report whether you observed this job and how it ended."""

    DIRECTION: ClassVar[Direction] = "to_worker"

    type: Literal["reconcile_request"] = "reconcile_request"
    reason_code: ReasonCode
    detail: str | None = Field(default=None, max_length=canonical.MAX_DETAIL_CHARS)


class ReconcileResultMessage(Message):
    """Worker to control plane: what it knows about an uncertain job. Never authority by itself."""

    DIRECTION: ClassVar[Direction] = "to_control"

    type: Literal["reconcile_result"] = "reconcile_result"
    outcome: Literal["succeeded", "failed", "absent"]
    result: dict[str, Any] | None = None
    result_sha256: Sha256 | None = None
    reason_code: ReasonCode | None = None
    detail: str | None = Field(default=None, max_length=canonical.MAX_DETAIL_CHARS)

    @model_validator(mode="before")
    @classmethod
    def _fill_digest(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and data.get("outcome") == "succeeded"
            and data.get("result_sha256") is None
            and "result" in data
            and data["result"] is not None
        ):
            return {**data, "result_sha256": canonical.digest(data["result"])}
        return data

    @model_validator(mode="after")
    def _outcome_is_consistent(self) -> Self:
        if self.outcome == "succeeded":
            if self.result is None or self.result_sha256 is None:
                raise ValueError("a reconciled success must carry its result and digest")
            if canonical.digest(self.result) != self.result_sha256:
                raise ValueError("result digest does not cover the result")
        elif self.result is not None or self.result_sha256 is not None:
            raise ValueError("only a reconciled success may carry a result")
        if self.outcome == "failed" and self.reason_code is None:
            raise ValueError("a reconciled failure must carry its reason code")
        return self


MessageType = Union[
    LeaseMessage,
    StartedMessage,
    ProgressMessage,
    SucceededMessage,
    FailedMessage,
    HeartbeatMessage,
    RenewedMessage,
    LeaseLostMessage,
    ReconcileRequestMessage,
    ReconcileResultMessage,
]

ALL_MESSAGES = (
    LeaseMessage,
    StartedMessage,
    ProgressMessage,
    SucceededMessage,
    FailedMessage,
    HeartbeatMessage,
    RenewedMessage,
    LeaseLostMessage,
    ReconcileRequestMessage,
    ReconcileResultMessage,
)
INBOUND_TYPES = tuple(kind for kind in ALL_MESSAGES if kind.DIRECTION == "to_control")
OUTBOUND_TYPES = tuple(kind for kind in ALL_MESSAGES if kind.DIRECTION == "to_worker")

_MESSAGE = TypeAdapter(Annotated[MessageType, Field(discriminator="type")])


def message_json(message: MessageType) -> str:
    """The canonical wire text; the same value always produces the same bytes."""
    text = canonical.json_text(message.model_dump(mode="json"))
    if len(text.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise canonical.ContractError("a control-plane message is too large")
    return text


def parse_message(text: str | bytes) -> MessageType:
    """Parse exactly one message; oversized, unknown, extra-field or malformed text is refused."""
    raw = text.encode("utf-8") if isinstance(text, str) else text
    if not isinstance(raw, bytes):
        raise canonical.ContractError("a control-plane message must be text or bytes")
    if len(raw) > MAX_MESSAGE_BYTES:
        raise canonical.ContractError("a control-plane message is too large")
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise canonical.ContractError("a control-plane message must be one JSON object") from error
    if not isinstance(data, dict):
        raise canonical.ContractError("a control-plane message must be one JSON object")
    try:
        return _MESSAGE.validate_python(data)
    except ValidationError as error:
        raise canonical.ContractError(f"control-plane message refused: {_summary(error)}") from error


def parse_inbound(text: str | bytes) -> MessageType:
    message = parse_message(text)
    if type(message) not in INBOUND_TYPES:
        raise canonical.ContractError(f"{message.type!r} is not a worker-to-control-plane message")
    return message


def parse_outbound(text: str | bytes) -> MessageType:
    message = parse_message(text)
    if type(message) not in OUTBOUND_TYPES:
        raise canonical.ContractError(f"{message.type!r} is not a control-plane-to-worker message")
    return message


def _summary(error: ValidationError) -> str:
    first = error.errors()[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "message"
    reason = str(first.get("msg", "invalid value")).splitlines()[0]
    return f"{location}: {reason}"[:canonical.MAX_DETAIL_CHARS]
