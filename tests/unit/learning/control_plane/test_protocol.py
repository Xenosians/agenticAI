"""The socket contract: one envelope, typed messages, and refusal before the store."""
from datetime import datetime, timedelta, timezone
import json

import pytest
from pydantic import ValidationError

from learning.control_plane import canonical, protocol

JOB = "specialist-20261125T000000Z"
TOKEN = "01234567-89ab-cdef-0123-456789abcdef"
NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
EXPIRES = canonical.iso(NOW + timedelta(seconds=300))


def from_worker(message_type, **overrides):
    fields = {"sender": "worker-a", "job_id": JOB}
    if "lease_token" in message_type.model_fields:
        fields["lease_token"] = TOKEN
        fields["attempt"] = 1
    fields.update(overrides)
    return message_type.create(**fields)


def from_control(message_type, **overrides):
    fields = {"sender": "control-plane", "job_id": JOB}
    if "lease_token" in message_type.model_fields:
        fields["lease_token"] = TOKEN
        fields["attempt"] = 1
    fields.update(overrides)
    return message_type.create(**fields)


def a_lease(**overrides):
    fields = {
        "worker_id": "worker-a",
        "lease_token": TOKEN,
        "expires_at": EXPIRES,
        "attempt": 1,
        "kind": "specialist",
        "payload": {"dataset": "d1", "epochs": 1},
    }
    fields.update(overrides)
    return from_control(protocol.LeaseMessage, **fields)


def every_message():
    return [
        a_lease(),
        from_worker(protocol.StartedMessage),
        from_worker(protocol.ProgressMessage, stage="sft", fraction=0.25, detail="step 1"),
        from_worker(protocol.SucceededMessage, result={"adapter": "a"}),
        from_worker(protocol.FailedMessage, reason_code="cuda_oom", retryable=True,
                    detail="out of memory"),
        from_worker(protocol.HeartbeatMessage),
        from_control(protocol.RenewedMessage, lease_token=TOKEN, attempt=1, expires_at=EXPIRES),
        from_control(protocol.LeaseLostMessage, lease_token=TOKEN, attempt=1,
                     reason_code="lease_expired"),
        from_control(protocol.ReconcileRequestMessage, reason_code="lease_expired"),
        from_worker(protocol.ReconcileResultMessage, outcome="succeeded", result={"adapter": "a"}),
        from_worker(protocol.ReconcileResultMessage, outcome="failed", reason_code="crashed"),
        from_worker(protocol.ReconcileResultMessage, outcome="absent"),
    ]


def test_every_message_round_trips_through_canonical_json():
    for message in every_message():
        text = protocol.message_json(message)
        parsed = protocol.parse_message(text)
        assert type(parsed) is type(message)
        assert parsed == message
        assert protocol.message_json(parsed) == text


def test_a_lease_carries_the_digest_of_its_payload():
    message = a_lease()
    assert message.payload_sha256 == canonical.digest(message.payload)


def test_a_lease_whose_payload_no_longer_matches_its_digest_is_refused():
    with pytest.raises(ValidationError):
        a_lease(payload_sha256="0" * 64)

    data = json.loads(protocol.message_json(a_lease()))
    data["payload"]["epochs"] = 2
    with pytest.raises(canonical.ContractError):
        protocol.parse_message(json.dumps(data))


def test_a_success_whose_result_no_longer_matches_its_digest_is_refused():
    with pytest.raises(ValidationError):
        from_worker(protocol.SucceededMessage, result={"adapter": "a"}, result_sha256="f" * 64)

    message = from_worker(protocol.SucceededMessage, result={"adapter": "a"})
    assert message.result_sha256 == canonical.digest({"adapter": "a"})


def test_an_unknown_type_or_an_extra_field_is_refused():
    data = json.loads(protocol.message_json(from_worker(protocol.StartedMessage)))

    for changed in ({**data, "type": "reboot"}, {**data, "surprise": True}):
        with pytest.raises(canonical.ContractError):
            protocol.parse_message(json.dumps(changed))


def test_a_missing_envelope_field_is_refused():
    data = json.loads(protocol.message_json(from_worker(protocol.StartedMessage)))
    del data["sender"]

    with pytest.raises(canonical.ContractError):
        protocol.parse_message(json.dumps(data))


@pytest.mark.parametrize("stamp", ["2026-11-25T12:00:00", "yesterday", 12])
def test_a_timestamp_without_a_timezone_is_refused(stamp):
    with pytest.raises(ValidationError):
        from_worker(protocol.StartedMessage, **{"sent_at": stamp})

    data = json.loads(protocol.message_json(from_worker(protocol.StartedMessage)))
    data["sent_at"] = stamp
    with pytest.raises(canonical.ContractError):
        protocol.parse_message(json.dumps(data))


def test_a_timestamp_is_stored_in_utc():
    message = from_worker(protocol.StartedMessage,
                          sent_at="2026-11-25T19:00:00+07:00")
    assert message.sent_at == "2026-11-25T12:00:00.000000+00:00"


@pytest.mark.parametrize("fraction", [1.5, -0.1, True, "0.5"])
def test_a_fraction_outside_zero_to_one_is_refused(fraction):
    with pytest.raises(ValidationError):
        from_worker(protocol.ProgressMessage, stage="sft", fraction=fraction)

    data = json.loads(protocol.message_json(
        from_worker(protocol.ProgressMessage, stage="sft", fraction=0.5)))
    data["fraction"] = fraction
    with pytest.raises(canonical.ContractError):
        protocol.parse_message(json.dumps(data))


def test_the_boundary_fractions_are_accepted():
    assert from_worker(protocol.ProgressMessage, stage="sft", fraction=0).fraction == 0.0
    assert from_worker(protocol.ProgressMessage, stage="sft", fraction=1).fraction == 1.0


def test_inbound_and_outbound_parsers_refuse_the_wrong_direction():
    lease_text = protocol.message_json(a_lease())
    started_text = protocol.message_json(from_worker(protocol.StartedMessage))

    assert isinstance(protocol.parse_outbound(lease_text), protocol.LeaseMessage)
    assert isinstance(protocol.parse_inbound(started_text), protocol.StartedMessage)
    with pytest.raises(canonical.ContractError):
        protocol.parse_inbound(lease_text)
    with pytest.raises(canonical.ContractError):
        protocol.parse_outbound(started_text)


def test_a_reconciled_success_must_carry_its_result_and_digest():
    with pytest.raises(ValidationError):
        from_worker(protocol.ReconcileResultMessage, outcome="succeeded")

    message = from_worker(protocol.ReconcileResultMessage, outcome="succeeded",
                          result={"adapter": "a"})
    assert message.result_sha256 == canonical.digest({"adapter": "a"})


def test_only_a_reconciled_success_may_carry_a_result():
    with pytest.raises(ValidationError):
        from_worker(protocol.ReconcileResultMessage, outcome="absent", result={"adapter": "a"})

    with pytest.raises(ValidationError):
        from_worker(protocol.ReconcileResultMessage, outcome="absent",
                    result_sha256=canonical.digest({}))


def test_a_reconciled_failure_must_name_its_reason():
    with pytest.raises(ValidationError):
        from_worker(protocol.ReconcileResultMessage, outcome="failed")

    assert from_worker(protocol.ReconcileResultMessage, outcome="failed",
                       reason_code="crashed").outcome == "failed"


def test_oversized_and_malformed_wire_text_is_refused():
    with pytest.raises(canonical.ContractError):
        protocol.parse_message(b"{" + b" " * (protocol.MAX_MESSAGE_BYTES + 1))

    for text in (b"{", b"[]", b"null", b"\xff"):
        with pytest.raises(canonical.ContractError):
            protocol.parse_message(text)


def test_an_oversized_message_is_refused_on_the_way_out_too():
    with pytest.raises(canonical.ContractError):
        protocol.message_json(from_worker(protocol.SucceededMessage,
                                          result={"adapter": "x" * protocol.MAX_MESSAGE_BYTES}))


def test_the_protocol_and_version_are_fixed():
    data = json.loads(protocol.message_json(from_worker(protocol.StartedMessage)))

    for changed in ({**data, "protocol": "other"}, {**data, "version": 2}):
        with pytest.raises(canonical.ContractError):
            protocol.parse_message(json.dumps(changed))


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", "bad job"),
        ("message_id", "x"),
        ("sender", "Worker A"),
        ("lease_token", "short"),
    ],
)
def test_a_value_with_the_wrong_shape_is_refused(field, value):
    data = json.loads(protocol.message_json(from_worker(protocol.StartedMessage)))
    data[field] = value

    with pytest.raises(canonical.ContractError):
        protocol.parse_message(json.dumps(data))


def test_retryable_is_a_strict_boolean():
    with pytest.raises(ValidationError):
        from_worker(protocol.FailedMessage, reason_code="cuda_oom", retryable="yes")


@pytest.mark.parametrize("attempt", [0, -1, "1", True, 1.5])
def test_attempt_is_a_positive_integer(attempt):
    with pytest.raises(ValidationError):
        from_worker(protocol.StartedMessage, attempt=attempt)


def test_create_fills_a_unique_id_and_a_normalized_time():
    first = from_worker(protocol.StartedMessage)
    second = from_worker(protocol.StartedMessage)

    assert first.message_id != second.message_id
    assert canonical.parse_iso(first.sent_at).tzinfo is not None
    assert first.protocol == protocol.PROTOCOL
    assert first.version == protocol.VERSION
