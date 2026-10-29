import asyncio

from types import (
    SimpleNamespace,
)

from agent.completion_outbox import (
    OutboxEntry,
)

from api.app import (
    deliver_outbox_entry,
)


class FakeSettings:
    def require_phoenix_base_url(
        self,
    ) -> str:
        return (
            "http://phoenix.test"
        )

    def require_internal_job_token(
        self,
    ) -> str:
        return (
            "test-token"
        )


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        body,
    ) -> None:
        self.status_code = (
            status_code
        )

        self.body = (
            body
        )

        self.text = (
            str(
                body
            )
        )

    def json(
        self,
    ):
        return (
            self.body
        )


class FakePhoenixClient:
    def __init__(
        self,
        response: FakeResponse,
    ) -> None:
        self.response = (
            response
        )

    async def post(
        self,
        url,
        *,
        headers,
        json,
    ):
        return (
            self.response
        )


class FakeOutbox:
    def __init__(
        self,
    ) -> None:
        self.deleted = []
        self.delivery_attempts = []

    def delete(
        self,
        job_id,
        attempt,
    ) -> None:
        self.deleted.append(
            (
                job_id,
                attempt,
            )
        )

    def mark_delivery_attempt(
        self,
        job_id,
        attempt,
        error,
    ) -> None:
        self.delivery_attempts.append(
            (
                job_id,
                attempt,
                error,
            )
        )


def build_entry(
) -> OutboxEntry:
    return OutboxEntry(
        job_id=(
            "job-123"
        ),
        attempt=3,
        payload={
            "attempt":
                3,

            "status":
                "completed",

            "selected_agent":
                None,

            "proposed_tool":
                None,

            "result": {
                "answer":
                    "done",
            },
        },
        delivery_attempts=0,
        last_error=None,
        created_at=(
            "2026-09-13T00:00:00"
        ),
    )


def build_runtime(
    response: FakeResponse,
):
    outbox = (
        FakeOutbox()
    )

    runtime = (
        SimpleNamespace(
            settings=(
                FakeSettings()
            ),
            phoenix_client=(
                FakePhoenixClient(
                    response
                )
            ),
            completion_outbox=(
                outbox
            ),
        )
    )

    return (
        runtime,
        outbox,
    )


def test_valid_applied_ack_deletes_outbox_entry():
    async def scenario():
        response = (
            FakeResponse(
                200,
                {
                    "job_id":
                        "job-123",

                    "status":
                        "completed",

                    "acknowledgement":
                        "applied",
                },
            )
        )

        runtime, outbox = (
            build_runtime(
                response
            )
        )

        result = (
            await deliver_outbox_entry(
                runtime,
                build_entry(),
            )
        )

        assert result is True

        assert (
            outbox.deleted
            == [
                (
                    "job-123",
                    3,
                )
            ]
        )

        assert (
            outbox.delivery_attempts
            == []
        )

    asyncio.run(
        scenario()
    )


def test_invalid_success_ack_keeps_outbox_entry():
    async def scenario():
        response = (
            FakeResponse(
                200,
                {
                    "job_id":
                        "wrong-job",

                    "status":
                        "completed",

                    "acknowledgement":
                        "applied",
                },
            )
        )

        runtime, outbox = (
            build_runtime(
                response
            )
        )

        result = (
            await deliver_outbox_entry(
                runtime,
                build_entry(),
            )
        )

        assert result is False

        assert (
            outbox.deleted
            == []
        )

        assert (
            len(
                outbox.delivery_attempts
            )
            == 1
        )

    asyncio.run(
        scenario()
    )


def test_duplicate_ack_may_report_existing_terminal_status():
    async def scenario():
        response = (
            FakeResponse(
                200,
                {
                    "job_id":
                        "job-123",

                    "status":
                        "failed",

                    "acknowledgement":
                        "duplicate",
                },
            )
        )

        runtime, outbox = (
            build_runtime(
                response
            )
        )

        result = (
            await deliver_outbox_entry(
                runtime,
                build_entry(),
            )
        )

        assert result is True

        assert (
            outbox.deleted
            == [
                (
                    "job-123",
                    3,
                )
            ]
        )

    asyncio.run(
        scenario()
    )