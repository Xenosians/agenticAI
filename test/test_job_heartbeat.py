import asyncio

from types import (
    SimpleNamespace,
)

from api.app import (
    JobExecuteRequest,
    job_heartbeat_worker,
)


class FakeSettings:
    job_heartbeat_interval_seconds = (
        0.001
    )

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
    ) -> None:
        self.calls = []

        self.responses = [
            FakeResponse(
                200,
                {
                    "job_id":
                        "job-123",

                    "attempt":
                        3,

                    "status":
                        "processing",

                    "lease_expires_at":
                        "2026-09-13T22:30:00Z",
                },
            ),

            FakeResponse(
                409,
                {
                    "error":
                        "stale_attempt",

                    "current_attempt":
                        4,

                    "received_attempt":
                        3,
                },
            ),
        ]

    async def post(
        self,
        url,
        *,
        headers,
        json,
    ):
        self.calls.append(
            {
                "url":
                    url,

                "headers":
                    headers,

                "json":
                    json,
            }
        )

        return (
            self.responses.pop(
                0
            )
        )


def test_heartbeat_renews_exact_attempt_and_stops_when_stale():
    async def scenario():
        phoenix_client = (
            FakePhoenixClient()
        )

        runtime = (
            SimpleNamespace(
                settings=(
                    FakeSettings()
                ),
                phoenix_client=(
                    phoenix_client
                ),
            )
        )

        payload = (
            JobExecuteRequest(
                job_id=(
                    "job-123"
                ),
                attempt=3,
                user_id=(
                    "jdoe"
                ),
                message=(
                    "Long running AI request"
                ),
            )
        )

        await job_heartbeat_worker(
            runtime,
            payload,
        )

        assert (
            len(
                phoenix_client.calls
            )
            == 2
        )

        first_call = (
            phoenix_client.calls[
                0
            ]
        )

        assert (
            first_call[
                "url"
            ]
            == (
                "http://phoenix.test"
                "/api/internal/v1/jobs/"
                "job-123/heartbeat"
            )
        )

        assert (
            first_call[
                "headers"
            ][
                "x-internal-token"
            ]
            == "test-token"
        )

        assert (
            first_call[
                "json"
            ]
            == {
                "attempt":
                    3
            }
        )

        second_call = (
            phoenix_client.calls[
                1
            ]
        )

        assert (
            second_call[
                "json"
            ][
                "attempt"
            ]
            == 3
        )

    asyncio.run(
        scenario()
    )