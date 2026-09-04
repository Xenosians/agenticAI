from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import httpx
from fastapi import (
    FastAPI,
    Request,
    Response,
)
from fastapi.responses import JSONResponse


REAL_AI_URL = os.getenv(
    "REAL_AI_URL",
    "http://127.0.0.1:8003",
).rstrip("/")

REAL_PHOENIX_URL = os.getenv(
    "REAL_PHOENIX_URL",
    "http://127.0.0.1:4000",
).rstrip("/")

FAIL_COMPLETIONS = int(
    os.getenv(
        "FAIL_COMPLETIONS",
        "3",
    )
)


app = FastAPI(
    title="ITSM Handshake Fault Proxy",
)


# ============================================================
# Fault state
# ============================================================


completion_failures: dict[
    str,
    int,
] = defaultdict(int)

completion_forwards: dict[
    str,
    int,
] = defaultdict(int)

execution_requests: dict[
    str,
    int,
] = defaultdict(int)

released_jobs: set[str] = set()

events: dict[
    str,
    list[dict[str, Any]],
] = defaultdict(list)


# ============================================================
# Helpers
# ============================================================


def record_event(
    job_id: str,
    event: str,
    **details: Any,
) -> None:
    entry = {
        "event": event,
        **details,
    }

    events[job_id].append(
        entry
    )

    print(
        "[FAULT] "
        f"{event} "
        f"job_id={job_id} "
        f"{details}"
    )


async def forward(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: Any = None,
) -> Response:
    async with httpx.AsyncClient(
        timeout=30.0,
    ) as client:
        response = await client.request(
            method,
            url,
            headers=headers,
            json=json_body,
        )

    response_headers = {}

    content_type = response.headers.get(
        "content-type"
    )

    if content_type:
        response_headers[
            "content-type"
        ] = content_type

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=response_headers,
    )


def forwarded_headers(
    request: Request,
) -> dict[str, str]:
    result: dict[str, str] = {}

    internal_token = (
        request.headers.get(
            "x-internal-token"
        )
    )

    if internal_token:
        result[
            "x-internal-token"
        ] = internal_token

    return result


# ============================================================
# Test-control API
# ============================================================


@app.get("/__faults")
async def fault_state() -> dict:
    return {
        "fail_completions":
            FAIL_COMPLETIONS,
        "real_ai_url":
            REAL_AI_URL,
        "real_phoenix_url":
            REAL_PHOENIX_URL,
    }


@app.get(
    "/__faults/jobs/{job_id}"
)
async def job_fault_state(
    job_id: str,
) -> dict:
    return {
        "job_id":
            job_id,
        "execution_requests":
            execution_requests[
                job_id
            ],
        "completion_failures":
            completion_failures[
                job_id
            ],
        "completion_forwards":
            completion_forwards[
                job_id
            ],
        "released":
            job_id in released_jobs,
        "events":
            events[
                job_id
            ],
    }


@app.post(
    "/__faults/release/{job_id}"
)
async def release_job(
    job_id: str,
) -> dict:
    released_jobs.add(
        job_id
    )

    record_event(
        job_id,
        "manual_release",
    )

    return {
        "job_id":
            job_id,
        "released":
            True,
    }


@app.post(
    "/__faults/reset/{job_id}"
)
async def reset_job(
    job_id: str,
) -> dict:
    completion_failures.pop(
        job_id,
        None,
    )

    completion_forwards.pop(
        job_id,
        None,
    )

    execution_requests.pop(
        job_id,
        None,
    )

    events.pop(
        job_id,
        None,
    )

    released_jobs.discard(
        job_id
    )

    return {
        "job_id":
            job_id,
        "reset":
            True,
    }


# ============================================================
# Phoenix -> Python
# ============================================================


@app.get("/ready")
async def ai_ready() -> Response:
    return await forward(
        "GET",
        f"{REAL_AI_URL}/ready",
    )


@app.get("/health")
async def ai_health() -> Response:
    return await forward(
        "GET",
        f"{REAL_AI_URL}/health",
    )


@app.post(
    "/v1/jobs/execute"
)
async def execute_job(
    request: Request,
) -> Response:
    payload = await request.json()

    job_id = payload.get(
        "job_id",
        "unknown",
    )

    attempt = payload.get(
        "attempt"
    )

    execution_requests[
        job_id
    ] += 1

    record_event(
        job_id,
        "phoenix_to_python",
        attempt=attempt,
    )

    response = await forward(
        "POST",
        (
            f"{REAL_AI_URL}"
            "/v1/jobs/execute"
        ),
        json_body=payload,
    )

    record_event(
        job_id,
        "python_execution_ack",
        http_status=
            response.status_code,
    )

    return response


# ============================================================
# Python -> Phoenix
# ============================================================


@app.post(
    "/api/internal/v1/jobs/"
    "{job_id}/completion"
)
async def completion(
    job_id: str,
    request: Request,
) -> Response:
    payload = await request.json()

    attempt = payload.get(
        "attempt"
    )

    failures_so_far = (
        completion_failures[
            job_id
        ]
    )

    should_fail = (
        job_id not in released_jobs
        and (
            FAIL_COMPLETIONS < 0
            or failures_so_far
            < FAIL_COMPLETIONS
        )
    )

    if should_fail:
        completion_failures[
            job_id
        ] += 1

        failure_number = (
            completion_failures[
                job_id
            ]
        )

        record_event(
            job_id,
            "completion_blocked",
            attempt=attempt,
            failure_number=
                failure_number,
        )

        return JSONResponse(
            status_code=503,
            content={
                "error":
                    "fault_injection",
                "job_id":
                    job_id,
                "failure":
                    failure_number,
            },
        )

    completion_forwards[
        job_id
    ] += 1

    record_event(
        job_id,
        "completion_forwarded",
        attempt=attempt,
    )

    response = await forward(
        "POST",
        (
            f"{REAL_PHOENIX_URL}"
            f"/api/internal/v1/jobs/"
            f"{job_id}"
            "/completion"
        ),
        headers=forwarded_headers(
            request
        ),
        json_body=payload,
    )

    record_event(
        job_id,
        "phoenix_completion_ack",
        http_status=
            response.status_code,
    )

    return response


# ============================================================
# Transitional endpoints
# ============================================================


@app.post(
    "/v1/agent/run"
)
async def agent_run(
    request: Request,
) -> Response:
    payload = await request.json()

    return await forward(
        "POST",
        f"{REAL_AI_URL}/v1/agent/run",
        json_body=payload,
    )


@app.post(
    "/v1/approvals/"
    "{approval_id}/approve"
)
async def approval(
    approval_id: str,
) -> Response:
    return await forward(
        "POST",
        (
            f"{REAL_AI_URL}"
            f"/v1/approvals/"
            f"{approval_id}"
            "/approve"
        ),
    )