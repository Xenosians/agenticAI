from contextlib import asynccontextmanager
import asyncio
import os
import uuid
from dataclasses import asdict
from typing import Any

import httpx
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
)
from pydantic import BaseModel, Field

from agent.approvals import approve_approval
from agent.completion_outbox import (
    CompletionOutbox,
    OutboxEntry,
)
from agent.mcp_client import mcp_runtime
from subagents.hub import build_hub


# ============================================================
# Configuration
# ============================================================


def phoenix_base_url() -> str:
    return os.getenv(
        "PHOENIX_BASE_URL",
        "http://127.0.0.1:4000",
    ).rstrip("/")


def internal_job_token() -> str:
    return os.getenv(
        "ITSM_INTERNAL_JOB_TOKEN",
        "itsm-dev-internal-2026",
    )


def completion_outbox_path() -> str:
    return os.getenv(
        "COMPLETION_OUTBOX_PATH",
        ".runtime/completion_outbox.sqlite3",
    )


OUTBOX_RETRY_SECONDS = 5.0
OUTBOX_BATCH_SIZE = 100


# ============================================================
# Request models
# ============================================================


class AgentRunRequest(BaseModel):
    user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class JobExecuteRequest(BaseModel):
    job_id: str = Field(min_length=1)

    attempt: int = Field(
        ge=1,
    )

    user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


# ============================================================
# Completion payload helpers
# ============================================================


def completion_status(
    result_dict: dict[str, Any],
) -> str:
    hub_status = result_dict.get(
        "status"
    )

    if hub_status == "approval_required":
        return "waiting_approval"

    if hub_status in {
        "success",
        "no_route",
    }:
        return "completed"

    return "failed"


def selected_agent(
    result_dict: dict[str, Any],
) -> str | None:
    routes = result_dict.get(
        "routes"
    )

    if (
        isinstance(routes, list)
        and routes
        and isinstance(routes[0], str)
    ):
        return routes[0]

    return None


def proposed_tool(
    result_dict: dict[str, Any],
) -> dict[str, Any] | None:
    results = result_dict.get(
        "results"
    )

    if not isinstance(
        results,
        list,
    ):
        return None

    for result in results:
        if not isinstance(
            result,
            dict,
        ):
            continue

        tool_name = result.get(
            "proposed_tool"
        )

        if not tool_name:
            continue

        return {
            "tool":
                tool_name,
            "arguments":
                result.get(
                    "proposed_arguments"
                )
                or {},
            "approval_id":
                result.get(
                    "approval_id"
                ),
        }

    return None


def completion_payload(
    payload: JobExecuteRequest,
    result_dict: dict[str, Any],
) -> dict[str, Any]:
    final_status = completion_status(
        result_dict
    )

    callback: dict[str, Any] = {
        "attempt":
            payload.attempt,
        "status":
            final_status,
        "selected_agent":
            selected_agent(
                result_dict
            ),
        "proposed_tool":
            proposed_tool(
                result_dict
            ),
    }

    if final_status in {
        "completed",
        "waiting_approval",
    }:
        callback["result"] = (
            result_dict
        )

    else:
        callback["error"] = (
            result_dict.get("error")
            or (
                "AI job finished with "
                f"status={result_dict.get('status')}"
            )
        )

    return callback


def failure_payload(
    payload: JobExecuteRequest,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "attempt":
            payload.attempt,
        "status":
            "failed",
        "error":
            repr(exc),
        "selected_agent":
            None,
        "proposed_tool":
            None,
    }


# ============================================================
# Durable completion persistence
# ============================================================


def persist_completion(
    app: FastAPI,
    payload: JobExecuteRequest,
    callback_payload: dict[str, Any],
) -> None:
    """
    Persist completion BEFORE attempting network delivery.

    Once this returns successfully, Python may crash and the
    completion can still be replayed after restart.
    """

    outbox: CompletionOutbox = (
        app.state.completion_outbox
    )

    outbox.put(
        payload.job_id,
        payload.attempt,
        callback_payload,
    )

    print(
        "[OUTBOX] Persisted "
        f"job_id={payload.job_id} "
        f"attempt={payload.attempt}"
    )

    app.state.outbox_wakeup.set()


# ============================================================
# Phoenix completion delivery
# ============================================================


def response_body(
    response: httpx.Response,
) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


async def deliver_outbox_entry(
    app: FastAPI,
    entry: OutboxEntry,
) -> bool:
    """
    Deliver one persisted completion.

    Returns True when the entry is resolved and removed from
    the outbox.

    Returns False when it must remain durable for retry.
    """

    outbox: CompletionOutbox = (
        app.state.completion_outbox
    )

    client: httpx.AsyncClient = (
        app.state.phoenix_client
    )

    url = (
        f"{phoenix_base_url()}"
        f"/api/internal/v1/jobs/"
        f"{entry.job_id}"
        f"/completion"
    )

    headers = {
        "x-internal-token":
            internal_job_token(),
    }

    try:
        response = await client.post(
            url,
            headers=headers,
            json=entry.payload,
        )

    except Exception as exc:
        error = repr(exc)

        outbox.mark_delivery_attempt(
            entry.job_id,
            entry.attempt,
            error,
        )

        print(
            "[OUTBOX] Delivery failed "
            f"job_id={entry.job_id} "
            f"attempt={entry.attempt} "
            f"error={error}"
        )

        return False

    body = response_body(
        response
    )

    # --------------------------------------------------------
    # Phoenix ACK
    # --------------------------------------------------------

    if 200 <= response.status_code < 300:
        outbox.delete(
            entry.job_id,
            entry.attempt,
        )

        print(
            "[OUTBOX] Delivery ACK "
            f"job_id={entry.job_id} "
            f"attempt={entry.attempt} "
            f"response={body}"
        )

        return True

    # --------------------------------------------------------
    # Stale execution attempt
    #
    # Phoenix has already moved to a newer attempt.
    # This old completion must never overwrite it.
    # It is permanently obsolete and can be removed.
    # --------------------------------------------------------

    if (
        response.status_code == 409
        and isinstance(body, dict)
        and body.get("error")
        == "stale_attempt"
    ):
        outbox.delete(
            entry.job_id,
            entry.attempt,
        )

        print(
            "[OUTBOX] Discarded stale completion "
            f"job_id={entry.job_id} "
            f"attempt={entry.attempt} "
            f"response={body}"
        )

        return True

    # --------------------------------------------------------
    # Any other rejection remains durable.
    #
    # Examples:
    #   Phoenix temporarily unavailable
    #   incorrect internal token
    #   unexpected state conflict
    #
    # We do NOT throw the completion away.
    # --------------------------------------------------------

    error = (
        f"HTTP {response.status_code}: "
        f"{body!r}"
    )

    outbox.mark_delivery_attempt(
        entry.job_id,
        entry.attempt,
        error,
    )

    print(
        "[OUTBOX] Delivery rejected "
        f"job_id={entry.job_id} "
        f"attempt={entry.attempt} "
        f"status={response.status_code} "
        f"response={body}"
    )

    return False


# ============================================================
# Persistent outbox delivery worker
# ============================================================


async def outbox_delivery_worker(
    app: FastAPI,
) -> None:
    """
    Replay persisted completions until Phoenix ACKs them.

    This worker is intentionally independent from AI job
    execution. It also replays entries left behind by a
    previous Python process.
    """

    outbox: CompletionOutbox = (
        app.state.completion_outbox
    )

    print(
        "[OUTBOX] Delivery worker started "
        f"pending={outbox.count()}"
    )

    try:
        while True:
            entries = outbox.list_pending(
                limit=OUTBOX_BATCH_SIZE
            )

            for entry in entries:
                await deliver_outbox_entry(
                    app,
                    entry,
                )

            try:
                await asyncio.wait_for(
                    app.state.outbox_wakeup.wait(),
                    timeout=OUTBOX_RETRY_SECONDS,
                )

            except asyncio.TimeoutError:
                pass

            finally:
                app.state.outbox_wakeup.clear()

    except asyncio.CancelledError:
        print(
            "[OUTBOX] Delivery worker stopped"
        )

        raise


# ============================================================
# Background durable-job execution
# ============================================================


async def execute_job(
    app: FastAPI,
    payload: JobExecuteRequest,
) -> None:
    job_id = payload.job_id

    try:
        print(
            "[JOB] Starting "
            f"job_id={job_id} "
            f"attempt={payload.attempt}"
        )

        hub = app.state.hub

        result = await hub.run(
            payload.message
        )

        result_dict = asdict(
            result
        )

        print(
            "[JOB] AI finished "
            f"job_id={job_id} "
            f"attempt={payload.attempt} "
            f"status="
            f"{result_dict.get('status')}"
        )

        callback_payload = (
            completion_payload(
                payload,
                result_dict,
            )
        )

        # Critical durability boundary:
        #
        # result is persisted locally BEFORE any callback
        # attempt is made.
        persist_completion(
            app,
            payload,
            callback_payload,
        )

    except asyncio.CancelledError:
        print(
            "[JOB] Cancelled "
            f"job_id={job_id} "
            f"attempt={payload.attempt}"
        )

        raise

    except Exception as exc:
        print(
            "[JOB] Failed "
            f"job_id={job_id} "
            f"attempt={payload.attempt} "
            f"error={exc!r}"
        )

        try:
            callback_payload = (
                failure_payload(
                    payload,
                    exc,
                )
            )

            persist_completion(
                app,
                payload,
                callback_payload,
            )

        except Exception as outbox_exc:
            print(
                "[OUTBOX] CRITICAL: "
                "failed to persist failure completion "
                f"job_id={job_id} "
                f"attempt={payload.attempt} "
                f"error={outbox_exc!r}"
            )

    finally:
        app.state.active_jobs.pop(
            job_id,
            None,
        )


def schedule_job(
    app: FastAPI,
    payload: JobExecuteRequest,
) -> asyncio.Task:
    task = asyncio.create_task(
        execute_job(
            app,
            payload,
        )
    )

    app.state.active_jobs[
        payload.job_id
    ] = task

    return task


# ============================================================
# Application lifecycle
# ============================================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    app.state.ready = False
    app.state.hub = None
    app.state.active_jobs = {}

    # --------------------------------------------------------
    # Durable completion outbox
    # --------------------------------------------------------

    outbox = CompletionOutbox(
        completion_outbox_path()
    )

    outbox.initialize()

    app.state.completion_outbox = (
        outbox
    )

    app.state.outbox_wakeup = (
        asyncio.Event()
    )

    app.state.phoenix_client = (
        httpx.AsyncClient(
            timeout=10.0,
        )
    )

    app.state.outbox_task = (
        asyncio.create_task(
            outbox_delivery_worker(
                app
            )
        )
    )

    # --------------------------------------------------------
    # MCP + AI runtime
    # --------------------------------------------------------

    await mcp_runtime.start()

    try:
        print(
            "[API] Loading AI runtime..."
        )

        app.state.hub = build_hub()

        app.state.ready = True

        print(
            "[API] AI runtime ready."
        )

        yield

    finally:
        app.state.ready = False

        # ----------------------------------------------------
        # Stop active AI executions.
        #
        # Anything already persisted in the outbox survives.
        # Anything still executing is recovered later through
        # Phoenix lease recovery.
        # ----------------------------------------------------

        active_tasks = list(
            app.state.active_jobs.values()
        )

        for task in active_tasks:
            task.cancel()

        if active_tasks:
            await asyncio.gather(
                *active_tasks,
                return_exceptions=True,
            )

        app.state.active_jobs.clear()

        # ----------------------------------------------------
        # Stop outbox worker
        # ----------------------------------------------------

        outbox_task = (
            app.state.outbox_task
        )

        outbox_task.cancel()

        await asyncio.gather(
            outbox_task,
            return_exceptions=True,
        )

        await app.state.phoenix_client.aclose()

        app.state.hub = None

        await mcp_runtime.stop()

        print(
            "[API] AI runtime stopped."
        )


# ============================================================
# FastAPI application
# ============================================================


app = FastAPI(
    title="ITSM AI Service",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================
# Health endpoints
# ============================================================


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "itsm-ai",
    }


@app.get("/ready")
async def ready(
    request: Request,
) -> dict[str, str]:
    if not request.app.state.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI runtime is not ready."
            ),
        )

    return {
        "status": "ready",
        "service": "itsm-ai",
    }


# ============================================================
# Transitional synchronous endpoint
# ============================================================


@app.post("/v1/agent/run")
async def run_agent(
    payload: AgentRunRequest,
    request: Request,
) -> dict:
    if not request.app.state.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI runtime is not ready."
            ),
        )

    hub = request.app.state.hub

    result = await hub.run(
        payload.message
    )

    return {
        "request_id":
            uuid.uuid4().hex,
        "user_id":
            payload.user_id,
        **asdict(result),
    }


# ============================================================
# Durable async execution endpoint
# Handshake #1
# ============================================================


@app.post(
    "/v1/jobs/execute",
    status_code=status.HTTP_202_ACCEPTED,
)
async def execute_durable_job(
    payload: JobExecuteRequest,
    request: Request,
) -> dict:
    if not request.app.state.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI runtime is not ready."
            ),
        )

    active_jobs = (
        request.app.state.active_jobs
    )

    existing_task = active_jobs.get(
        payload.job_id
    )

    if (
        existing_task is not None
        and not existing_task.done()
    ):
        return {
            "job_id":
                payload.job_id,
            "attempt":
                payload.attempt,
            "status":
                "accepted",
            "duplicate":
                True,
        }

    schedule_job(
        request.app,
        payload,
    )

    return {
        "job_id":
            payload.job_id,
        "attempt":
            payload.attempt,
        "status":
            "accepted",
        "duplicate":
            False,
    }


# ============================================================
# Approval execution
# ============================================================


@app.post(
    "/v1/approvals/{approval_id}/approve"
)
async def approve(
    approval_id: str,
    request: Request,
) -> dict:
    if not request.app.state.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI runtime is not ready."
            ),
        )

    approval_result = (
        await approve_approval(
            approval_id
        )
    )

    if not approval_result.get("ok"):
        error = (
            approval_result.get("error")
            or approval_result
            .get(
                "result",
                {},
            )
            .get("error")
            or (
                "Approval execution "
                "failed."
            )
        )

        raise HTTPException(
            status_code=400,
            detail=error,
        )

    return {
        "approval_id":
            approval_id,
        "status":
            "executed",
        "result":
            approval_result["result"],
    }