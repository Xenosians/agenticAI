from contextlib import asynccontextmanager
import asyncio
import uuid
from dataclasses import asdict

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
)
from pydantic import BaseModel, Field

from agent.approvals import approve_approval
from agent.mcp_client import mcp_runtime
from subagents.hub import build_hub


# ============================================================
# Request models
# ============================================================


class AgentRunRequest(BaseModel):
    user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class JobExecuteRequest(BaseModel):
    job_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


# ============================================================
# Background durable-job execution
# ============================================================


async def execute_job(
    app: FastAPI,
    payload: JobExecuteRequest,
) -> None:
    """
    Execute one job after FastAPI has already returned 202.

    Completion callback to Phoenix will be added
    in the next implementation step.
    """

    job_id = payload.job_id

    try:
        print(
            f"[JOB] Starting "
            f"job_id={job_id}"
        )

        hub = app.state.hub

        result = await hub.run(
            payload.message
        )

        result_dict = asdict(result)

        print(
            f"[JOB] Completed "
            f"job_id={job_id} "
            f"status={result_dict.get('status')}"
        )

        print(
            f"[JOB] Result "
            f"job_id={job_id}: "
            f"{result_dict}"
        )

    except asyncio.CancelledError:
        print(
            f"[JOB] Cancelled "
            f"job_id={job_id}"
        )

        raise

    except Exception as exc:
        print(
            f"[JOB] Failed "
            f"job_id={job_id}: "
            f"{exc!r}"
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
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.hub = None

    # Process-local duplicate protection.
    #
    # This is NOT durable storage.
    # Phoenix + SurrealDB remain the durable job owner.
    app.state.active_jobs = {}

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