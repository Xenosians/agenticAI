from contextlib import (
    asynccontextmanager,
)

import asyncio
import uuid

from dataclasses import (
    asdict,
)

from typing import (
    Any,
)

import httpx

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
)

from pydantic import (
    BaseModel,
    Field,
)

from agent.approvals import (
    ApprovalManager,
)

from agent.approval_store import (
    ApprovalStore,
)

from agent.completion_outbox import (
    CompletionOutbox,
    OutboxEntry,
)

from agent.mcp_client import (
    MCPRuntime,
)

from api.runtime import (
    ApplicationRuntime,
)

from config import (
    Settings,
)

from subagents.core.tool_gateway import (
    ToolGateway,
)

from subagents.hub import (
    build_hub,
)

from subagents.llm.inference import (
    InferenceCoordinator,
)

from subagents.llm.model_manager import (
    ModelManager,
)

from subagents.llm.scheduler import (
    GpuScheduler,
)


# ============================================================
# Request models
# ============================================================


class AgentRunRequest(
    BaseModel
):
    user_id: str = Field(
        min_length=1
    )

    message: str = Field(
        min_length=1
    )


class JobExecuteRequest(
    BaseModel
):
    job_id: str = Field(
        min_length=1
    )

    attempt: int = Field(
        ge=1
    )

    user_id: str = Field(
        min_length=1
    )

    message: str = Field(
        min_length=1
    )


# ============================================================
# Runtime access
# ============================================================


def application_runtime(
    app: FastAPI,
) -> ApplicationRuntime:
    """
    Return the explicit application-owned runtime.

    Endpoints should not discover individual process resources
    through module globals.
    """

    runtime = getattr(
        app.state,
        "runtime",
        None,
    )

    if not isinstance(
        runtime,
        ApplicationRuntime,
    ):
        raise RuntimeError(
            "AI application runtime "
            "has not been initialized."
        )

    return runtime


def require_ready_runtime(
    request: Request,
) -> ApplicationRuntime:
    runtime = (
        application_runtime(
            request.app
        )
    )

    if not runtime.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI runtime is not ready."
            ),
        )

    return runtime


# ============================================================
# Completion payload helpers
# ============================================================


def completion_status(
    result_dict: dict[
        str,
        Any,
    ],
) -> str:
    hub_status = (
        result_dict.get(
            "status"
        )
    )

    if (
        hub_status
        == "approval_required"
    ):
        return (
            "waiting_approval"
        )

    if hub_status in {
        "success",
        "no_route",
    }:
        return (
            "completed"
        )

    return (
        "failed"
    )


def selected_agent(
    result_dict: dict[
        str,
        Any,
    ],
) -> str | None:
    routes = (
        result_dict.get(
            "routes"
        )
    )

    if (
        isinstance(
            routes,
            list,
        )
        and routes
        and isinstance(
            routes[0],
            str,
        )
    ):
        return (
            routes[0]
        )

    return None


def proposed_tool(
    result_dict: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
] | None:
    results = (
        result_dict.get(
            "results"
        )
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

        tool_name = (
            result.get(
                "proposed_tool"
            )
        )

        if not tool_name:
            continue

        return {
            "tool":
                tool_name,

            "arguments": (
                result.get(
                    "proposed_arguments"
                )
                or {}
            ),

            "approval_id":
                result.get(
                    "approval_id"
                ),
        }

    return None


def completion_payload(
    payload: JobExecuteRequest,
    result_dict: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    final_status = (
        completion_status(
            result_dict
        )
    )

    callback: dict[
        str,
        Any,
    ] = {
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
        callback[
            "result"
        ] = result_dict

    else:
        callback[
            "error"
        ] = (
            result_dict.get(
                "error"
            )
            or (
                "AI job finished with "
                "status="
                f"{result_dict.get('status')}"
            )
        )

    return callback


def failure_payload(
    payload: JobExecuteRequest,
    exc: Exception,
) -> dict[
    str,
    Any,
]:
    return {
        "attempt":
            payload.attempt,

        "status":
            "failed",

        "error":
            repr(
                exc
            ),

        "selected_agent":
            None,

        "proposed_tool":
            None,
    }


# ============================================================
# Durable completion persistence
# ============================================================


def persist_completion(
    runtime: ApplicationRuntime,
    payload: JobExecuteRequest,
    callback_payload: dict[
        str,
        Any,
    ],
) -> None:
    """
    Persist completion BEFORE network delivery.

    Once this returns successfully, process failure cannot lose
    the completion callback.
    """

    runtime.completion_outbox.put(
        payload.job_id,
        payload.attempt,
        callback_payload,
    )

    print(
        "[OUTBOX] Persisted "
        f"job_id={payload.job_id} "
        f"attempt={payload.attempt}"
    )

    runtime.outbox_wakeup.set()


# ============================================================
# Phoenix completion delivery
# ============================================================


def response_body(
    response: httpx.Response,
) -> Any:
    try:
        return (
            response.json()
        )

    except ValueError:
        return (
            response.text
        )


async def deliver_outbox_entry(
    runtime: ApplicationRuntime,
    entry: OutboxEntry,
) -> bool:
    """
    Deliver one persisted completion.

    True:
        completion resolved and removed.

    False:
        completion remains durable for retry.
    """

    settings = (
        runtime.settings
    )

    url = (
        f"{settings.require_phoenix_base_url()}"
        f"/api/internal/v1/jobs/"
        f"{entry.job_id}"
        "/completion"
    )

    headers = {
        "x-internal-token":
            settings
            .require_internal_job_token(),
    }

    try:
        response = (
            await
            runtime
            .phoenix_client
            .post(
                url,
                headers=(
                    headers
                ),
                json=(
                    entry.payload
                ),
            )
        )

    except Exception as exc:
        error = (
            repr(
                exc
            )
        )

        runtime.completion_outbox.mark_delivery_attempt(
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

    body = (
        response_body(
            response
        )
    )

    # --------------------------------------------------------
    # Phoenix ACK
    # --------------------------------------------------------

    if (
        200
        <= response.status_code
        < 300
    ):
        runtime.completion_outbox.delete(
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
    # Stale attempt
    #
    # Phoenix has already moved to a newer execution attempt.
    # The old completion is obsolete and must not overwrite it.
    # --------------------------------------------------------

    if (
        response.status_code
        == 409
        and isinstance(
            body,
            dict,
        )
        and body.get(
            "error"
        )
        == "stale_attempt"
    ):
        runtime.completion_outbox.delete(
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
    # Other rejection remains durable.
    # --------------------------------------------------------

    error = (
        f"HTTP {response.status_code}: "
        f"{body!r}"
    )

    runtime.completion_outbox.mark_delivery_attempt(
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
# Durable outbox worker
# ============================================================


async def outbox_delivery_worker(
    runtime: ApplicationRuntime,
) -> None:
    """
    Replay persisted completions until Phoenix ACKs them.

    Delivery remains independent of AI execution.
    """

    print(
        "[OUTBOX] Delivery worker started "
        "pending="
        f"{runtime.completion_outbox.count()}"
    )

    try:
        while True:
            entries = (
                runtime
                .completion_outbox
                .list_pending(
                    limit=(
                        runtime
                        .settings
                        .outbox_batch_size
                    )
                )
            )

            for entry in entries:
                await deliver_outbox_entry(
                    runtime,
                    entry,
                )

            try:
                await asyncio.wait_for(
                    runtime
                    .outbox_wakeup
                    .wait(),
                    timeout=(
                        runtime
                        .settings
                        .outbox_retry_seconds
                    ),
                )

            except asyncio.TimeoutError:
                pass

            finally:
                runtime.outbox_wakeup.clear()

    except asyncio.CancelledError:
        print(
            "[OUTBOX] Delivery worker stopped"
        )

        raise


# ============================================================
# Background durable-job execution
# ============================================================


async def execute_job(
    runtime: ApplicationRuntime,
    payload: JobExecuteRequest,
) -> None:
    job_id = (
        payload.job_id
    )

    try:
        print(
            "[JOB] Starting "
            f"job_id={job_id} "
            f"attempt={payload.attempt}"
        )

        result = (
            await
            runtime.hub.run(
                payload.message
            )
        )

        result_dict = (
            asdict(
                result
            )
        )

        print(
            "[JOB] AI finished "
            f"job_id={job_id} "
            f"attempt={payload.attempt} "
            "status="
            f"{result_dict.get('status')}"
        )

        callback_payload = (
            completion_payload(
                payload,
                result_dict,
            )
        )

        persist_completion(
            runtime,
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
                runtime,
                payload,
                callback_payload,
            )

        except Exception as outbox_exc:
            print(
                "[OUTBOX] CRITICAL: "
                "failed to persist failure "
                "completion "
                f"job_id={job_id} "
                f"attempt={payload.attempt} "
                f"error={outbox_exc!r}"
            )

    finally:
        runtime.active_jobs.pop(
            job_id,
            None,
        )


def schedule_job(
    runtime: ApplicationRuntime,
    payload: JobExecuteRequest,
) -> asyncio.Task:
    task = (
        asyncio.create_task(
            execute_job(
                runtime,
                payload,
            )
        )
    )

    runtime.active_jobs[
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
    """
    Application composition root.

    This is the single lifecycle owner for process-wide runtime
    resources.

    It intentionally creates ordinary objects rather than
    process-global Singletons.
    """

    app.state.runtime = None

    # ========================================================
    # SETTINGS
    # ========================================================

    settings = (
        Settings()
    )

    # Validate Phoenix handshake configuration before any
    # long-lived process resources are started.
    settings.require_phoenix_base_url()
    settings.require_internal_job_token()

    # ========================================================
    # DURABLE COMPLETION OUTBOX
    # ========================================================

    outbox_path = (
        settings.resolve_runtime_path(
            settings
            .completion_outbox_path
        )
    )

    completion_outbox = (
        CompletionOutbox(
            outbox_path
        )
    )

    completion_outbox.initialize()

    outbox_wakeup = (
        asyncio.Event()
    )

    # ========================================================
    # PHOENIX CALLBACK CLIENT
    # ========================================================

    phoenix_client = (
        httpx.AsyncClient(
            timeout=(
                settings
                .phoenix_http_timeout_seconds
            ),
        )
    )

    # ========================================================
    # MCP RUNTIME
    # ========================================================

    mcp = (
        MCPRuntime()
    )

    # ========================================================
    # DURABLE APPROVAL EXECUTION
    # ========================================================

    approval_store_path = (
        settings.resolve_runtime_path(
            settings
            .approval_store_path
        )
    )

    approval_store = (
        ApprovalStore(
            approval_store_path
        )
    )

    approval_store.initialize()

    approvals = (
        ApprovalManager(
            store=(
                approval_store
            ),
            mcp=(
                mcp
            ),
        )
    )

    # ========================================================
    # MODEL / GPU RUNTIME
    # ========================================================

    model_manager = (
        ModelManager(
            settings=(
                settings
            )
        )
    )

    gpu_scheduler = (
        GpuScheduler()
    )

    inference = (
        InferenceCoordinator(
            model_manager=(
                model_manager
            ),
            scheduler=(
                gpu_scheduler
            ),
        )
    )

    # ========================================================
    # TRUSTED TOOL EXECUTION BOUNDARY
    # ========================================================

    tool_gateway = (
        ToolGateway(
            approval_creator=(
                approvals
                .create_approval
            ),
            mcp=(
                mcp
            ),
        )
    )

    runtime: (
        ApplicationRuntime | None
    ) = None

    try:
        # ====================================================
        # START MCP
        # ====================================================

        await mcp.start()

        # ====================================================
        # WARM HUB MODEL
        #
        # Preserve previous readiness semantics: the AI service
        # does not become ready until its main Hub model is
        # available.
        # ====================================================

        print(
            "[API] Loading AI runtime..."
        )

        await inference.warm(
            settings.hub_model_key
        )

        # ====================================================
        # REASONING GRAPH
        # ====================================================

        hub = (
            build_hub(
                settings=(
                    settings
                ),
                model_manager=(
                    model_manager
                ),
                inference=(
                    inference
                ),
                tool_gateway=(
                    tool_gateway
                ),
            )
        )

        # ====================================================
        # APPLICATION RUNTIME CONTAINER
        # ====================================================

        runtime = (
            ApplicationRuntime(
                settings=(
                    settings
                ),
                completion_outbox=(
                    completion_outbox
                ),
                outbox_wakeup=(
                    outbox_wakeup
                ),
                phoenix_client=(
                    phoenix_client
                ),
                mcp=(
                    mcp
                ),
                approvals=(
                    approvals
                ),
                model_manager=(
                    model_manager
                ),
                gpu_scheduler=(
                    gpu_scheduler
                ),
                inference=(
                    inference
                ),
                tool_gateway=(
                    tool_gateway
                ),
                hub=(
                    hub
                ),
            )
        )

        app.state.runtime = (
            runtime
        )

        # ====================================================
        # OUTBOX REPLAY
        # ====================================================

        runtime.outbox_task = (
            asyncio.create_task(
                outbox_delivery_worker(
                    runtime
                )
            )
        )

        runtime.ready = (
            True
        )

        print(
            "[API] AI runtime ready."
        )

        yield

    finally:
        # ====================================================
        # MARK NOT READY FIRST
        # ====================================================

        if runtime is not None:
            runtime.ready = (
                False
            )

        # ====================================================
        # STOP ACTIVE AI EXECUTIONS
        # ====================================================

        if runtime is not None:
            active_tasks = list(
                runtime
                .active_jobs
                .values()
            )

            for task in (
                active_tasks
            ):
                task.cancel()

            if active_tasks:
                await asyncio.gather(
                    *active_tasks,
                    return_exceptions=True,
                )

            runtime.active_jobs.clear()

        # ====================================================
        # STOP OUTBOX WORKER
        # ====================================================

        if (
            runtime is not None
            and runtime.outbox_task
            is not None
        ):
            runtime.outbox_task.cancel()

            await asyncio.gather(
                runtime.outbox_task,
                return_exceptions=True,
            )

            runtime.outbox_task = (
                None
            )

        # ====================================================
        # CLOSE HTTP CLIENT
        # ====================================================

        await phoenix_client.aclose()

        # ====================================================
        # STOP MCP
        # ====================================================

        await mcp.stop()

        app.state.runtime = (
            None
        )

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


@app.get(
    "/health"
)
async def health(
) -> dict[
    str,
    str,
]:
    return {
        "status":
            "ok",

        "service":
            "itsm-ai",
    }


@app.get(
    "/ready"
)
async def ready(
    request: Request,
) -> dict[
    str,
    str,
]:
    runtime = getattr(
        request.app.state,
        "runtime",
        None,
    )

    if (
        not isinstance(
            runtime,
            ApplicationRuntime,
        )
        or not runtime.ready
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "AI runtime is not ready."
            ),
        )

    return {
        "status":
            "ready",

        "service":
            "itsm-ai",
    }


# ============================================================
# Transitional synchronous endpoint
# ============================================================


@app.post(
    "/v1/agent/run"
)
async def run_agent(
    payload: AgentRunRequest,
    request: Request,
) -> dict:
    runtime = (
        require_ready_runtime(
            request
        )
    )

    result = (
        await
        runtime.hub.run(
            payload.message
        )
    )

    return {
        "request_id":
            uuid.uuid4().hex,

        "user_id":
            payload.user_id,

        **asdict(
            result
        ),
    }


# ============================================================
# Durable async execution endpoint
# Handshake #1
# ============================================================


@app.post(
    "/v1/jobs/execute",
    status_code=(
        status.HTTP_202_ACCEPTED
    ),
)
async def execute_durable_job(
    payload: JobExecuteRequest,
    request: Request,
) -> dict:
    runtime = (
        require_ready_runtime(
            request
        )
    )

    existing_task = (
        runtime
        .active_jobs
        .get(
            payload.job_id
        )
    )

    if (
        existing_task
        is not None
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
        runtime,
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
    "/v1/approvals/"
    "{approval_id}/approve"
)
async def approve(
    approval_id: str,
    request: Request,
) -> dict:
    runtime = (
        require_ready_runtime(
            request
        )
    )

    approval_result = (
        await
        runtime
        .approvals
        .approve_approval(
            approval_id
        )
    )

    if not approval_result.get(
        "ok"
    ):
        error = (
            approval_result.get(
                "error"
            )
            or approval_result
            .get(
                "result",
                {},
            )
            .get(
                "error"
            )
            or (
                "Approval execution failed."
            )
        )

        raise HTTPException(
            status_code=400,
            detail=(
                error
            ),
        )

    return {
        "approval_id":
            approval_id,

        "status":
            "executed",

        "result":
            approval_result[
                "result"
            ],
    }