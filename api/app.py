from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from agent.mcp_client import mcp_runtime
from subagents.hub import build_hub
import uuid
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from agent.approvals import approve_approval


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.hub = None

    await mcp_runtime.start()

    try:
        print("[API] Loading AI runtime...")

        app.state.hub = build_hub()
        app.state.ready = True

        print("[API] AI runtime ready.")

        yield

    finally:
        app.state.ready = False
        app.state.hub = None

        await mcp_runtime.stop()

        print("[API] AI runtime stopped.")


app = FastAPI(
    title="ITSM AI Service",
    version="0.1.0",
    lifespan=lifespan,
)

class AgentRunRequest(BaseModel):
    user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "itsm-ai",
    }


@app.get("/ready")
async def ready() -> dict[str, str]:
    if not app.state.ready:
        raise HTTPException(
            status_code=503,
            detail="AI runtime is not ready.",
        )

    return {
        "status": "ready",
        "service": "itsm-ai",
    }
    
@app.post("/v1/agent/run")
async def run_agent(
    payload: AgentRunRequest,
    request: Request,
) -> dict:
    if not request.app.state.ready:
        raise HTTPException(
            status_code=503,
            detail="AI runtime is not ready.",
        )

    hub = request.app.state.hub

    result = await hub.run(
        payload.message
    )

    return {
        "request_id": uuid.uuid4().hex,
        "user_id": payload.user_id,
        **asdict(result),
    }
    
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
            detail="AI runtime is not ready.",
        )

    approval_result = await approve_approval(
        approval_id
    )

    if not approval_result.get("ok"):
        error = (
            approval_result.get("error")
            or approval_result
            .get("result", {})
            .get("error")
            or "Approval execution failed."
        )

        raise HTTPException(
            status_code=400,
            detail=error,
        )

    return {
        "approval_id": approval_id,
        "status": "executed",
        "result": approval_result["result"],
    }