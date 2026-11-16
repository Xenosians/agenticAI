"""Phoenix-to-AI Proxmox dashboard: the live overview and governed power-action proposals.

Phoenix authenticates the owner and calls these routes with the internal
token. A dashboard button runs through exactly the gateway a chat request
uses: owner tool switches, grounding, trusted policy (which freezes the guest's
type and name) and a durable approval. Nothing here executes an action; the
approval is decided through the existing /v1/approvals routes.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from api.extensions import require_internal

router = APIRouter()

ACTION_TOOLS = {
    "start": "proxmox_guest_start",
    "shutdown": "proxmox_guest_shutdown",
    "reboot": "proxmox_guest_reboot",
    "force_stop": "proxmox_guest_force_stop",
}
ACTION_WORDS = {"start": "Start", "shutdown": "Shut down", "reboot": "Reboot", "force_stop": "Force stop"}
SPECIALIST = "proxmox-specialist"


def require_ready_runtime(request: Request):
    from api.app import require_ready_runtime as ready

    return ready(request)


def owner_disabled_tools(owner: str) -> frozenset[str]:
    from services.extensions.service import extension_store
    from services.extensions.tool_switches import ToolSwitches

    return frozenset(ToolSwitches(extension_store()).disabled(owner))


class ActionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner: str = Field(min_length=1, max_length=200)
    node: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    vmid: StrictInt = Field(ge=1, le=999_999_999)
    action: Literal["start", "shutdown", "reboot", "force_stop"]


@router.post("/internal/v1/proxmox/actions/propose")
async def propose_action(body: ActionProposal, request: Request):
    """Create one pending approval for the exact guest, or say why not."""
    require_internal(request)
    runtime = require_ready_runtime(request)
    from services.extensions.context import disabled_tools, request_owner
    from tools.registry import format_approval_required

    tool = ACTION_TOOLS[body.action]
    agent = runtime.hub.runtime.agent_registry.get(SPECIALIST)
    # The dashboard request is the owner's explicit, exact instruction; the
    # gateway grounds node and VMID in this text like any chat request.
    instruction = f"{ACTION_WORDS[body.action]} guest {body.vmid} on Proxmox node {body.node}."
    owner_token = request_owner.set(body.owner)
    disabled_token = disabled_tools.set(owner_disabled_tools(body.owner))
    try:
        result = await runtime.tool_gateway.execute(agent, instruction, tool, {"node": body.node, "vmid": body.vmid})
    finally:
        disabled_tools.reset(disabled_token)
        request_owner.reset(owner_token)

    if result.get("status") != "approval_required" or not result.get("approval_id"):
        message = str(result.get("error") or "The action could not be prepared.")[:300]
        if result.get("decision_code") == "tool_disabled_by_owner":
            raise HTTPException(403, {"code": "proxmox_action_disabled", "message": message})
        if result.get("status") == "denied":
            raise HTTPException(409, {"code": "proxmox_action_refused", "message": message})
        raise HTTPException(503, {"code": "proxmox_action_unavailable", "message": message})
    arguments = result["approval_arguments"]
    return {
        "status": "approval_required",
        "approval_id": result["approval_id"],
        "tool": tool,
        "action": body.action,
        "risk": result.get("risk"),
        "node": arguments["node"],
        "vmid": arguments["vmid"],
        "type": arguments.get("expected_type"),
        "name": arguments.get("expected_name"),
        "approval_text": format_approval_required(tool, arguments, None),
    }


@router.get("/internal/v1/proxmox/overview")
def proxmox_overview(request: Request):
    """Return a live Proxmox snapshot to the authenticated backend.

    A failed read names its cause (unreachable, timeout, refused, invalid
    answer, a Lua refusal) with a fixed message that never echoes the provider.
    """
    from services.proxmox import build_proxmox_service
    from tools.provider_reads import provider_read

    require_internal(request)
    runtime = require_ready_runtime(request)
    try:
        service = build_proxmox_service(runtime.settings)
    except ValueError:
        return JSONResponse(status_code=503, content={"error": "proxmox_configuration_invalid"})
    if service is None:
        return JSONResponse(status_code=503, content={"error": "proxmox_not_configured"})
    try:
        result = provider_read("proxmox", "Proxmox VE", service.overview)
    finally:
        service.close()
    if result.get("ok"):
        return result
    return JSONResponse(status_code=502, content={"error": "proxmox_read_failed",
                                                  "reason": result.get("error_code") or "provider_error",
                                                  "message": result.get("error") or "Proxmox VE could not be read."})


@router.get("/internal/v1/proxmox/health")
def proxmox_health(request: Request):
    """Return backups and every node's physical disks to the authenticated backend.

    Read separately from the overview (it lists backup storages and runs SMART
    on each node), so the dashboard refreshes it less often. Each part is a
    typed read result: one unreadable node's disks never hide the rest.
    """
    from services.proxmox import build_proxmox_service
    from tools.provider_reads import provider_read

    require_internal(request)
    runtime = require_ready_runtime(request)
    try:
        service = build_proxmox_service(runtime.settings)
    except ValueError:
        return JSONResponse(status_code=503, content={"error": "proxmox_configuration_invalid"})
    if service is None:
        return JSONResponse(status_code=503, content={"error": "proxmox_not_configured"})
    try:
        nodes = provider_read("proxmox", "Proxmox VE", service.nodes)
        backups = provider_read("proxmox", "Proxmox VE", service.backups)
        names = [row["node"] for row in nodes["data"]["nodes"]] if nodes.get("ok") else []
        disks = [{"node": name, "result": provider_read("proxmox", "Proxmox VE",
                                                        lambda name=name: service.node_disks(name))}
                 for name in names]
    finally:
        service.close()
    if not nodes.get("ok"):
        return JSONResponse(status_code=502, content={"error": "proxmox_read_failed",
                                                      "reason": nodes.get("error_code") or "provider_error",
                                                      "message": nodes.get("error") or "Proxmox VE could not be read."})
    return {"schema": "proxmox-health.v1", "backups": backups, "disks": disks}
