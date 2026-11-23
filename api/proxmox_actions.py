"""Phoenix-to-AI Proxmox dashboard: live reads and governed guest-change proposals.

Phoenix authenticates the owner and calls these routes with the internal
token. A dashboard button runs through exactly the gateway a chat request
uses: owner tool switches, grounding, trusted policy (which freezes the guest's
type and name) and a durable approval. Nothing here executes an action; the
approval is decided through the existing /v1/approvals routes.
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from api.extensions import require_internal

router = APIRouter()

ACTION_TOOLS = {
    "start": "proxmox_guest_start",
    "shutdown": "proxmox_guest_shutdown",
    "reboot": "proxmox_guest_reboot",
    "force_stop": "proxmox_guest_force_stop",
    "set_resources": "proxmox_guest_set_resources",
    "resize_disk": "proxmox_guest_resize_disk",
    "set_options": "proxmox_guest_set_options",
    "snapshot_create": "proxmox_snapshot_create",
    "snapshot_delete": "proxmox_snapshot_delete",
    "snapshot_rollback": "proxmox_snapshot_rollback",
    "backup_now": "proxmox_backup_now",
    "backup_restore": "proxmox_backup_restore",
    "clone": "proxmox_guest_clone",
    "create": "proxmox_guest_create",
    "delete": "proxmox_guest_delete",
    "database_create": "proxmox_database_create",
}
NEW_GUEST_ACTIONS = frozenset({"create", "backup_restore", "database_create"})
POWER_ACTIONS = frozenset({"start", "shutdown", "reboot", "force_stop"})
SPECIALIST = "proxmox-specialist"
# The signed-in account, sent by Phoenix from its session (never by the browser);
# the read uses that account's own Proxmox link, else the machine default.
OWNER = Query(default=None, min_length=1, max_length=200)


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
    vmid: StrictInt | None = Field(default=None, ge=1, le=999_999_999)
    action: Literal[
        "start", "shutdown", "reboot", "force_stop", "set_resources", "resize_disk", "set_options",
        "snapshot_create", "snapshot_delete", "snapshot_rollback", "backup_now", "backup_restore", "clone",
        "create", "delete", "database_create",
    ]
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_change_parameters(self):
        _validate_action_parameters(self.action, self.vmid, self.params)
        return self


def _validate_action_parameters(action: str, vmid: int | None, params: dict[str, Any]) -> None:
    from tools.registry import get_tool

    if not isinstance(params, dict):
        raise ValueError("Proxmox action parameters must be an object.")
    if action not in NEW_GUEST_ACTIONS and vmid is None:
        raise ValueError("This Proxmox action requires an exact guest VMID.")

    tool = get_tool(ACTION_TOOLS[action])
    if not isinstance(tool, dict):
        raise ValueError("The Proxmox action has no registered capability.")

    normalized = dict(params)
    if action in ("create", "database_create") and "newid" in normalized:
        requested = normalized.pop("newid")
        if "vmid" in normalized and normalized["vmid"] != requested:
            raise ValueError("The requested new VMID was supplied twice with different values.")
        normalized["vmid"] = requested
    if action in NEW_GUEST_ACTIONS and vmid is not None:
        parameter_vmid = normalized.get("newid" if action == "backup_restore" else "vmid")
        if parameter_vmid is not None and vmid != parameter_vmid:
            raise ValueError("The requested new VMID was supplied twice with different values.")

    schema = tool.get("parameters")
    if not isinstance(schema, dict):
        raise ValueError("The Proxmox capability has an invalid parameter schema.")
    # Only what an owner names: never a trusted argument (a frozen plan, an allocated VMID).
    allowed = set(schema) - {"node"} - set(tool.get("trusted_policy_arguments") or ())
    if set(normalized) - allowed:
        raise ValueError("Proxmox action contains unsupported parameters.")
    if any(not _matches_parameter_type(value, schema[key].get("type")) for key, value in normalized.items()):
        raise ValueError("Proxmox action parameter types are invalid.")
    required = set(tool.get("required_arguments", [])) - {"node", "vmid"}
    if not required <= normalized.keys():
        raise ValueError("Proxmox action is missing required parameters.")


def _matches_parameter_type(value: Any, expected: str | None) -> bool:
    if expected == "int":
        return type(value) is int and 1 <= value <= 999_999_999
    if expected == "bool":
        return type(value) is bool
    if expected == "str":
        return isinstance(value, str) and len(value) <= 256
    return False


def _tool_arguments(body: ActionProposal) -> dict[str, Any]:
    arguments: dict[str, Any] = {"node": body.node}
    if body.action not in NEW_GUEST_ACTIONS and body.vmid is not None:
        arguments["vmid"] = body.vmid
    arguments.update(body.params)
    if body.action in NEW_GUEST_ACTIONS:
        requested_vmid = body.vmid if body.vmid is not None else body.params.get("newid")
        if requested_vmid is not None:
            target_argument = "newid" if body.action == "backup_restore" else "vmid"
            arguments[target_argument] = requested_vmid
        if body.action != "backup_restore":
            arguments.pop("newid", None)
    return arguments


def _proposal_instruction(action: str, node: str, arguments: dict[str, Any]) -> str:
    values = ", ".join(f"{key}={value}" for key, value in arguments.items() if key != "node")
    target = f"guest VMID {arguments['vmid']}" if "vmid" in arguments else "a new guest"
    if action in NEW_GUEST_ACTIONS and not ("vmid" in arguments or "newid" in arguments):
        values = ", ".join(part for part in (values, "leave the new VMID for Proxmox to allocate") if part)
    detail = f" Exact supplied values: {values}." if values else ""
    return f"Owner requested {action.replace('_', ' ')} for {target} on Proxmox node {node}.{detail}"


@router.post("/internal/v1/proxmox/actions/propose")
async def propose_action(body: ActionProposal, request: Request):
    """Create one pending approval for the exact guest, or say why not."""
    require_internal(request)
    runtime = require_ready_runtime(request)
    from services.extensions.context import disabled_tools, request_owner
    from tools.registry import format_approval_required

    tool = ACTION_TOOLS[body.action]
    agent = runtime.hub.runtime.agent_registry.get(SPECIALIST)
    # The authenticated backend forwards explicit form values. Include their
    # exact text in the grounding request; the gateway still applies the same
    # tool schema, owner switches, live policy and approval path used by chat.
    arguments = _tool_arguments(body)
    instruction = _proposal_instruction(body.action, body.node, arguments)
    owner_token = request_owner.set(body.owner)
    disabled_token = disabled_tools.set(owner_disabled_tools(body.owner))
    try:
        result = await runtime.tool_gateway.execute(agent, instruction, tool, arguments)
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
    approval_arguments = result.get("approval_arguments")
    if not isinstance(approval_arguments, dict):
        raise HTTPException(503, {"code": "proxmox_action_unavailable",
                                  "message": "The AI service returned an invalid approval proposal."})

    allocated_vmid = approval_arguments.get("allocated_vmid")
    new_vmid = allocated_vmid
    if new_vmid is None:
        new_vmid = approval_arguments.get("newid")
    if new_vmid is None and body.action in ("create", "database_create"):
        new_vmid = approval_arguments.get("vmid")

    if body.action == "clone":
        target_vmid = approval_arguments.get("vmid")
    elif body.action in NEW_GUEST_ACTIONS:
        target_vmid = new_vmid
    else:
        target_vmid = approval_arguments.get("vmid")
    if type(target_vmid) is not int or not (1 <= target_vmid <= 999_999_999):
        raise HTTPException(503, {"code": "proxmox_action_unavailable",
                                  "message": "The AI service did not return the exact approved VMID."})

    response = {
        "status": "approval_required",
        "approval_id": result["approval_id"],
        "tool": tool,
        "action": body.action,
        "risk": result.get("risk"),
        "node": approval_arguments.get("node"),
        "vmid": target_vmid,
        "type": approval_arguments.get("expected_type") or approval_arguments.get("type"),
        "name": (approval_arguments.get("expected_name") or approval_arguments.get("name")
                 or approval_arguments.get("hostname")),
        "approval_text": format_approval_required(tool, approval_arguments, None),
    }
    if approval_arguments.get("node") != body.node:
        raise HTTPException(503, {"code": "proxmox_action_unavailable",
                                  "message": "The AI service returned a different Proxmox node."})
    if body.action in {"clone", *NEW_GUEST_ACTIONS}:
        if type(new_vmid) is not int or not (1 <= new_vmid <= 999_999_999):
            raise HTTPException(503, {"code": "proxmox_action_unavailable",
                                      "message": "The AI service did not return the exact approved new VMID."})
        response["new_vmid"] = new_vmid
    return response


@router.get("/internal/v1/proxmox/overview")
def proxmox_overview(request: Request, owner: str | None = OWNER):
    """Return a live Proxmox snapshot of the signed-in account to the authenticated backend.

    A failed read names its cause (unreachable, timeout, refused, invalid
    answer, a Lua refusal) with a fixed message that never echoes the provider.
    """
    from services.provider_links import as_owner, owner_proxmox_service
    from tools.provider_reads import provider_read

    require_internal(request)
    runtime = require_ready_runtime(request)
    try:
        with as_owner(owner):
            service = owner_proxmox_service(runtime.settings)
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
def proxmox_health(request: Request, owner: str | None = OWNER):
    """Return the signed-in account's backups and every node's physical disks to the authenticated backend.

    Read separately from the overview (it lists backup storages and runs SMART
    on each node), so the dashboard refreshes it less often. Each part is a
    typed read result: one unreadable node's disks never hide the rest.
    """
    from services.provider_links import as_owner, owner_proxmox_service
    from tools.provider_reads import provider_read

    require_internal(request)
    runtime = require_ready_runtime(request)
    try:
        with as_owner(owner):
            service = owner_proxmox_service(runtime.settings)
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
