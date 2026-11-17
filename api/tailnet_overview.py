"""Phoenix-to-AI tailnet overview: each device's health and the tailnet's DNS settings, read-only.

Phoenix authenticates the owner and calls this route with the internal token.
It reads the same normalized data the networking tools use (no addresses,
keys, endpoints or user details) and changes nothing on the tailnet.
"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from api.extensions import require_internal

router = APIRouter()
LABEL = "The Tailscale network"


def require_ready_runtime(request: Request):
    from api.app import require_ready_runtime as ready

    return ready(request)


@router.get("/internal/v1/tailnet/overview")
def tailnet_overview(request: Request, owner: str | None = Query(default=None, min_length=1, max_length=200)):
    """The signed-in account's tailnet: devices decide whether the overview succeeds; a failed DNS read is named."""
    from services.provider_links import as_owner, owner_tailnet_service
    from tools.provider_reads import provider_read

    require_internal(request)
    runtime = require_ready_runtime(request)
    with as_owner(owner):
        service = owner_tailnet_service(runtime.settings)
    if service is None:
        return JSONResponse(status_code=503, content={"error": "tailnet_not_configured"})
    try:
        devices = provider_read("tailscale", LABEL, service.devices)
        dns = provider_read("tailscale", LABEL, service.dns_settings) if devices.get("ok") else None
    finally:
        service.close()
    if not devices.get("ok"):
        return JSONResponse(status_code=502, content={"error": "tailnet_read_failed",
                                                      "reason": devices.get("error_code") or "provider_error",
                                                      "message": devices.get("error") or f"{LABEL} could not be read."})
    dns = dns or {}
    if dns.get("ok"):
        dns_data, dns_error = dns["data"], None
    else:
        dns_data, dns_error = None, {"reason": dns.get("error_code") or "provider_error",
                                     "message": dns.get("error") or f"{LABEL} DNS settings could not be read."}
    return {"ok": True, "status": "success", "provider": "tailscale", "observed_at": devices["observed_at"],
            "data": {"schema": "tailnet-overview.v1", "devices": devices["data"], "dns": dns_data,
                     "dns_error": dns_error}}
