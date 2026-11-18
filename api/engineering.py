"""Authenticated Phoenix-to-AI engineering reads (FR-093): drift and netlist inspection.

Phoenix owns topologies and their owners; it sends one exact saved version
and names the signed-in account in the query, as for the other dashboard reads,
so the tailnet observed is that account's own. These routes observe through the
fixed read adapters and compare. They never change a device and never accept a
provider, URL or path from the request.
"""
import asyncio

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from api.extensions import require_internal
from config import get_settings
from services.engineering.netlist import NetlistError, inspect_netlist
from services.engineering.observe import observe
from services.engineering.topology import TopologyError, compare, parse_topology
from services.openwrt import build_openwrt_service
from services.operations import OperationsService
from services.provider_links import as_owner, owner_tailnet_service

router = APIRouter()


class DriftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    topology_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    version: int = Field(gt=0)
    topology: dict


class NetlistRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    netlist: str


def observation_sources(owner: str | None = None) -> dict:
    """The same fixed read services the container, health and network tools use.

    The tailnet is the account's own link, else the machine default (owner
    decision, 17 November: every connection per account, not per machine).
    """
    settings = get_settings()
    operations = OperationsService(settings)
    with as_owner(owner):
        tailnet = owner_tailnet_service(settings)
    return {"containers": operations, "endpoints": operations, "router": build_openwrt_service(settings),
            "tailnet": tailnet, "default_interfaces": list(settings.engineering_router_interfaces)}


async def _body(request: Request, model):
    try:
        return model.model_validate(await request.json())
    except Exception:  # noqa: BLE001 - malformed JSON and schema errors share one public code
        raise HTTPException(422, "invalid_engineering_request") from None


def _observe_and_compare(topology: dict, topology_id: str, version: int, owner: str | None) -> dict:
    """Blocking reads; runs in a worker thread so this service can still answer
    its own health probe, which is one of the observed endpoints."""
    sources = observation_sources(owner)
    readers = [sources[name] for name in ("router", "tailnet") if sources.get(name) is not None]
    try:
        observation = observe(containers=sources["containers"], endpoints=sources["endpoints"],
                              router=sources["router"], tailnet=sources.get("tailnet"), topology=topology,
                              default_interfaces=sources["default_interfaces"])
    finally:
        for reader in readers:
            reader.close()
    return compare(topology, observation, topology_id=topology_id, version=version)


@router.post("/internal/v1/engineering/topology/drift")
async def topology_drift(request: Request, owner: str | None = Query(default=None, max_length=200)):
    require_internal(request)
    if owner is not None and not owner:
        raise HTTPException(422, "invalid_engineering_request")
    body = await _body(request, DriftRequest)
    try:
        topology = parse_topology(body.topology)
    except TopologyError as exc:
        raise HTTPException(422, exc.code) from None
    return await asyncio.to_thread(_observe_and_compare, topology, body.topology_id, body.version, owner)


@router.post("/internal/v1/engineering/netlist")
async def netlist(request: Request):
    require_internal(request)
    body = await _body(request, NetlistRequest)
    try:
        # Parsing up to 2 MB is CPU work; keep the event loop free for other requests.
        return await asyncio.to_thread(inspect_netlist, body.netlist)
    except NetlistError as exc:
        raise HTTPException(422, exc.code) from None
