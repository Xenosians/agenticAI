"""Authenticated Phoenix-to-AI engineering reads (FR-093): drift and netlist inspection.

Phoenix owns topologies and their owners; it sends one exact saved version.
These routes observe through the fixed read adapters and compare. They never
change a device and never accept a provider, URL or path from the request.
"""
import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from api.extensions import require_internal
from config import get_settings
from services.engineering.netlist import NetlistError, inspect_netlist
from services.engineering.observe import observe
from services.engineering.topology import TopologyError, compare, parse_topology
from services.openwrt import build_openwrt_service
from services.operations import OperationsService

router = APIRouter()


class DriftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    topology_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    version: int = Field(gt=0)
    topology: dict


class NetlistRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    netlist: str


def observation_sources() -> dict:
    """The same fixed read services the container, health and network tools use."""
    settings = get_settings()
    operations = OperationsService(settings)
    return {"containers": operations, "endpoints": operations, "router": build_openwrt_service(settings),
            "default_interfaces": list(settings.engineering_router_interfaces)}


async def _body(request: Request, model):
    try:
        return model.model_validate(await request.json())
    except Exception:  # noqa: BLE001 - malformed JSON and schema errors share one public code
        raise HTTPException(422, "invalid_engineering_request") from None


def _observe_and_compare(topology: dict, topology_id: str, version: int) -> dict:
    """Blocking reads; runs in a worker thread so this service can still answer
    its own health probe, which is one of the observed endpoints."""
    sources = observation_sources()
    router_service = sources["router"]
    try:
        observation = observe(containers=sources["containers"], endpoints=sources["endpoints"],
                              router=router_service, topology=topology,
                              default_interfaces=sources["default_interfaces"])
    finally:
        if router_service is not None:
            router_service.close()
    return compare(topology, observation, topology_id=topology_id, version=version)


@router.post("/internal/v1/engineering/topology/drift")
async def topology_drift(request: Request):
    require_internal(request)
    body = await _body(request, DriftRequest)
    try:
        topology = parse_topology(body.topology)
    except TopologyError as exc:
        raise HTTPException(422, exc.code) from None
    return await asyncio.to_thread(_observe_and_compare, topology, body.topology_id, body.version)


@router.post("/internal/v1/engineering/netlist")
async def netlist(request: Request):
    require_internal(request)
    body = await _body(request, NetlistRequest)
    try:
        # Parsing up to 2 MB is CPU work; keep the event loop free for other requests.
        return await asyncio.to_thread(inspect_netlist, body.netlist)
    except NetlistError as exc:
        raise HTTPException(422, exc.code) from None
