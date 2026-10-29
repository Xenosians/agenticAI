"""Authenticated Phoenix-to-AI Learning: catalog, readiness and candidate runs."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import Field

from api.extensions import require_internal
from services.extensions.service import extension_store
from services.learning_control.contracts import Contract
from services.learning_control.application import (
    LearningSelectionError, SelectionRequest, learning_catalog, selection_preflight,
)
from services.learning_control.runs import RunRequest, run_registry

router = APIRouter()


class OwnedCatalog(Contract):
    owner: str = Field(min_length=1, max_length=200)


class OwnedPreflight(OwnedCatalog):
    selection: SelectionRequest


@router.post("/internal/v1/learning/catalog")
async def catalog(body: OwnedCatalog, request: Request):
    require_internal(request)
    try:
        return learning_catalog(body.owner, extension_store())
    except LearningSelectionError:
        raise HTTPException(503, "learning_service_unavailable") from None


@router.post("/internal/v1/learning/preflight")
async def preflight(body: OwnedPreflight, request: Request):
    require_internal(request)
    try:
        return selection_preflight(body.owner, body.selection, extension_store())
    except LearningSelectionError as exc:
        raise HTTPException(422, str(exc)) from None


class OwnedRun(OwnedCatalog):
    run: RunRequest


class OwnedRunIdentity(OwnedCatalog):
    run_id: str = Field(min_length=1, max_length=64)


@router.post("/internal/v1/learning/runs")
async def runs(body: OwnedCatalog, request: Request):
    require_internal(request)
    return {"schema": "learning-runs.v1", "runs": await run_in_threadpool(run_registry().list, body.owner)}


@router.post("/internal/v1/learning/runs/start")
async def start_run(body: OwnedRun, request: Request):
    require_internal(request)
    try:
        return await run_in_threadpool(run_registry().start, body.owner, body.run, extension_store())
    except LearningSelectionError as exc:
        raise HTTPException(409 if str(exc) == "learning_host_busy" else 422, str(exc)) from None


@router.post("/internal/v1/learning/runs/cancel")
async def cancel_run(body: OwnedRunIdentity, request: Request):
    require_internal(request)
    try:
        return await run_in_threadpool(run_registry().cancel, body.owner, body.run_id)
    except LearningSelectionError as exc:
        raise HTTPException(404, str(exc)) from None
