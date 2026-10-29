"""Authenticated Phoenix-to-AI Learning: catalog, readiness and candidate runs."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import Field

from api.extensions import require_internal
from services.extensions.service import extension_store
from services.learning_control.contracts import Contract
from services.learning_control.application import (
    LearningSelectionError, ResourceId, SelectionRequest, learning_catalog, selection_preflight,
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


class OwnedSubmission(OwnedCatalog):
    submission_id: ResourceId


def run_error(exc: LearningSelectionError, *, missing_status: int = 422) -> HTTPException:
    code = str(exc)
    status = {"learning_host_busy": 409, "learning_submission_conflict": 409,
              "learning_run_outcome_unknown": 503, "learning_registry_unavailable": 503,
              "learning_run_unavailable": missing_status}.get(code, 422)
    return HTTPException(status, code)


@router.post("/internal/v1/learning/runs")
async def runs(body: OwnedCatalog, request: Request):
    require_internal(request)
    try:
        return {"schema": "learning-runs.v2", "runs": await run_in_threadpool(run_registry().list, body.owner)}
    except LearningSelectionError as exc:
        raise run_error(exc) from None


@router.post("/internal/v1/learning/runs/resolve")
async def resolve_run(body: OwnedSubmission, request: Request):
    require_internal(request)
    try:
        return await run_in_threadpool(run_registry().resolve, body.owner, body.submission_id)
    except LearningSelectionError as exc:
        raise run_error(exc, missing_status=404) from None


@router.post("/internal/v1/learning/runs/start")
async def start_run(body: OwnedRun, request: Request):
    require_internal(request)
    try:
        return await run_in_threadpool(run_registry().start, body.owner, body.run, extension_store())
    except LearningSelectionError as exc:
        raise run_error(exc) from None


@router.post("/internal/v1/learning/runs/cancel")
async def cancel_run(body: OwnedRunIdentity, request: Request):
    require_internal(request)
    try:
        return await run_in_threadpool(run_registry().cancel, body.owner, body.run_id)
    except LearningSelectionError as exc:
        raise run_error(exc, missing_status=404) from None
