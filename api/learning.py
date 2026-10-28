"""Authenticated Phoenix-to-AI Learning inspection; no training dispatch."""
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from api.extensions import require_internal
from services.extensions.service import extension_store
from services.learning_control.contracts import Contract
from services.learning_control.application import (
    LearningSelectionError, SelectionRequest, learning_catalog, selection_preflight,
)

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
