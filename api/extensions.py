import secrets

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict

from config import get_settings
from services.extensions.service import command


router = APIRouter()


def require_internal(request: Request):
    expected = get_settings().require_internal_job_token()
    supplied = request.headers.get("x-internal-token", "")
    if not supplied or not secrets.compare_digest(expected, supplied):
        raise HTTPException(401, "Internal authentication required")


class ExtensionCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner: str = Field(min_length=1, max_length=200)
    action: str = Field(min_length=1, max_length=40)
    payload: dict = Field(default_factory=dict)


@router.post("/internal/v1/extensions")
async def extensions(body: ExtensionCommand, request: Request):
    require_internal(request)
    try:
        return await command(body.owner, body.action, body.payload, get_settings())
    except (ValueError, KeyError, TypeError):
        # Public responses must never interpolate stored tokens or raw provider bodies.
        raise HTTPException(422, "Extension request rejected. Check fields, resource ownership, admission and endpoint access.") from None
