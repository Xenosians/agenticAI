import secrets

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict

from config import get_settings
from services.extensions.service import command
from services.extensions.probe import ProviderProbeError


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
    except ProviderProbeError as exc:
        # Only curated probe diagnostics may cross the public boundary.
        # Raw provider response bodies and credentials are never forwarded.
        raise HTTPException(
            422,
            exc.public_detail(),
        ) from None

    except (ValueError, KeyError, TypeError):
        # Public responses must never interpolate stored tokens or raw provider bodies.
        raise HTTPException(
            422,
            (
                "Extension request rejected. Check fields, "
                "resource ownership, admission and endpoint access."
            ),
        ) from None
