from __future__ import annotations

from pydantic import BaseModel

from services.palo_alto import PaloAltoService


class PaloAltoSystemInfoResult(BaseModel):
    ok: bool
    status: str
    hostname: str | None = None
    ip_address: str | None = None
    model: str | None = None
    serial: str | None = None
    sw_version: str | None = None
    app_version: str | None = None
    av_version: str | None = None
    uptime: str | None = None


def register_palo_alto_tools(server, service: PaloAltoService | None) -> None:
    @server.tool()
    def palo_alto_system_info() -> PaloAltoSystemInfoResult:
        if service is None:
            return PaloAltoSystemInfoResult(
                ok=False,
                status="not_configured",
            )

        return PaloAltoSystemInfoResult(
            **service.system_info()
        )
