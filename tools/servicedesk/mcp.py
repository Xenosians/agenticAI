"""Jira Service Management tools."""
from mcp.server import MCPServer

from config import Settings
from services.servicedesk import (
    NOT_CONFIGURED,
    QueueListResult,
    ServiceDeskService,
    ServiceRequestResult,
    SlaStatusResult,
    build_service_desk_service,
)


def register_service_desk_tools(
    server: MCPServer,
    settings: Settings,
    service: ServiceDeskService | None = None,
) -> None:
    """Register service desk tools; they report "not configured" without Jira."""

    provider = service if service is not None else build_service_desk_service(settings)

    @server.tool()
    def service_request_create(
        service_desk: str,
        request_type: str,
        summary: str,
        description: str | None = None,
        expected_service_desk_id: str | None = None,
        expected_request_type_id: str | None = None,
    ) -> ServiceRequestResult:
        if provider is None:
            return ServiceRequestResult(ok=False, status="error", error=NOT_CONFIGURED, error_code="not_configured")
        return provider.create_request(
            service_desk=service_desk,
            request_type=request_type,
            summary=summary,
            description=description,
            expected_service_desk_id=expected_service_desk_id,
            expected_request_type_id=expected_request_type_id,
        )

    @server.tool()
    def sla_status(request_key: str) -> SlaStatusResult:
        if provider is None:
            return SlaStatusResult(ok=False, status="error", error=NOT_CONFIGURED, error_code="not_configured")
        return provider.sla_status(request_key)

    @server.tool()
    def queue_list(service_desk: str) -> QueueListResult:
        if provider is None:
            return QueueListResult(ok=False, status="error", error=NOT_CONFIGURED, error_code="not_configured")
        return provider.queue_list(service_desk)
