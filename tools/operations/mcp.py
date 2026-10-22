"""Platform operations tools."""
from typing import Any

from mcp.server import MCPServer
from pydantic import BaseModel

from config import Settings
from services.notifications import UserNotificationService
from services.operations import OperationsService


class NotificationResult(BaseModel):
    ok: bool
    status: str
    provider: str
    operation: str
    email: str | None = None
    queued: bool = False
    outbox_id: str | None = None
    delivery_status: str | None = None
    reconciliation_required: bool = False
    retry_safe: bool | None = None
    error: str | None = None
    error_code: str | None = None


class OperationsResult(BaseModel):
    ok: bool
    status: str
    provider: str
    observed_at: str | None = None
    data: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None


def register_operations_tools(server: MCPServer, settings: Settings, service: OperationsService | None = None,
                              notifications: UserNotificationService | None = None) -> None:
    """Register platform operations tools."""
    service = service or OperationsService(settings)
    notifications = notifications or UserNotificationService(settings)

    @server.tool()
    def container_status(container: str | None = None) -> OperationsResult:
        return OperationsResult(**service.container_status(container))

    @server.tool()
    def container_logs(container: str, lines: int | None = None) -> OperationsResult:
        return OperationsResult(**service.container_logs(container, lines))

    @server.tool()
    def service_health() -> OperationsResult:
        return OperationsResult(**service.service_health())

    @server.tool()
    def notify_user(email: str, message: str, subject: str | None = None,
                    dedupe_key: str | None = None) -> NotificationResult:
        return NotificationResult(**notifications.notify(email=email, message=message, subject=subject,
                                                         dedupe_key=dedupe_key))
