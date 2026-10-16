from mcp.server import MCPServer

from services.mailbox import MailboxResult, MaddyMailboxService


def register_mailbox_tools(server: MCPServer, service: MaddyMailboxService) -> None:
    @server.tool()
    def mailbox_create(email: str) -> MailboxResult:
        return service.create(email)

    @server.tool()
    def mailbox_status(email: str) -> MailboxResult:
        return service.status(email)
