from pydantic import (
    BaseModel,
)

from mcp.server import (
    MCPServer,
)

from services.directory import (
    AccountLifecycleService,
)


class AccountLifecycleMCPResult(
    BaseModel
):
    ok: bool

    status: str

    changed: bool = False

    user_id: (
        str | None
    ) = None

    enabled: (
        bool | None
    ) = None

    message: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


def register_account_lifecycle_tools(
    server: MCPServer,
    lifecycle: AccountLifecycleService,
) -> None:

    @server.tool()
    def enable_user(
        user_id: str,
    ) -> AccountLifecycleMCPResult:

        result = (
            lifecycle
            .enable_user(
                user_id
            )
        )

        return (
            AccountLifecycleMCPResult(
                **result.model_dump()
            )
        )

    @server.tool()
    def disable_user(
        user_id: str,
    ) -> AccountLifecycleMCPResult:

        result = (
            lifecycle
            .disable_user(
                user_id
            )
        )

        return (
            AccountLifecycleMCPResult(
                **result.model_dump()
            )
        )
