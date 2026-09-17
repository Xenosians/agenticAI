from pydantic import (
    BaseModel,
)

from mcp.server import (
    MCPServer,
)

from services.directory import (
    AccessMutationService,
)


class AccessMutationMCPResult(
    BaseModel
):
    ok: bool

    status: str

    changed: bool = False

    user_id: (
        str | None
    ) = None

    resource: (
        str | None
    ) = None

    has_access: (
        bool | None
    ) = None

    message: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


def register_access_mutation_tools(
    server: MCPServer,
    access_mutations: AccessMutationService,
) -> None:

    @server.tool()
    def grant_access(
        user_id: str,
        resource: str,
    ) -> AccessMutationMCPResult:

        result = (
            access_mutations
            .grant_access(
                user_id,
                resource,
            )
        )

        return (
            AccessMutationMCPResult(
                **result.model_dump()
            )
        )

    @server.tool()
    def revoke_access(
        user_id: str,
        resource: str,
    ) -> AccessMutationMCPResult:

        result = (
            access_mutations
            .revoke_access(
                user_id,
                resource,
            )
        )

        return (
            AccessMutationMCPResult(
                **result.model_dump()
            )
        )
