from pydantic import BaseModel
from mcp.server import MCPServer

from services.directory import get_directory_service


mcp = MCPServer("ITSM Tools")


class AccountStatusResult(BaseModel):
    ok: bool
    user_id: str | None = None
    enabled: bool | None = None
    locked: bool | None = None
    error: str | None = None


class AccessCheckResult(BaseModel):
    ok: bool
    user_id: str | None = None
    resource: str | None = None
    has_access: bool | None = None
    error: str | None = None


class MutationResult(BaseModel):
    ok: bool
    status: str
    changed: bool | None = None
    user_id: str | None = None
    password_reset_count: int | None = None
    message: str | None = None
    error: str | None = None


@mcp.tool()
def account_status(
    user_id: str,
) -> AccountStatusResult:
    directory = get_directory_service()

    result = directory.account_status(
        user_id
    )

    return AccountStatusResult(**result)


@mcp.tool()
def check_access(
    user_id: str,
    resource: str,
) -> AccessCheckResult:
    directory = get_directory_service()

    result = directory.check_access(
        user_id,
        resource,
    )

    return AccessCheckResult(**result)


@mcp.tool()
def unlock_user(
    user_id: str,
) -> MutationResult:
    directory = get_directory_service()

    result = directory.unlock_user(
        user_id
    )

    return MutationResult(**result)


@mcp.tool()
def reset_password(
    user_id: str,
) -> MutationResult:
    directory = get_directory_service()

    result = directory.reset_password(
        user_id
    )

    return MutationResult(**result)


if __name__ == "__main__":
    mcp.run()