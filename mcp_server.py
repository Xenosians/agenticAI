from pydantic import BaseModel
from mcp.server import MCPServer

from tools.account import account_status as local_account_status
from tools.access import check_access as local_check_access


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


@mcp.tool()
def account_status(
    user_id: str,
) -> AccountStatusResult:
    """
    Check whether an Active Directory account
    is enabled or locked.
    """

    result = local_account_status(user_id)

    return AccountStatusResult(**result)


@mcp.tool()
def check_access(
    user_id: str,
    resource: str,
) -> AccessCheckResult:
    """
    Check whether a user has access
    to a resource.
    """

    result = local_check_access(
        user_id,
        resource,
    )

    return AccessCheckResult(**result)


if __name__ == "__main__":
    mcp.run()