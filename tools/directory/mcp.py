"""Directory read tools."""
from pydantic import BaseModel, Field

from mcp.server import MCPServer

from services.directory import DirectoryService


class DirectoryAccount(BaseModel):
    username: str
    display_name: str | None = None
    email: str | None = None
    enabled: bool | None = None
    locked: bool | None = None


class AccountSearchResult(BaseModel):
    ok: bool
    query: str | None = None
    accounts: list[DirectoryAccount] = Field(default_factory=list)
    count: int = 0
    truncated: bool = False
    error: str | None = None
    error_code: str | None = None


class DirectoryGroup(BaseModel):
    name: str
    membership: str
    resources: list[str] = Field(default_factory=list)


class UserGroupsResult(BaseModel):
    ok: bool
    user_id: str | None = None
    groups: list[DirectoryGroup] = Field(default_factory=list)
    count: int = 0
    truncated: bool = False
    error: str | None = None
    error_code: str | None = None


class GroupMember(BaseModel):
    name: str
    display_name: str | None = None
    type: str | None = None
    membership: str


class GroupMembersResult(BaseModel):
    ok: bool
    group: str | None = None
    resources: list[str] = Field(default_factory=list)
    members: list[GroupMember] = Field(default_factory=list)
    count: int = 0
    truncated: bool = False
    error: str | None = None
    error_code: str | None = None


class AccessChecks(BaseModel):
    account_exists: bool | None = None
    enabled: bool | None = None
    locked: bool | None = None
    resource_mapped: bool | None = None
    membership: str | None = None


class AccessReason(BaseModel):
    code: str
    message: str
    suggested_tool: str | None = None


class AccessExplainResult(BaseModel):
    ok: bool
    user_id: str | None = None
    resource: str | None = None
    group: str | None = None
    has_access: bool | None = None
    checks: AccessChecks = Field(default_factory=AccessChecks)
    reasons: list[AccessReason] = Field(default_factory=list)
    error: str | None = None
    error_code: str | None = None


def register_directory_tools(server: MCPServer, directory: DirectoryService) -> None:
    """Register directory read tools."""

    @server.tool()
    def account_search(query: str) -> AccountSearchResult:
        return AccountSearchResult(**directory.account_search(query))

    @server.tool()
    def user_groups(user_id: str) -> UserGroupsResult:
        return UserGroupsResult(**directory.user_groups(user_id))

    @server.tool()
    def group_members(group: str) -> GroupMembersResult:
        return GroupMembersResult(**directory.group_members(group))

    @server.tool()
    def access_explain(user_id: str, resource: str) -> AccessExplainResult:
        return AccessExplainResult(**directory.access_explain(user_id, resource))
