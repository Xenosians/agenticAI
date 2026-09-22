from pydantic import BaseModel, Field
from mcp.server import MCPServer

from services.atlassian import AtlassianAdminService


class CredentialStatusResult(BaseModel):
    ok: bool
    status: str
    admin_api_key_configured: bool = False
    jira_basic_configured: bool = False
    default_org_configured: bool = False
    admin_auth_type: str | None = None
    jira_auth_type: str | None = None
    secret_values_exposed: bool = False
    error: str | None = None


class OrganizationItem(BaseModel):
    id: str
    name: str | None = None
    type: str | None = None


class OrganizationListResult(BaseModel):
    ok: bool
    status: str
    organizations: list[OrganizationItem] = Field(default_factory=list)
    count: int = 0
    truncated: bool = False
    next_cursor_available: bool = False
    error: str | None = None


class OrganizationGetResult(BaseModel):
    ok: bool
    status: str
    organization: OrganizationItem | None = None
    error: str | None = None


class WorkspaceItem(BaseModel):
    id: str
    name: str | None = None
    product_type: str | None = None
    type_key: str | None = None
    status: str | None = None
    host_url: str | None = None
    realm: str | None = None
    regions: list[str] = Field(default_factory=list)


class WorkspaceListResult(BaseModel):
    ok: bool
    status: str
    organization_id: str | None = None
    workspaces: list[WorkspaceItem] = Field(default_factory=list)
    count: int = 0
    truncated: bool = False
    next_cursor_available: bool = False
    error: str | None = None


class TokenMetadataItem(BaseModel):
    id: str
    label: str | None = None
    created_at: str | None = None
    expiry: str | None = None
    last_access: str | None = None


class TokenMetadataResult(BaseModel):
    ok: bool
    status: str
    account_id: str | None = None
    tokens: list[TokenMetadataItem] = Field(default_factory=list)
    count: int = 0
    truncated: bool = False
    secret_values_exposed: bool = False
    error: str | None = None


def register_atlassian_tools(server: MCPServer, service: AtlassianAdminService) -> None:
    @server.tool()
    def atlassian_credential_status() -> CredentialStatusResult:
        return CredentialStatusResult(**service.credential_status())

    @server.tool()
    def atlassian_org_list(limit: int | None = None) -> OrganizationListResult:
        return OrganizationListResult(
            **service.list_organizations(limit=25 if limit is None else limit)
        )

    @server.tool()
    def atlassian_org_get(org_id: str | None = None) -> OrganizationGetResult:
        return OrganizationGetResult(**service.get_organization(org_id=org_id))

    @server.tool()
    def atlassian_workspace_list(
        org_id: str | None = None,
        name: str | None = None,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        return WorkspaceListResult(
            **service.list_workspaces(
                org_id=org_id,
                name=name,
                limit=25 if limit is None else limit,
            )
        )

    @server.tool()
    def atlassian_api_token_metadata(
        account_id: str,
        limit: int | None = None,
    ) -> TokenMetadataResult:
        return TokenMetadataResult(
            **service.list_api_token_metadata(
                account_id=account_id,
                limit=25 if limit is None else limit,
            )
        )
