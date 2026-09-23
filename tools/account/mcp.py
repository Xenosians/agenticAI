from pydantic import BaseModel
from mcp.server import MCPServer

from services.directory import (
    AccountCreationService,
    AccountLifecycleService,
)


class AccountLifecycleMCPResult(BaseModel):
    ok: bool
    status: str
    changed: bool = False
    user_id: str | None = None
    enabled: bool | None = None
    message: str | None = None
    error: str | None = None


class AccountCreateMCPResult(BaseModel):
    ok: bool
    status: str
    changed: bool = False
    operation: str = "account_create"
    user_id: str | None = None
    email: str | None = None
    account_created: bool | None = None
    verification_ok: bool = False
    credential_persisted: bool = False
    account_record_id: str | None = None
    credential_id: str | None = None
    credential_expires_at: str | None = None
    retry_safe: bool = True
    message: str | None = None
    error: str | None = None


def register_account_lifecycle_tools(
    server: MCPServer,
    lifecycle: AccountLifecycleService,
    creation: AccountCreationService | None = None,
) -> None:
    @server.tool()
    def enable_user(user_id: str) -> AccountLifecycleMCPResult:
        result = lifecycle.enable_user(user_id)
        return AccountLifecycleMCPResult(**result.model_dump())

    @server.tool()
    def disable_user(user_id: str) -> AccountLifecycleMCPResult:
        result = lifecycle.disable_user(user_id)
        return AccountLifecycleMCPResult(**result.model_dump())

    if creation is not None:
        @server.tool()
        def account_create(
            given_name: str,
            family_name: str,
            department: str | None = None,
            role: str | None = None,
            expected_username: str = "",
            expected_email: str = "",
            expected_container_dn: str | None = None,
            identity_policy_version: str = "corporate-account-identity.v1",
        ) -> AccountCreateMCPResult:
            result = creation.create_account(
                given_name,
                family_name,
                department=department,
                role=role,
                expected_username=expected_username,
                expected_email=expected_email,
                expected_container_dn=expected_container_dn,
                identity_policy_version=identity_policy_version,
            )
            return AccountCreateMCPResult(**result.model_dump())
