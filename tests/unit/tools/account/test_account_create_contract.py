from services.directory.account_creation import AccountCreateResult
from tools.account.catalog import ACCOUNT_TOOLS
from tools.account.mcp import register_account_lifecycle_tools


class FakeLifecycle:
    def enable_user(self, user_id):
        raise AssertionError("not used")

    def disable_user(self, user_id):
        raise AssertionError("not used")


class FakeCreation:
    def create_account(self, *args, **kwargs):
        return AccountCreateResult(
            ok=True,
            status="executed",
            changed=True,
            user_id=kwargs["expected_username"],
            email=kwargs["expected_email"],
            account_created=True,
            verification_ok=True,
            credential_persisted=True,
            credential_id="cred-test",
            account_record_id="acct-test",
            credential_expires_at="2026-09-24T00:00:00Z",
            retry_safe=False,
        )


class FakeServer:
    def __init__(self):
        self.functions = {}

    def tool(self):
        def decorator(fn):
            self.functions[fn.__name__] = fn
            return fn
        return decorator


def test_account_create_catalog_is_high_risk_and_policy_bound():
    tool = ACCOUNT_TOOLS["account_create"]
    assert tool["risk"] == "high"
    assert tool["requires_approval"] is True
    assert "expected_username" in tool["trusted_policy_arguments"]
    assert "expected_email" in tool["trusted_policy_arguments"]
    assert "temporary_password" not in tool["parameters"]
    assert "temporary_password" not in tool["trusted_policy_arguments"]


def test_account_create_mcp_never_returns_plaintext_password():
    server = FakeServer()
    register_account_lifecycle_tools(server, FakeLifecycle(), FakeCreation())

    result = server.functions["account_create"](
        given_name="Grace",
        family_name="Hopper",
        expected_username="ghopper",
        expected_email="ghopper@corp.example",
    )

    payload = result.model_dump()
    assert payload["credential_id"] == "cred-test"
    assert "password" not in payload
    assert "temporary_password" not in payload
