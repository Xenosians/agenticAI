from dataclasses import dataclass

from config import Settings
from services.directory.account_creation import (
    IDENTITY_POLICY_VERSION,
    MockAccountCreationService,
    generate_temporary_password,
    prepare_account_create,
)
from services.directory.mock import MockDirectoryService
from services.credentials import ProvisionedAccountPersistenceResult


@dataclass
class RecordingCredentialStore:
    password: str | None = None
    email: str | None = None
    user_id: str | None = None

    def persist_provisioned_account(self, **kwargs):
        self.password = kwargs["temporary_password"]
        self.email = kwargs["email"]
        self.user_id = kwargs["directory_user_id"]
        return ProvisionedAccountPersistenceResult(
            account_record_id="acct-test",
            credential_id="cred-test",
            credential_expires_at="2026-09-24T00:00:00Z",
        )


def _settings() -> Settings:
    return Settings(
        directory_backend="mock",
        account_email_domain="corp.example",
        account_temporary_password_length=24,
    )


def test_prepare_derives_identity_and_resolves_collision():
    directory = MockDirectoryService()
    # asmith already exists in the seeded mock directory.
    prepared = prepare_account_create(
        directory,
        _settings(),
        given_name="Alice",
        family_name="Smith",
        department="Engineering",
        role="Backend Engineer",
    )

    assert prepared["ok"] is True
    args = prepared["execution_arguments"]
    assert args["expected_username"] == "asmith2"
    assert args["expected_email"] == "asmith2@corp.example"
    assert args["identity_policy_version"] == IDENTITY_POLICY_VERSION
    assert "temporary_password" not in args


def test_mock_create_generates_secret_after_policy_and_never_returns_it():
    directory = MockDirectoryService()
    store = RecordingCredentialStore()
    settings = _settings()

    prepared = prepare_account_create(
        directory,
        settings,
        given_name="Grace",
        family_name="Hopper",
        department="Engineering",
        role="Engineer",
    )
    args = prepared["execution_arguments"]

    service = MockAccountCreationService(directory, store, settings)
    result = service.create_account(
        "Grace",
        "Hopper",
        department="Engineering",
        role="Engineer",
        expected_username=args["expected_username"],
        expected_email=args["expected_email"],
        expected_container_dn=args.get("expected_container_dn"),
        identity_policy_version=args["identity_policy_version"],
    )

    assert result.ok is True
    assert result.user_id == "ghopper"
    assert result.email == "ghopper@corp.example"
    assert result.credential_id == "cred-test"
    assert result.credential_persisted is True
    assert result.verification_ok is True

    dumped = result.model_dump()
    assert "password" not in dumped
    assert "temporary_password" not in dumped

    assert store.password is not None
    assert len(store.password) == 24
    assert store.password not in repr(dumped)


def test_password_generator_has_required_classes():
    value = generate_temporary_password(24)
    assert len(value) == 24
    assert any(c.isupper() for c in value)
    assert any(c.islower() for c in value)
    assert any(c.isdigit() for c in value)
    assert any(not c.isalnum() for c in value)


def test_revalidation_blocks_approved_username_if_it_becomes_taken():
    directory = MockDirectoryService()
    store = RecordingCredentialStore()
    settings = _settings()

    prepared = prepare_account_create(
        directory,
        settings,
        given_name="Grace",
        family_name="Hopper",
    )
    args = prepared["execution_arguments"]

    directory.users[args["expected_username"]] = {
        "enabled": True,
        "locked": False,
        "password_reset_count": 0,
    }

    service = MockAccountCreationService(directory, store, settings)
    result = service.create_account(
        "Grace",
        "Hopper",
        expected_username=args["expected_username"],
        expected_email=args["expected_email"],
        expected_container_dn=args.get("expected_container_dn"),
        identity_policy_version=args["identity_policy_version"],
    )

    assert result.ok is False
    assert result.status == "conflict"
    assert result.account_created is False
    assert result.retry_safe is True
    assert store.password is None


def test_revalidation_blocks_identity_if_email_policy_changes_after_approval():
    directory = MockDirectoryService()
    store = RecordingCredentialStore()
    settings = _settings()

    prepared = prepare_account_create(
        directory,
        settings,
        given_name="Grace",
        family_name="Hopper",
    )
    args = prepared["execution_arguments"]

    changed_settings = Settings(
        directory_backend="mock",
        account_email_domain="new.corp.example",
        account_temporary_password_length=24,
    )
    service = MockAccountCreationService(directory, store, changed_settings)
    result = service.create_account(
        "Grace",
        "Hopper",
        expected_username=args["expected_username"],
        expected_email=args["expected_email"],
        expected_container_dn=args.get("expected_container_dn"),
        identity_policy_version=args["identity_policy_version"],
    )

    assert result.ok is False
    assert result.status == "denied"
    assert result.account_created is None
    assert store.password is None
