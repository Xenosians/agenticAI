import pytest
from pydantic import ValidationError

from config import Settings


def test_account_creation_settings_normalize_domain_and_container():
    settings = Settings(
        account_email_domain="Corp.Example.",
        ad_account_container_dn=" OU=People,DC=corp,DC=example ",
        account_temporary_password_length=32,
    )
    assert settings.account_email_domain == "corp.example"
    assert settings.ad_account_container_dn == "OU=People,DC=corp,DC=example"
    assert settings.account_temporary_password_length == 32


def test_account_creation_settings_reject_short_password_policy():
    with pytest.raises(ValidationError):
        Settings(account_temporary_password_length=12)
