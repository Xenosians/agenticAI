"""Per-account database credentials (owner decisions 18 Nov item 2, 23 Nov item 2).

The superuser password of a database guest is generated when the approved create runs and stored only
here, encrypted and scoped to the account; listing never returns it.
"""
import pytest

from services.extensions.database_credentials import DatabaseCredentialError, DatabaseCredentials
from services.extensions.store import ExtensionStore

PASSWORD = "Xq7-generated_by_janus_9f2Kp0aZ"


@pytest.fixture
def credentials(tmp_path):
    return DatabaseCredentials(ExtensionStore(tmp_path))


def test_a_login_is_kept_encrypted_per_account_and_listed_without_its_secret(credentials, tmp_path):
    saved = credentials.save("owner-a", node="hummingbird", vmid=120, name="db-pg-1", engine="postgresql",
                             address="10.77.0.20", port=5432, username="postgres", password=PASSWORD)
    assert saved["credential_configured"] is True and "password" not in saved
    assert credentials.get("owner-a", "hummingbird", 120)["password"] == PASSWORD
    assert credentials.list("owner-a") == [saved]
    # Another account sees nothing of it, and the database file never holds the password in clear.
    assert credentials.get("owner-b", "hummingbird", 120) is None and credentials.list("owner-b") == []
    assert PASSWORD.encode() not in (tmp_path / "extensions.sqlite3").read_bytes()


def test_a_login_is_removed_only_for_its_own_account(credentials):
    credentials.save("owner-a", node="hummingbird", vmid=121, name="db-m-1", engine="mariadb",
                     address="10.77.0.21", port=3306, username="root", password=PASSWORD)
    assert credentials.remove("owner-b", "hummingbird", 121) is False
    assert credentials.remove("owner-a", "hummingbird", 121) is True
    assert credentials.get("owner-a", "hummingbird", 121) is None


@pytest.mark.parametrize("change", [{"engine": "mongodb"}, {"password": "short"}, {"port": 0}, {"name": "Bad Name"}])
def test_an_invalid_login_is_refused_without_repeating_it(credentials, change):
    values = dict(node="hummingbird", vmid=120, name="db-pg-1", engine="postgresql", address="10.77.0.20",
                  port=5432, username="postgres", password=PASSWORD)
    with pytest.raises(DatabaseCredentialError) as refused:
        credentials.save("owner-a", **{**values, **change})
    assert PASSWORD not in str(refused.value)
