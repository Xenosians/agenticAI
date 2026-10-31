"""The shared privacy/secret scanner redacts credentials, not ordinary words."""
import base64

import pytest

from learning.evidence.sanitizer import sanitize_text


def basic_header(credential: str) -> str:
    return "Basic " + base64.b64encode(credential.encode("utf-8")).decode("ascii")


@pytest.mark.parametrize("text", [
    "Basic setup",
    "Follow the basic wireless configuration first.",
    "Basic Service Set (BSS)",
    "basic 802.11 settings",
    "A basic relay configuration bridges two networks.",
    "Basic internationalization is covered later.",
])
def test_ordinary_words_after_basic_are_not_credentials(text):
    assert sanitize_text(text) == text


def test_a_long_opaque_token_after_basic_is_redacted():
    # 25 characters cannot be base64-decoded, so only its shape marks it as a key.
    token = "k8sZ0pQ2vX9rT4mW7yB1nC3dQ"
    assert token not in sanitize_text(f"Set the upstream header to Basic {token} before syncing.")


@pytest.mark.parametrize("credential", ["user:password", "admin:secret", "u:p", "jos\u00e9:p\u00e4ssw\u00f6rd"])
def test_http_basic_credentials_in_text_are_redacted(credential):
    header = basic_header(credential)
    assert sanitize_text(f"The proxy sends {header} upstream.") == "The proxy sends Basic <redacted> upstream."


def test_an_unpadded_basic_credential_is_redacted():
    token = base64.b64encode(b"admin:hunter22").decode("ascii").rstrip("=")
    assert token not in sanitize_text(f"curl --header 'X-Upstream: Basic {token}'")


@pytest.mark.parametrize("token", ["YWJj", "c2V0dXA=", "dXNlcjpwYXNzd29yZA=="])
def test_whatever_follows_basic_in_an_authorization_header_is_redacted(token):
    assert sanitize_text(f"Proxy-Authorization: Basic {token}") == "Proxy-Authorization: Basic <redacted>"


@pytest.mark.parametrize("text, secret", [
    ('Cleartext-Password := "hello world"', "hello"),
    ("Cleartext-Password := hunter2", "hunter2"),
    ('password: "correct horse battery staple"', "horse"),
    ("client_secret='s3cr3t tail-part'", "tail-part"),
    ('password="hunter2', "hunter2"),
])
def test_an_assigned_secret_is_redacted_whole(text, secret):
    assert secret not in sanitize_text(text)


@pytest.mark.parametrize("text", ['password=""', "password=''", "password=\u201c\u201d", 'api_key: ""'])
def test_an_empty_value_is_not_a_secret(text):
    assert sanitize_text(text) == text


def test_a_redacted_assignment_keeps_its_name():
    assert sanitize_text("token=abcdef123") == "token=<redacted>"
