"""A machine profile selects allowed destinations; presets cannot override it."""
import hashlib

import pytest

from services.extensions.transport import validate_url


@pytest.fixture
def machine(tmp_path, monkeypatch):
    def configure(origins):
        script = tmp_path / "machine.lua"
        text = 'return {links={models={allowed_origins={' + ','.join('"' + url + '"' for url in origins) + '}}}}'
        script.write_text(text)
        monkeypatch.setenv("ITSM_LUA_ADAPTER_FILE", str(script))
        monkeypatch.setenv("ITSM_LUA_ADAPTER_SHA256", hashlib.sha256(script.read_bytes()).hexdigest())
        monkeypatch.delenv("ITSM_LINK_SECRETS_FILE", raising=False)
        monkeypatch.setenv("EXTENSION_ALLOWED_ORIGINS", "https://stale-env.example")
    yield configure


def test_declared_origins_do_not_inherit_vendor_presets_or_old_environment(machine):
    machine(["https://chosen.example"])
    assert validate_url("https://chosen.example/v1") == "https://chosen.example/v1"
    for value in ("https://api.groq.com/openai/v1", "https://api.anthropic.com/v1", "https://stale-env.example/v1"):
        with pytest.raises(ValueError, match="origin is not"):
            validate_url(value)


def test_explicit_empty_origins_selects_no_outbound_endpoint(machine):
    machine([])
    with pytest.raises(ValueError, match="origin is not"):
        validate_url("https://api.groq.com/openai/v1")
