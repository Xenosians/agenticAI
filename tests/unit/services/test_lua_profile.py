"""Real Lua declaration reads and host-owned configuration rejection cases."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from services.lua_adapters import profile
from services.lua_adapters.runtime import LuaAdapterError

PROFILE = """
return {
  links = {
    directory = {host = "dc.example.test", port = 636, tls = true, base_dn = "DC=example,DC=test",
                 account_container = "OU=Staff,DC=example,DC=test", email_domain = "example.test",
                 groups = {vpn = "CN=VPN Users,OU=Groups,DC=example,DC=test"},
                 read = {user = "svc-read", secret = "ldap_read"},
                 write = {user = "svc-write", secret = "ldap_write"}},
    router = {url = "https://192.0.2.1", user = "auditor", secret = "router", verify_tls = true},
    jira = {url = "https://example.atlassian.net", email = "bot@example.test", secret = "jira"},
    atlassian = {org_id = "org-1", jira_resource = "ari:cloud:jira::site/1", secret = "atlassian"},
    mailbox = {enabled = true, domain = "lab.test", smtp_port = 2587, imap_port = 2143},
    models = {hub = "hub-main",
              profiles = {["hub-main"] = {backend = "transformers", model_path = "/models/hub"}},
              allowed_origins = {"https://api.anthropic.com", "http://192.0.2.9:20128"},
              insecure_http_origins = {"http://192.0.2.9:20128"}},
  },
  adapt = function(event) return event end,
}
"""
SECRETS = {"ldap_read": "read-pass", "ldap_write": "write-pass", "router": "router-pass",
           "jira": "jira-token", "atlassian": "admin-key"}
CREDENTIAL_FIELDS = {"ad_bind_password", "ad_write_bind_password", "openwrt_password",
                     "jira_api_token", "atlassian_admin_api_key"}


@pytest.fixture
def install(tmp_path, monkeypatch):
    def install(source=PROFILE, secrets=SECRETS, mode=0o600):
        script = tmp_path / "machine.lua"
        script.write_text(source, encoding="utf-8")
        monkeypatch.setenv("ITSM_LUA_ADAPTER_FILE", str(script))
        monkeypatch.setenv("ITSM_LUA_ADAPTER_SHA256", hashlib.sha256(script.read_bytes()).hexdigest())
        if secrets is None:
            monkeypatch.delenv("ITSM_LINK_SECRETS_FILE", raising=False)
        else:
            store = tmp_path / "link-secrets.json"
            store.write_text(json.dumps(secrets), encoding="utf-8")
            store.chmod(mode)
            monkeypatch.setenv("ITSM_LINK_SECRETS_FILE", str(store))
    return install


def refused(call):
    with pytest.raises(LuaAdapterError) as caught:
        call()
    assert caught.value.code in {"lua_adapter_configuration_invalid", "lua_adapter_output_invalid"}


def test_the_machine_profile_becomes_the_settings_for_every_link(install):
    install()
    values = profile.settings_values()
    assert values["directory_backend"] == "ldap" and values["ticketing_backend"] == "jira"
    assert (values["ad_host"], values["ad_port"], values["ad_use_ssl"]) == ("dc.example.test", 636, True)
    assert values["ad_base_dn"] == "DC=example,DC=test"
    assert values["ad_account_container_dn"] == "OU=Staff,DC=example,DC=test"
    assert values["account_email_domain"] == "example.test"
    assert values["ad_access_groups"] == {"vpn": "CN=VPN Users,OU=Groups,DC=example,DC=test"}
    assert (values["ad_bind_user"], values["ad_bind_password"]) == ("svc-read", "read-pass")
    assert (values["ad_write_bind_user"], values["ad_write_bind_password"]) == ("svc-write", "write-pass")
    assert (values["openwrt_base_url"], values["openwrt_username"]) == ("https://192.0.2.1", "auditor")
    assert values["openwrt_password"] == "router-pass" and values["openwrt_verify_tls"] is True
    assert (values["jira_base_url"], values["jira_email"]) == ("https://example.atlassian.net", "bot@example.test")
    assert values["jira_api_token"] == "jira-token" and values["atlassian_admin_api_key"] == "admin-key"
    assert (values["atlassian_org_id"], values["atlassian_jira_resource_ari"]) == ("org-1", "ari:cloud:jira::site/1")
    assert values["mailbox_lab_enabled"] is True and values["mailbox_lab_domain"] == "lab.test"
    assert (values["mailbox_smtp_port"], values["mailbox_imap_port"]) == (2587, 2143)
    assert values["hub_model_key"] == "hub-main"
    assert values["model_profiles"] == {"hub-main": {"backend": "transformers", "model_path": "/models/hub"}}


def test_every_profile_value_is_a_real_settings_field(install):
    from config.settings import Settings

    install()
    # Settings ignores unknown names, so a mapping typo would silently drop a link.
    assert set(profile.settings_values()) <= set(Settings.model_fields)


def test_settings_take_their_links_from_the_profile_over_the_environment(install, monkeypatch):
    from config.settings import Settings

    install()
    monkeypatch.setenv("AD_HOST", "stale-host.example.test")
    monkeypatch.setenv("JIRA_API_TOKEN", "stale-token")
    monkeypatch.setenv("MODEL_PROFILES", json.dumps({"hub-main": {"backend": "stale", "adapter_path": "/stale"},
                                                     "stale-extra": {"backend": "stale"}}))
    settings = Settings(_env_file=None)
    assert settings.ad_host == "dc.example.test"
    assert settings.jira_api_token == "jira-token"
    # A declared mapping replaces the environment's; the two are never merged.
    assert set(settings.model_profiles) == {"hub-main"}
    assert settings.model_profiles["hub-main"].model_path.as_posix() == "/models/hub"
    assert settings.model_profiles["hub-main"].adapter_path is None
    # Explicit construction arguments still win, as before.
    assert Settings(_env_file=None, ad_host="explicit.example.test").ad_host == "explicit.example.test"


def test_the_profile_declares_the_model_endpoint_policy_for_the_process(install, monkeypatch, tmp_path):
    from config.settings import export_process_policy

    install()
    monkeypatch.setenv("EXTENSION_ALLOWED_ORIGINS", "https://stale.example.test")
    monkeypatch.setenv("EXTENSION_INSECURE_HTTP_ORIGINS", "http://stale.example.test")
    exported = export_process_policy(tmp_path / "absent.env")
    assert sorted(exported) == ["EXTENSION_ALLOWED_ORIGINS", "EXTENSION_INSECURE_HTTP_ORIGINS"]
    assert os.environ["EXTENSION_ALLOWED_ORIGINS"] == "https://api.anthropic.com,http://192.0.2.9:20128"
    assert os.environ["EXTENSION_INSECURE_HTTP_ORIGINS"] == "http://192.0.2.9:20128"


def test_plain_http_is_allowed_only_for_a_listed_origin(install):
    install(source=PROFILE.replace('insecure_http_origins = {"http://192.0.2.9:20128"}',
                                   'insecure_http_origins = {"http://198.51.100.4"}'))
    refused(profile.process_policy)


def test_only_secret_names_cross_into_lua(install):
    install()
    declared = profile.load_links().declared
    rendered = json.dumps(declared)
    for value in SECRETS.values():
        assert value not in rendered
    assert declared["directory"]["read"] == {"user": "svc-read", "secret": "ldap_read"}


def test_a_credential_written_into_the_profile_is_refused(install):
    install(source=PROFILE.replace('secret = "router"', 'password = "router-pass"'))
    refused(profile.settings_values)


def test_a_missing_named_secret_refuses_the_profile_instead_of_connecting_without_it(install):
    install(secrets={key: value for key, value in SECRETS.items() if key != "jira"})
    refused(profile.settings_values)


def test_a_process_given_no_secrets_file_gets_the_links_without_credentials(install):
    install(secrets=None)  # e.g. the trainer, which never opens these links
    values = profile.settings_values()
    assert values["ad_host"] == "dc.example.test" and values["ad_bind_user"] == "svc-read"
    # Explicit empty values prevent a stale environment credential from being
    # merged into a section that this credential-free process declared.
    assert all(values[field] is None for field in CREDENTIAL_FIELDS)


@pytest.mark.parametrize("mode", [0o644, 0o622])
def test_a_secrets_file_others_can_read_or_change_is_refused(install, mode):
    install(mode=mode)
    refused(profile.settings_values)


def test_a_profile_that_is_not_plain_data_is_refused(install):
    install(source='return {links = {router = {url = function() return "x" end}}}')
    refused(profile.settings_values)


def test_a_misspelled_profile_key_is_refused_instead_of_declaring_nothing(install):
    install(source='return {link = {router = {url = "https://192.0.2.1"}}}')
    with pytest.raises(LuaAdapterError) as caught:
        profile.settings_values()
    assert caught.value.code == "lua_adapter_contract_invalid"


@pytest.mark.parametrize("entry", ["port = 0", "port = 70000", 'host = "has space"', 'tls = "yes"',
                                   'hostname = "dc.example.test"', 'read = {user = "a", secret = "Bad Name"}'])
def test_malformed_or_unknown_link_values_are_refused(install, entry):
    install(source=f'return {{links = {{directory = {{{entry}}}}}}}')
    refused(profile.settings_values)


def test_an_unknown_link_section_is_refused(install):
    install(source='return {links = {printer = {host = "p.example.test"}}}')
    refused(profile.settings_values)


def test_an_unknown_model_profile_field_is_refused(install):
    install(source='return {links = {models = {profiles = {hub = {backend = "x", model_pth = "/m"}}}}}')
    refused(profile.settings_values)


@pytest.mark.parametrize("origin", ["ftp://198.51.100.1", "https://user@198.51.100.1", "https://198.51.100.1/path"])
def test_model_origins_must_be_exact_web_origins(install, origin):
    install(source=f'return {{links = {{models = {{allowed_origins = {{"{origin}"}}}}}}}}')
    refused(profile.process_policy)


def test_an_event_only_script_declares_no_links(install):
    from services.lua_adapters.host_boundaries import inspect_data

    install(source="return function(event) return event end")
    assert profile.settings_values() == {} and profile.process_policy() == {}
    inspect_data("connection", "request", {"origin": "https://x.example"}, {"transport": "https"})


def test_a_profile_without_adapt_passes_events_through(install):
    from services.lua_adapters.host_boundaries import inspect_data

    install(source='return {links = {jira = {url = "https://example.atlassian.net"}}}')
    inspect_data("connection", "request", {"origin": "https://x.example"}, {"transport": "https"})
    assert profile.settings_values()["jira_base_url"] == "https://example.atlassian.net"


def test_the_adapt_function_of_a_profile_runs_at_every_boundary(install):
    from services.lua_adapters.host_boundaries import inspect_data

    install(source="""
return {links = {}, adapt = function(e)
  if e.kind == 'connection' and e.payload.origin == 'https://blocked.example' then error('refused') end
  return e
end}
""")
    inspect_data("connection", "request", {"origin": "https://allowed.example"}, {"transport": "https"})
    with pytest.raises(LuaAdapterError):
        inspect_data("connection", "request", {"origin": "https://blocked.example"}, {"transport": "https"})


def test_without_a_profile_no_links_are_declared(monkeypatch):
    for name in ("ITSM_LUA_ADAPTER_FILE", "ITSM_LUA_ADAPTER_SHA256", "ITSM_LINK_SECRETS_FILE"):
        monkeypatch.delenv(name, raising=False)
    assert profile.settings_values() == {} and profile.process_policy() == {}


def test_declared_partial_sections_never_inherit_environment_credentials(install, monkeypatch):
    from config.settings import Settings
    install(source='return {links = {jira = {url = "https://new.example.test"}, directory = {host = "dc.example.test"}}}', secrets=None)
    for name in ("JIRA_EMAIL", "JIRA_API_TOKEN", "AD_BIND_DN", "AD_BIND_PASSWORD", "AD_WRITE_BIND_USER", "AD_WRITE_BIND_PASSWORD"):
        monkeypatch.setenv(name, "stale-value")
    monkeypatch.setenv("AD_ACCESS_GROUPS", "not-json")
    settings = Settings(_env_file=None)
    assert settings.jira_base_url == "https://new.example.test"
    assert settings.jira_email is settings.jira_api_token is None
    assert settings.ad_bind_dn is settings.ad_bind_password is settings.ad_write_bind_user is settings.ad_write_bind_password is None
    assert settings.ad_access_groups == {}


def test_declared_section_does_not_read_dotenv_or_secret_file_fallbacks(install, tmp_path):
    from config.settings import Settings
    install(source='return {links = {jira = {}, models = {}, directory = {}}}', secrets=None)
    dotenv = tmp_path / "old.env"
    dotenv.write_text('JIRA_EMAIL=stale@example.test\nJIRA_API_TOKEN=stale-token\nMODEL_PROFILES=not-json\nAD_ACCESS_GROUPS=not-json\n')
    secrets = tmp_path / "old-secrets"
    secrets.mkdir()
    (secrets / "jira_api_token").write_text("another-stale-token")
    settings = Settings(_env_file=dotenv, _secrets_dir=secrets)
    assert settings.jira_email is settings.jira_api_token is None
    assert settings.model_profiles == {} and settings.ad_access_groups == {}
    assert settings.directory_backend == settings.ticketing_backend == "unconfigured"


def test_explicit_mapping_replaces_the_profile_mapping_without_merging(install):
    from config.settings import Settings
    install()
    settings = Settings(_env_file=None, model_profiles={})
    assert settings.model_profiles == {}


def test_declared_models_clear_undeclared_origin_lists(install, monkeypatch, tmp_path):
    from config.settings import export_process_policy
    install(source='return {links = {models = {profiles = {}}}}')
    monkeypatch.setenv("EXTENSION_ALLOWED_ORIGINS", "https://stale.example")
    monkeypatch.setenv("EXTENSION_INSECURE_HTTP_ORIGINS", "http://stale.example")
    old = tmp_path / "old.env"
    old.write_text("EXTENSION_ALLOWED_ORIGINS=https://another-stale.example\n")
    assert set(export_process_policy(old)) == {"EXTENSION_ALLOWED_ORIGINS", "EXTENSION_INSECURE_HTTP_ORIGINS"}
    assert os.environ["EXTENSION_ALLOWED_ORIGINS"] == os.environ["EXTENSION_INSECURE_HTTP_ORIGINS"] == ""


def test_profile_bytes_are_reverified_even_when_file_metadata_is_restored(install):
    install()
    profile.load_links()
    path = Path(os.environ["ITSM_LUA_ADAPTER_FILE"])
    before = path.stat()
    path.write_text(PROFILE.replace("dc.example.test", "dc.changed.test"))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    refused(profile.load_links)


def test_cached_profile_snapshots_cannot_be_mutated_by_a_caller(install):
    install()
    first = profile.load_links()
    first.declared["jira"]["url"] = "https://changed.example.test"
    first.secrets["jira"] = "changed-secret"
    current = profile.load_links()
    assert current.declared["jira"]["url"] == "https://example.atlassian.net"
    assert current.secrets["jira"] == "jira-token"
    assert "jira-token" not in repr(current)


def test_credential_rotation_invalidates_cached_settings_even_with_same_metadata(install):
    from config.settings import get_settings
    install()
    get_settings.cache_clear()
    try:
        first = get_settings()
        assert first.jira_api_token == "jira-token"
        path = Path(os.environ["ITSM_LINK_SECRETS_FILE"])
        before = path.stat()
        path.write_text(json.dumps({**SECRETS, "jira": "jira-other"}))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert get_settings().jira_api_token == "jira-other"
    finally:
        get_settings.cache_clear()


def test_secret_file_permissions_are_rechecked_after_the_first_load(install):
    install()
    profile.settings_values()
    Path(os.environ["ITSM_LINK_SECRETS_FILE"]).chmod(0o644)
    refused(profile.settings_values)


def test_duplicate_secret_names_are_refused(install):
    install()
    Path(os.environ["ITSM_LINK_SECRETS_FILE"]).write_text('{"jira":"first","jira":"second"}')
    refused(profile.settings_values)


def test_symlinked_secret_files_are_refused(install, tmp_path, monkeypatch):
    install()
    link = tmp_path / "link.json"
    link.symlink_to(Path(os.environ["ITSM_LINK_SECRETS_FILE"]))
    monkeypatch.setenv("ITSM_LINK_SECRETS_FILE", str(link))
    refused(profile.settings_values)


def test_profile_relative_model_paths_follow_the_profile_not_the_checkout(install, tmp_path):
    install(source='return {links = {models = {profiles = {hub = {backend = "hf-causal", model_path = "models/hub"}}}}}')
    assert profile.settings_values()["model_profiles"]["hub"]["model_path"] == str(tmp_path / "models/hub")


@pytest.mark.parametrize("origin", ["https://example.test:0", "https://@example.test", "https://example.test?", "https://example.test#", "https://[broken"])
def test_malformed_origins_are_curated_configuration_errors(install, origin):
    install(source=f'return {{links = {{models = {{allowed_origins = {{"{origin}"}}}}}}}}')
    refused(profile.process_policy)


def test_enabling_mailboxes_requires_an_explicit_domain(install):
    install(source='return {links = {mailbox = {enabled = true}}}')
    refused(profile.settings_values)


def plugin_profile(install, tmp_path, relative="plugins/custom.lua", change=None):
    path = tmp_path / "plugins/custom.lua"
    path.parent.mkdir(exist_ok=True)
    path.write_text("return function(event) event.payload.profile_tag = 'mapped'; return event end")
    entry = {"file": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **(change or {})}
    source = 'return {plugins = {["model.openai"] = {file = ' + json.dumps(entry["file"]) + ', sha256 = ' + json.dumps(entry["sha256"]) + '}}}'
    install(source=source, secrets=None)
    return path


def test_a_profile_resolves_its_reviewed_lua_plugin_file(install, tmp_path):
    path = plugin_profile(install, tmp_path)
    assert profile.plugin_artifact("model.openai") == {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert profile.plugin_artifact("model.local") is None


def test_plugin_digest_changes_are_not_hidden_by_the_profile_cache(install, tmp_path):
    path = plugin_profile(install, tmp_path)
    profile.plugin_artifact("model.openai")
    path.write_text("return function(event) error('changed') end")
    refused(lambda: profile.plugin_artifact("model.openai"))


@pytest.mark.parametrize("relative", ["../custom.lua", "/tmp/outside-custom.lua", "plugins/custom.py"])
def test_plugin_files_cannot_escape_the_profile_or_change_language(install, tmp_path, relative):
    plugin_profile(install, tmp_path, relative=relative)
    refused(lambda: profile.plugin_artifact("model.openai"))


def test_plugin_parent_symlinks_are_refused(install, tmp_path):
    path = plugin_profile(install, tmp_path, relative="alias/custom.lua")
    (tmp_path / "alias").symlink_to(path.parent, target_is_directory=True)
    refused(lambda: profile.plugin_artifact("model.openai"))


def test_unknown_plugin_ids_or_ignored_fields_are_refused(install, tmp_path):
    install(source='return {plugins = {unknown = {file = "x.lua", sha256 = "' + 'a' * 64 + '"}}}')
    refused(profile.load_links)
    install(source='return {plugins = {["model.local"] = {file = "x.lua", sha256 = "' + 'a' * 64 + '", surprise = true}}}')
    refused(profile.load_links)
