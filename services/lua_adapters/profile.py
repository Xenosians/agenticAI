"""Read reviewed machine links and codec references without giving Lua secrets.

The pinned machine profile can declare six link families, seven named plugin
artifacts and an optional generic event mapper. The isolated worker evaluates
the declarations; the host resolves protected credential references afterwards.
Declared Settings sections replace stale deployment values rather than merging
them. See ``docs/lua-machine-profiles.md`` for the supported contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import copy
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
from urllib.parse import urlsplit

from services.lua_adapters.runtime import MAX_SOURCE_BYTES, LuaAdapterError, run_adapter

MAX_SECRETS_BYTES = 64_000
MAX_ENTRIES = 64
SECRET_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
HOST = re.compile(rf"(?=.{{1,253}}\Z){LABEL}(?:\.{LABEL})*")
KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
TEXT = re.compile(r"[^\x00-\x1f\x7f]{1,512}")
SECTIONS = {
    "directory": {"host", "port", "tls", "base_dn", "account_container", "email_domain", "groups",
                  "read", "write"},
    "router": {"url", "user", "secret", "verify_tls", "allow_loopback_http"},
    "jira": {"url", "email", "secret"},
    "atlassian": {"org_id", "jira_resource", "secret"},
    "mailbox": {"enabled", "domain", "smtp_port", "imap_port"},
    "models": {"hub", "profiles", "allowed_origins", "insecure_http_origins"},
    "sources": {"allowed_origins", "insecure_http_origins"},
}
PLUGIN_IDS = frozenset({"model.local", "model.openai", "model.anthropic", "model.harness",
                        "connection.default", "source.default", "tool.default"})
CONTROLLED_FIELDS = {
    "directory": {"directory_backend", "ad_host", "ad_port", "ad_use_ssl", "ad_base_dn",
                  "ad_account_container_dn", "account_email_domain", "ad_access_groups", "ad_bind_user",
                  "ad_bind_dn", "ad_bind_password", "ad_write_bind_user", "ad_write_bind_dn", "ad_write_bind_password"},
    "router": {"openwrt_base_url", "openwrt_username", "openwrt_password", "openwrt_verify_tls", "openwrt_allow_loopback_http"},
    "jira": {"ticketing_backend", "jira_base_url", "jira_email", "jira_api_token"},
    "atlassian": {"atlassian_org_id", "atlassian_jira_resource_ari", "atlassian_admin_api_key"},
    "mailbox": {"mailbox_lab_enabled", "mailbox_lab_domain", "mailbox_smtp_port", "mailbox_imap_port"},
    "models": {"hub_model_key", "model_profiles"},
    "sources": set(),  # Read by the knowledge crawler through source_policy(), never Settings.
}


@dataclass(frozen=True)
class Links:
    declared: dict          # As written in Lua: secret names, never values.
    secrets: dict | None = field(repr=False)  # Never included in diagnostic object representations.
    plugins: dict = field(default_factory=dict)
    profile_path: Path | None = None


def _invalid():
    return LuaAdapterError("lua_adapter_configuration_invalid")


def _read_secrets() -> dict | None:
    filename = os.environ.get("ITSM_LINK_SECRETS_FILE", "")
    if not filename:
        return None
    path = Path(filename)
    try:
        if not path.is_absolute():
            raise _invalid()
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_size > MAX_SECRETS_BYTES):
                raise _invalid()
            raw = handle.read(MAX_SECRETS_BYTES + 1)
            if len(raw) > MAX_SECRETS_BYTES:
                raise _invalid()
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate secret reference")
                result[key] = value
            return result
        values = json.loads(raw, object_pairs_hook=unique)
    except (OSError, ValueError, UnicodeError):
        raise _invalid() from None
    if type(values) is not dict or len(values) > MAX_ENTRIES or not all(
            type(name) is str and SECRET_NAME.fullmatch(name) and type(value) is str and 0 < len(value) <= 16_384
            for name, value in values.items()):
        raise _invalid()
    return values


@lru_cache(maxsize=4)
def _read_declared(source: str) -> dict:
    result = run_adapter(kind="source", phase="admission", payload={}, binding={"profile": "links"},
                         source=source, read_profile=True)
    if (type(result.payload) is not dict or set(result.payload) != {"links", "plugins"}
            or type(result.payload["links"]) is not dict or type(result.payload["plugins"]) is not dict):
        raise _invalid()
    return result.payload


def load_links() -> Links:
    """Verify pinned bytes each time; cache Lua evaluation, never credential reads."""
    from services.lua_adapters.host_boundaries import configured_source
    source = configured_source()
    if source is None:
        return Links({}, None)
    snapshot = copy.deepcopy(_read_declared(source))
    declared, plugins = snapshot["links"], snapshot["plugins"]
    if set(declared) - set(SECTIONS) or set(plugins) - PLUGIN_IDS:
        raise _invalid()
    for artifact in plugins.values():
        if (type(artifact) is not dict or set(artifact) != {"file", "sha256"}
                or type(artifact["file"]) is not str or not artifact["file"].strip()
                or len(artifact["file"]) > 512 or any(ord(c) < 32 for c in artifact["file"])
                or type(artifact["sha256"]) is not str or not re.fullmatch(r"[a-f0-9]{64}", artifact["sha256"])):
            raise _invalid()
    return Links(declared, _read_secrets() if declared else None, plugins,
                 Path(os.environ["ITSM_LUA_ADAPTER_FILE"]))


def plugin_artifact(plugin_id: str) -> dict | None:
    """Resolve one reviewed Lua file inside the machine profile's directory tree."""
    if plugin_id not in PLUGIN_IDS:
        raise _invalid()
    links = load_links()
    artifact = links.plugins.get(plugin_id)
    if artifact is None:
        return None
    if links.profile_path is None:
        raise _invalid()
    base = links.profile_path.parent
    candidate = Path(artifact["file"])
    if ".." in candidate.parts or candidate.suffix != ".lua":
        raise _invalid()
    candidate = candidate if candidate.is_absolute() else base / candidate
    try:
        cursor = Path(candidate.anchor)
        for part in candidate.parts[1:]:
            cursor /= part
            if cursor.is_symlink():
                raise _invalid()
        resolved = candidate.resolve()
        if not resolved.is_relative_to(base.resolve()):
            raise _invalid()
        descriptor = os.open(candidate, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise _invalid()
            content = handle.read(MAX_SOURCE_BYTES + 1)
        if len(content) > MAX_SOURCE_BYTES or hashlib.sha256(content).hexdigest() != artifact["sha256"]:
            raise _invalid()
    except OSError:
        raise _invalid() from None
    return {"file": str(resolved), "sha256": artifact["sha256"]}


def configuration_identity() -> str | None:
    """Cache identity rechecks the pinned source and protected credentials."""
    links = load_links()
    if links.profile_path is None:
        return None
    snapshot = {"profile": str(links.profile_path), "source_sha256": os.environ.get("ITSM_LUA_ADAPTER_SHA256"), "links": links.declared,
                "plugins": links.plugins, "credentials": links.secrets}
    return hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _section(declared: dict, name: str) -> dict:
    """One declared section; a key the host does not read is refused, never ignored."""
    value = declared.get(name, {})
    if type(value) is not dict or set(value) - SECTIONS[name]:
        raise _invalid()
    return value


def _text(value, pattern=TEXT):
    if type(value) is not str or not value.strip() or not pattern.fullmatch(value):
        raise _invalid()
    return value


def _host(value):
    text = _text(value)
    try:
        ipaddress.ip_address(text)
    except ValueError:
        if not HOST.fullmatch(text):
            raise _invalid() from None
    return text


def _port(value):
    if type(value) is not int or not 1 <= value <= 65535:
        raise _invalid()
    return value


def _flag(value):
    if type(value) is not bool:
        raise _invalid()
    return value


def _origin(value):
    """An exact web origin: scheme, host and optional port, nothing else."""
    text = _text(value)
    try:
        parts = urlsplit(text)
        port = parts.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise _invalid() from None
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None or parts.password is not None
            or parts.path not in {"", "/"} or parts.query or parts.fragment or "?" in text or "#" in text or "\\" in text):
        raise _invalid()
    _host(parts.hostname)
    return text.rstrip("/")


def _origins(value):
    if value == {}:
        value = []  # A plain empty Lua table is unambiguous at this list field.
    if type(value) is not list or len(value) > MAX_ENTRIES:
        raise _invalid()
    return [_origin(item) for item in value]


def _copy(section: dict, values: dict, fields):
    for key, check, field in fields:
        if key in section:
            values[field] = check(section[key])


def _secret(section: dict, secrets: dict | None, field: str, values: dict):
    """Resolve a `secret = "name"` reference on the host; Lua only ever saw the name."""
    if "secret" not in section:
        return
    name = _text(section["secret"], SECRET_NAME)
    if secrets is None:
        return  # This process was given no credentials (e.g. the trainer).
    if name not in secrets:
        raise _invalid()  # Never connect with a missing credential.
    values[field] = secrets[name]


def _directory(section, secrets, values):
    if not section:
        return
    values["directory_backend"] = "ldap"
    _copy(section, values, (("host", _host, "ad_host"), ("port", _port, "ad_port"), ("tls", _flag, "ad_use_ssl"),
                            ("base_dn", _text, "ad_base_dn"), ("account_container", _text, "ad_account_container_dn"),
                            ("email_domain", _host, "account_email_domain")))
    if "groups" in section:
        groups = section["groups"]
        if type(groups) is not dict or len(groups) > MAX_ENTRIES:
            raise _invalid()
        values["ad_access_groups"] = {_text(name, KEY): _text(dn) for name, dn in groups.items()}
    for role, user_field, password_field in (("read", "ad_bind_user", "ad_bind_password"),
                                             ("write", "ad_write_bind_user", "ad_write_bind_password")):
        if role in section:
            account = section[role]
            if type(account) is not dict or set(account) - {"user", "secret"}:
                raise _invalid()
            _copy(account, values, (("user", _text, user_field),))
            _secret(account, secrets, password_field, values)


def _router(section, secrets, values):
    _copy(section, values, (("url", _origin, "openwrt_base_url"), ("user", _text, "openwrt_username"),
                            ("verify_tls", _flag, "openwrt_verify_tls"),
                            ("allow_loopback_http", _flag, "openwrt_allow_loopback_http")))
    _secret(section, secrets, "openwrt_password", values)


def _jira(section, secrets, values):
    if not section:
        return
    values["ticketing_backend"] = "jira"
    _copy(section, values, (("url", _origin, "jira_base_url"), ("email", _text, "jira_email")))
    _secret(section, secrets, "jira_api_token", values)


def _atlassian(section, secrets, values):
    _copy(section, values, (("org_id", _text, "atlassian_org_id"),
                            ("jira_resource", _text, "atlassian_jira_resource_ari")))
    _secret(section, secrets, "atlassian_admin_api_key", values)


def _mailbox(section, values):
    if section.get("enabled") is True and "domain" not in section:
        raise _invalid()
    _copy(section, values, (("enabled", _flag, "mailbox_lab_enabled"), ("domain", _host, "mailbox_lab_domain"),
                            ("smtp_port", _port, "mailbox_smtp_port"), ("imap_port", _port, "mailbox_imap_port")))


def _models(section, values, base: Path | None):
    from config.settings import ModelProfileSettings
    from config.path_portability import resolve_portable_path

    _copy(section, values, (("hub", lambda value: _text(value, KEY), "hub_model_key"),))
    if "profiles" in section:
        profiles = section["profiles"]
        if type(profiles) is not dict or len(profiles) > MAX_ENTRIES:
            raise _invalid()
        known = set(ModelProfileSettings.model_fields)
        for key, entry in profiles.items():
            # The model would silently ignore a misspelled field; refuse it here instead.
            if type(key) is not str or not KEY.fullmatch(key) or type(entry) is not dict or set(entry) - known:
                raise _invalid()
            normalized = dict(entry)
            try:
                for name in ("model_path", "adapter_path", "offload_folder"):
                    if normalized.get(name) is not None:
                        normalized[name] = resolve_portable_path(Path(_text(normalized[name])), base=base)
                ModelProfileSettings.model_validate(normalized, strict=True)
            except (TypeError, ValueError):
                raise _invalid() from None
            profiles[key] = {name: str(value) if isinstance(value, Path) else value for name, value in normalized.items()}
        values["model_profiles"] = profiles


def settings_values() -> dict:
    """Settings fields this machine's profile defines, with credentials resolved."""
    links = load_links()
    declared, secrets = links.declared, links.secrets
    if set(declared) - set(SECTIONS):
        raise _invalid()
    from config.settings import Settings
    controlled = set().union(*(CONTROLLED_FIELDS[name] for name in declared)) if declared else set()
    values: dict = {name: Settings.model_fields[name].get_default(call_default_factory=True) for name in controlled}
    _directory(_section(declared, "directory"), secrets, values)
    _router(_section(declared, "router"), secrets, values)
    _jira(_section(declared, "jira"), secrets, values)
    _atlassian(_section(declared, "atlassian"), secrets, values)
    _mailbox(_section(declared, "mailbox"), values)
    _models(_section(declared, "models"), values, links.profile_path.parent if links.profile_path else None)
    return values


def _origin_lists(section: dict) -> tuple[list[str], list[str]]:
    """Allowed origins and the plain-HTTP exceptions, each of which must also be allowed."""
    allowed = _origins(section.get("allowed_origins", []))
    insecure = _origins(section.get("insecure_http_origins", []))
    if any(urlsplit(origin).scheme != "http" or origin not in allowed for origin in insecure):
        raise _invalid()  # Plain HTTP is an explicit exception for an origin already allowed.
    return allowed, insecure


def process_policy() -> dict:
    """The outbound model origins this machine allows, as the transport's environment."""
    links = load_links()
    allowed, insecure = _origin_lists(_section(links.declared, "models"))
    policy = {}
    if "models" in links.declared:
        policy["EXTENSION_ALLOWED_ORIGINS"] = ",".join(allowed)
        policy["EXTENSION_INSECURE_HTTP_ORIGINS"] = ",".join(insecure)
    return policy


def source_policy() -> dict:
    """The web origins this machine allows the knowledge crawler to read.

    Separate from the model origins: a crawlable documentation site never becomes
    a model endpoint, and a model endpoint is never crawled. Undeclared means none.
    """
    allowed, insecure = _origin_lists(_section(load_links().declared, "sources"))
    return {"allowed_origins": allowed, "insecure_http_origins": insecure}
