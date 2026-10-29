# Lua machine profiles

A machine profile is the one reviewed Lua file that declares where this
deployment connects: its directory, router, ticketing, mailbox lab and model
endpoints, plus any reviewed protocol plugin files. Images and code carry no
destination, account or model path. Moving the application to another machine
means mounting that machine's profile and its separate secret store; every
inbound and outbound link follows them.

The profile is the same script the [adapter runtime](lua-adapters.md) runs at
every boundary, selected and pinned the same way:

- `ITSM_LUA_ADAPTER_FILE`: absolute path of the profile (not a symlink);
- `ITSM_LUA_ADAPTER_SHA256`: its SHA-256. A changed or unreadable file refuses
  every boundary rather than falling back to Python or to environment values;
- `ITSM_LINK_SECRETS_FILE`: absolute path of the private credential store
  described below. Optional; without it the process receives no credentials.

## Shape

```lua
return {
  links = {
    directory = {host = "dc.example.org", port = 636, tls = true,
                 base_dn = "DC=example,DC=org",
                 account_container = "OU=Staff,DC=example,DC=org",
                 email_domain = "example.org",
                 groups = {vpn = "CN=VPN Users,OU=Groups,DC=example,DC=org"},
                 read = {user = "svc-read", secret = "ldap_read"},
                 write = {user = "svc-write", secret = "ldap_write"}},
    router = {url = "https://192.0.2.1", user = "auditor", secret = "router", verify_tls = true},
    jira = {url = "https://example.atlassian.net", email = "bot@example.org", secret = "jira"},
    atlassian = {org_id = "org-id", jira_resource = "ari:cloud:jira::site/id", secret = "atlassian"},
    mailbox = {enabled = true, domain = "lab.example.org", smtp_port = 587, imap_port = 993},
    models = {hub = "hub-main",
              profiles = {["hub-main"] = {backend = "transformers", model_path = "models/hub"}},
              allowed_origins = {"https://api.anthropic.com"},
              insecure_http_origins = {}},
  },
  plugins = {
    ["model.harness"] = {file = "plugins/my_harness.lua", sha256 = "<lowercase sha256>"},
  },
  adapt = function(event) return event end,  -- optional; runs at every boundary
}
```

Every key is optional. A script that returns only a function declares no links
and keeps the packaged plugins. The host refuses, rather than ignores, an
unknown section, an unknown key in a section, a misspelled model-profile field
and a malformed host, port, origin or name.

| Section | Keys | Settings it controls |
| --- | --- | --- |
| `directory` | `host`, `port`, `tls`, `base_dn`, `account_container`, `email_domain`, `groups`, `read`, `write` | LDAP directory backend and its read and write bind accounts |
| `router` | `url`, `user`, `secret`, `verify_tls`, `allow_loopback_http` | OpenWrt base URL and account |
| `jira` | `url`, `email`, `secret` | Jira ticketing backend |
| `atlassian` | `org_id`, `jira_resource`, `secret` | Atlassian organization administration |
| `mailbox` | `enabled`, `domain`, `smtp_port`, `imap_port` | Lab mailbox provisioning |
| `models` | `hub`, `profiles`, `allowed_origins`, `insecure_http_origins` | Hub model, local model profiles and the outbound model origins |

A declared section replaces every setting it controls, including ones it leaves
out, which take their defaults. A stale host or credential left in the
environment or `.env` therefore never reaches a connection. Sections the profile
does not declare keep their existing configuration. Explicit constructor
arguments still win.

Relative model paths resolve beside the profile. `allowed_origins` lists exact
`scheme://host[:port]` origins; plain HTTP is allowed only for an origin listed
in both `allowed_origins` and `insecure_http_origins`.

## Credentials

A credential is never written into Lua. A key named `secret` holds the name of
an entry in the private JSON object at `ITSM_LINK_SECRETS_FILE`:

```json
{"ldap_read": "...", "ldap_write": "...", "router": "...", "jira": "...", "atlassian": "..."}
```

Only the host process reads that file, after Lua has returned its declarations.
The file must be a regular file (not a symlink), readable only by its owner
(no group or other permission bits), at most 64 KB, with at most 64 entries of
lowercase names. A missing entry, a duplicate name or a malformed file refuses
the configuration; the host never connects with a missing credential. A process
started without the variable, such as the trainer, receives the declared links
without any credentials.

## Protocol plugins

`plugins` selects reviewed Lua files for the named protocol codecs listed in
[Lua plugins](lua-plugins.md). Each entry names a `.lua` file beneath the
profile's directory and its SHA-256; the host rechecks the bytes at dispatch and
never falls back to the packaged file when a selected one is invalid.

## Checking a profile

`services.lua_adapters.profile.settings_values()` returns the settings a profile
defines, with credentials resolved, and `process_policy()` the outbound model
origins. Both refuse with `lua_adapter_configuration_invalid` instead of
returning partial values. The tests in `tests/unit/services/test_lua_profile.py`
show accepted and refused profiles.
