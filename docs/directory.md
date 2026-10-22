# Directory reads (Samba AD / Active Directory)

Four read-only tools inspect the configured directory over LDAP with the read
bind (`AD_BIND_DN` or `AD_BIND_USER`, `AD_BIND_PASSWORD`). None of them needs
approval, opens the write bind, or changes the directory. Password and hash
attributes are never requested.

| Tool | Owner | Arguments | Returns |
|---|---|---|---|
| `account_search` | account-specialist | `query`: exact text fragment from the user | Up to 25 accounts whose username, display name or email contains the fragment: username, display name, email, enabled, locked. |
| `user_groups` | access-specialist | `user_id`: exact username | Every group of the user: `direct` (memberOf), `nested` (through other groups) or `primary` (the primaryGroupID group, usually Domain Users). Groups that a logical resource in `AD_ACCESS_GROUPS` maps to carry that resource name. At most 200. |
| `group_members` | access-specialist | `group`: logical resource (`vpn`, `admin`, or an alias such as "vpn access") or exact group name | Members (users, groups, computers) with their membership kind. At most 100, read with the paged-results control. |
| `access_explain` | access-specialist | `user_id`, `resource` | `has_access`, the individual checks and a list of reasons. |

## How membership is read

Nested membership uses the AD matching rule `LDAP_MATCHING_RULE_IN_CHAIN`
(1.2.840.113556.1.4.1941), which Samba supports. Primary-group membership is not
stored in `member`/`memberOf`, so it is resolved from the account's SID and
`primaryGroupID`. User-supplied text is escaped with
`ldap3.utils.conv.escape_filter_chars`; search fragments must be 2 to 64
characters. Every bounded read uses a server time limit and reports
`truncated: true` when the cap was reached.

`group_members` resolves a configured logical resource first (the group in
`AD_ACCESS_GROUPS`, as a CN or full DN, exactly like `check_access` and
`grant_access`), then an exact group CN or account name. "the vpn group" is read
as the `vpn` resource. Two groups with the same CN return `ambiguous`.

## Access explanation

`access_explain` combines `user_groups`, `account_status` and `group_members`;
it has no LDAP logic of its own. Reasons and suggested fixes:

| Reason code | Meaning | Suggested tool |
|---|---|---|
| `account_not_found` | No account with that username. | none |
| `account_disabled` | userAccountControl has ACCOUNTDISABLE. | `enable_user` |
| `account_locked` | lockoutTime is set. | `unlock_user` |
| `resource_not_mapped` | No `AD_ACCESS_GROUPS` entry for the resource; access is undetermined. | none |
| `resource_group_missing` | The configured group does not exist in the directory. | none |
| `not_a_member` | Not a direct, nested or primary member of the mapped group. | `grant_access` |
| `membership_unknown` | The user has more groups than one read returns; membership could not be confirmed. | none |

Suggested tools are text only. The explanation never executes them; each one is
a separate, approval-gated request. `has_access: true` means the directory does
not block access; a failure can still come from outside the directory (VPN
server, network, client).

## Errors

Results carry `ok: false` with `error` and `error_code`: `invalid_argument`
(empty or out-of-range input), `not_found`, `ambiguous` or `provider_error`
(bind, timeout or LDAP result failure). A failing directory is never reported as
a missing account.

## Configuration

`AD_HOST`, `AD_PORT`, `AD_USE_SSL`, `AD_BASE_DN`, `AD_BIND_DN` (or
`AD_BIND_USER`), `AD_BIND_PASSWORD`, `AD_ACCESS_GROUPS` (JSON object such as
`{"vpn": "VPN-Users", "admin": "ITSM-Admins"}`; values are group CNs or DNs) and
`DIRECTORY_BACKEND` (`ldap` or `mock`). The mock backend derives its groups from
its in-memory access state.
