# Networking (OpenWrt)

The networking specialist reads one configured OpenWrt router over its ubus
JSON-RPC API with a dedicated read-only rpcd account. Every capability is a fixed
read; there is no generic RPC tool, and nothing can change, commit or restart the
router. Credentials never appear in tool arguments or results.

| Tool | Reads | Notes |
|---|---|---|
| `network_system_info` | `system board` | Hostname, hardware, firmware release. |
| `network_router_health` | `system info` | Uptime, load average, memory and storage use. |
| `network_interfaces` | `network.interface dump` | Every logical interface: state, protocol, device, addresses, routes and DNS servers. |
| `network_interface_status` | `network.interface.<name> status` | The user must name the logical interface. |
| `network_firewall_config` | `uci get firewall` | Saved policy only; it does not prove packets are filtered. |
| `network_dns_config` | `uci get dhcp` (type `dnsmasq`) | Saved resolver settings: upstream servers, whether the internet connection's resolvers are also used, rebind protection, DNSSEC. It does not perform lookups. |
| `network_dhcp_leases` | `luci-rpc getDHCPLeases` | IPv4 and IPv6 leases from the router's own DHCP server (at most 200 listed, with the total count). Static hosts are not listed. |
| `network_known_hosts` | `luci-rpc getHostHints` | Hosts the router knows by MAC (neighbour table, leases, static entries) with names and addresses (at most 200). It does not scan the network. |
| `network_wireless_clients` | `iwinfo devices`, `iwinfo assoclist` | Associated stations per wireless interface: MAC, signal, noise, rates and idle time (at most 200 per interface). |

A router without radios returns an empty client list with the note "The router
reports no wireless interfaces." An empty lease list means the DHCP server has
handed out no leases (or is disabled). Neither is an error.

The rpcd ACL must allow exactly these reads. The lab ACL is defined in
`projectOps/openwrt-lab/provision.py`. The local lab disables its DHCP server
and has no radios, so both client reads are empty there.

## Tailscale network (tailnet)

Four reads (and the coverage join below) use the Tailscale API with the deployment's own trust credential
(`TAILSCALE_CLIENT_ID`, `TAILSCALE_CLIENT_SECRET`), holding `devices:core:read`,
`dns:read`, `devices:posture_attributes:read` and `policy_file`. `policy_file`
also allows writing the policy; the reads only read it, and the two governed
grant changes below write it, each after the owner's approval. The credential
cannot approve, expire or remove devices, or change DNS. Its one-hour token is reused until five
minutes before it expires, so most reads make one API request.

| Tool | Reads | Notes |
|---|---|---|
| `network_tailnet_devices` | `GET /tailnet/-/devices?fields=all` | Each device's name, OS, Tailscale version, online state, last seen, key expiry (days left, or never), pending update, approval and tags (at most 200), plus its health: whether it blocks incoming connections, how many advertised routes wait for approval, whether it offers itself as an exit node, whether tailnet lock reports a problem, and how it reaches peers (UDP, hard NAT, nearest relay and its latency, from the device's last report). Each device lists its `issues` (below), and a summary counts online, offline and devices needing attention. Tailscale addresses, endpoints, route prefixes, node and machine keys, posture data and user details are not returned. |
| `network_tailnet_policy` | `GET /tailnet/-/acl`, `GET /tailnet/-/settings` | Who may reach what: each grant and ACL rule (sources, destinations, ports), SSH rules (action, sources, destinations, local users), tag owners, group sizes, the number of host aliases and policy tests, other policy sections by name, whether the policy is managed outside the admin console, and whether every member reaches every device on every port. Tags, groups, autogroups and `*` stay as written; users (emails) read "a user", host aliases "a host alias", addresses and CIDRs "an address", SSH users other than `root` and autogroups "a named user". At most 200 rules per section. |
| `network_tailnet_posture` | `GET /tailnet/-/devices`, then `GET /device/{id}/attributes` per device (at most 50) | Each device's OS and version, Tailscale version and release track, whether Tailscale auto-update is on and whether it stores its Tailscale state encrypted; custom and integration attributes are only counted. A device whose posture cannot be read is unknown. |
| `network_tailnet_coverage` | Proxmox `GET /cluster/resources` and the tailnet devices read | Which Proxmox nodes and guests (templates left out) have a tailnet device, which running machines have their device offline (`device_offline`), and which devices are named like no Proxmox machine. The two inventories share no identifier Janus may expose, so they match by name: the device's host name or the first label of its MagicDNS name, lower case with every other character run as one hyphen. A machine without a match has `no_tailnet_device` and may still be on the tailnet under another name; when either listing was cut, no machine is called missing. Each side's failure is named (`proxmox` or `tailscale`). |
| `network_tailnet_dns` | `GET /tailnet/-/dns/configuration` | MagicDNS, whether devices use the tailnet's resolvers instead of the local network's, each resolver and whether Tailscale reaches it over DNS-over-HTTPS, split DNS and search domains. |

Device issues, most urgent first: `key_expired`, `key_expiring` (within 14
days), `waiting_for_approval`, `tailnet_lock_error`,
`routes_waiting_for_approval`, `blocks_incoming`, `udp_blocked` (traffic goes
through a relay) and `update_available`. A device that sent no connectivity
report has `connectivity: null`; that is unknown, not healthy.

### Governed policy changes

Two tools change who may reach what, one grant at a time; nothing else in the
policy can be written, and the model never writes policy text.

| Tool | Changes | Approval |
|---|---|---|
| `network_tailnet_grant_add` | Adds one grant `{"src": [...], "dst": [...], "ip": [...]}` as the last entry of `grants` (creating the section if it is missing). | Owner approval of the exact grant. |
| `network_tailnet_grant_remove` | Removes the one grant with exactly those sources, destinations and ports (order does not matter). | Owner approval and the grant typed exactly as the approval shows it, for example `tag:janus-lab -> tag:janus-lab : 22`. |

Sources are tags (`tag:name`), groups (`group:name`) or autogroups
(`autogroup:member`, `autogroup:self`, ...); destinations may also be `*`.
Ports are `*`, a port, a range or `tcp:`/`udp:`/`icmp:` with one of those.
Users, addresses and host aliases are refused, so no identity or address
enters a proposal.

The policy file is HuJSON (comments and trailing commas), so a change edits
its text and keeps every other byte; the parsed result must differ from the
original by exactly that grant. Before approval, trusted policy reads the text
and its ETag (`Accept: application/hujson`), refuses a policy managed outside
the admin console, makes the edit, has Tailscale validate it with
`POST /tailnet/-/acl/validate` (which writes nothing) and freezes the ETag and
the SHA-256 of the current and the changed text. After approval the change
reads the policy again and refuses (`conflict`, nothing written) if its ETag
or digest moved, makes the same edit and checks its digest, writes it with
`POST /tailnet/-/acl` and `If-Match` (a concurrent edit answers 412: refused,
never overwritten), then reads the policy back. A lost answer after the write
is `outcome_unknown` with `retry_safe: false`. The receipt names the grant,
never the policy text.

The backend reads both through `GET /internal/v1/tailnet/overview` (internal
token) and serves them to the signed-in owner at `GET /api/v1/tailnet/overview`
for the Operations page. Devices decide whether the overview succeeds; a failed
DNS read is named beside them in `dns_error`.

A resolver counts as encrypted when it is one of the public resolvers Tailscale
upgrades to DNS-over-HTTPS automatically (Cloudflare, Google, Quad9). That is
what keeps a network that rewrites plain DNS answers from redirecting tailnet
devices. Reaching the lab router over the tailnet is described in
`projectOps/openwrt-lab/README.md`.

## DNS integrity check

`network_dns_integrity_check` checks, from the AI host, whether DNS answers are
being rewritten on the way: some networks intercept plain DNS on port 53 and
answer with their own addresses (block pages, redirection). For each name it
asks three ways and compares the IPv4 answers:

| Path | Can a network rewrite it? |
|---|---|
| Plain DNS over UDP to `DNS_CHECK_RESOLVER` (default `1.1.1.1`) | Yes |
| The host's own resolver (what the services here use) | Depends on its setup |
| DNS-over-HTTPS to `DNS_CHECK_DOH_URL` (default Cloudflare) | No, without breaking the certificate |

A plain answer that shares no address with the encrypted one is reported as
rewritten; overlapping answers are consistent, because CDNs rotate addresses.
The result also says whether the host's own resolver is protected (it gave the
true answers while plain DNS was rewritten). A path that fails is reported as
not checked, never as tampering. With no names given it checks
`DNS_CHECK_NAMES` (default `example.com,wikipedia.org,github.com`); at most 10
names per check. Each path crosses the Lua connection adapter first.
