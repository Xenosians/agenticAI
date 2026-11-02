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

Two more reads use the Tailscale API with the deployment's own OAuth client
(`TAILSCALE_CLIENT_ID`, `TAILSCALE_CLIENT_SECRET`). Give that client only the
`devices:core:read` and `dns:read` scopes; it cannot approve, expire or remove
devices, or change DNS. Its one-hour token is reused until five minutes before
it expires, so most reads make one API request.

| Tool | Reads | Notes |
|---|---|---|
| `network_tailnet_devices` | `GET /tailnet/-/devices` | Each device's name, OS, Tailscale version, online state, last seen, key expiry (days left, or never), pending update, approval and tags (at most 200). Tailscale addresses, node and machine keys and user details are not returned. |
| `network_tailnet_dns` | `GET /tailnet/-/dns/configuration` | MagicDNS, whether devices use the tailnet's resolvers instead of the local network's, each resolver and whether Tailscale reaches it over DNS-over-HTTPS, split DNS and search domains. |

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
