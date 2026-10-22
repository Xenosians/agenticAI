# Networking (OpenWrt)

The networking specialist reads one configured OpenWrt router over its ubus
JSON-RPC API with a dedicated read-only rpcd account. Every capability is a fixed
read; there is no generic RPC tool, and nothing can change, commit or restart the
router. Credentials never appear in tool arguments or results.

| Tool | Reads | Notes |
|---|---|---|
| `network_system_info` | `system board` | Hostname, hardware, firmware release. |
| `network_interface_status` | `network.interface.<name> status` | The user must name the logical interface. |
| `network_firewall_config` | `uci get firewall` | Saved policy only; it does not prove packets are filtered. |
| `network_dhcp_leases` | `luci-rpc getDHCPLeases` | IPv4 and IPv6 leases from the router's own DHCP server (at most 200 listed, with the total count). Static hosts are not listed. |
| `network_wireless_clients` | `iwinfo devices`, `iwinfo assoclist` | Associated stations per wireless interface: MAC, signal, noise, rates and idle time (at most 200 per interface). |

A router without radios returns an empty client list with the note "The router
reports no wireless interfaces." An empty lease list means the DHCP server has
handed out no leases (or is disabled). Neither is an error.

The rpcd ACL must allow exactly these reads. The lab ACL is defined in
`projectOps/openwrt-lab/provision.py`. The local lab disables its DHCP server
and has no radios, so both client reads are empty there.
