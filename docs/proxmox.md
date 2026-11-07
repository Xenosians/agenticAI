# Proxmox VE integration

The first integration slice is read only. It uses the configured Proxmox VE
REST API and the shared Lua-gated HTTP client. Provider hosts and credentials
are machine configuration, not model arguments or source constants.

The registered capabilities currently cover cluster membership, node resource
counters, QEMU/LXC inventory, per-node storage and recent per-node task
history. Results are schema checked, bounded and reduced to fields needed for
operations. A missing provider configuration or any non-200/malformed API
response is reported as unavailable or failed; no success data is synthesized.

## Tailscale prep (spare PC not yet installed)

Current state: **Proxmox is not installed; the dashboard stays
`not_configured` by design.** Tailscale is already present on the AI host
(OAuth client in `.env` drives the separate tailnet API: devices + DNS),
but that does not make Proxmox reachable. The spare PC needs its own
Tailscale join before any Proxmox URL means anything.

Decision (operator-owned, documented so it is not re-litigated per request):

| Need | Preferred tailnet shape | Why |
| --- | --- | --- |
| Read-only, authorized, non-sensitive inventory (what ships today) | **Tailscale serve delegation** — `https://<machine>.tailXXX.ts.net` proxying to the local Proxmox API at `https://127.0.0.1:8006` | Smallest operator burden: one `tailscale serve` command, public-trust Let's Encrypt cert for the MagicDNS name, no custom CA to distribute, `verify_tls = true` with no `ca_file`. Latency is one in-host hop. |
| Direct Proxmox interaction (future governed mutations) | **Native Proxmox `:8006`** — `https://<machine>.tailXXX.ts.net:8006` | The AI talks to the daemon directly, so the operator can allow/disallow direct access per need. Requires certificate trust (Step 3) and an explicit operator decision. No mutation ships until it has its own approval + read-back contract. |

Never use **Tailscale Funnel** for Proxmox: Funnel is public internet, not
tailnet-only. Serve is tailnet-only and honours access-control rules.

### Spare-PC checklist (do in order, no shortcuts)

1. **Install + join the tailnet.** Install the Tailscale client on the
   spare PC, `tailscale up` into the same tailnet, and confirm it owns a
   MagicDNS name (`<machine>.tailXXX.ts.net`). Note the exact name; do not
   use a LAN IP or a `*.local` name. The AI host must `tailscale status`
   and see the spare PC online.
2. **Enable HTTPS in the admin console.** In the Tailscale admin console,
   enable HTTPS certificate provisioning for the tailnet (DNS → HTTPS).
   `tailscale serve` will request a consent when it is missing — use that
   click; it is one-time. Without this, Serve has no certificate.
3. **Choose the TLS path (pick one, document which).**
   - *Serve path (preferred for reads):* `sudo tailscale serve --bg --https=443 https+insecure://127.0.0.1:8006`
     (Tailscale 1.52+ syntax; `--bg` keeps it across reboots; undo with
     `sudo tailscale serve --https=443 off`). Proxmox's `pveproxy` serves
     HTTPS only, with a self-signed certificate by default. `https+insecure`
     accepts that certificate for the in-host hop to `127.0.0.1` only; the
     tailnet side uses the Let's Encrypt certificate for the MagicDNS name.
     Then `proxmox_base_url = "https://<machine>.tailXXX.ts.net"` (`verify_tls = true`, no `ca_file`, public trust).
   - *Native path:* `tailscale cert <machine>.tailXXX.ts.net` on the spare PC,
     install the issued cert as Proxmox's own SSL certificate
     (`/etc/pve/pve-ssl.pem` + key, then `systemctl restart pveproxy`), **or**
     keep Proxmox's self-signed CA. Either way copy the CA that signed
     `:8006` to the AI host and point `ca_file` at it. Never set
     `verify_tls = false`.
4. **Least-privilege API token.** In Proxmox, create a user (e.g.
   `janus-reader@pam`, or a `pve` realm user), then Datacenter → Permissions
   grant only the reads below, then Privileges → API Tokens create
   `janus-reader@pam!operations` with **no** privilege separation. Keep the
   printed secret in the AI host's protected link-secrets file only.
   Reads used today: `GET /cluster/status`, `GET /nodes`,
   `GET /cluster/resources?type=vm`, `GET /nodes/{node}/storage`,
   `GET /nodes/{node}/tasks`. The official
   [Proxmox VE API Viewer](https://pve.proxmox.com/pve-docs/api-viewer/)
   defines each permission. Start with the built-in `View` role scope;
   widen only on a recorded 403 naming the exact permission.
5. **Wire the Lua machine profile** (see below) and reload the AI service.
6. **Acceptance from the AI host (real evidence, not fixtures):**
   ```bash
   curl --cacert <cafile-if-native> \
     -H "Authorization: PVEAPIToken=janus-reader@pam!operations=$SECRET" \
     https://<machine>.tailXXX.ts.net/api2/json/version
   curl --cacert <cafile-if-native> \
     -H "Authorization: PVEAPIToken=janus-reader@pam!operations=$SECRET" \
     https://<machine>.tailXXX.ts.net/api2/json/nodes
   ```
   Expect `{"data":…}` with HTTP 200 from both. Then refresh the
   Operations dashboard; `Connected` must show observed nodes/guests.
   A passing fixture test never substitutes for this.

### Troubleshooting (map the failure code, do not relax verification)

| Symptom | Likely cause | Fix direction |
| --- | --- | --- |
| `proxmox_not_configured` | Missing `PROXMOX_BASE_URL` / token id / secret in the Lua profile | Finish Steps 4–5; do not fabricate config |
| `proxmox_configuration_invalid` | `http://`, loopback host, credentials in URL, path suffix, or bad token id | Use the exact HTTPS origin from Step 3; token id must match `user@realm!token` |
| `proxmox_read_failed` (502) | DNS/TLS/TCP failure, token 401/403, or malformed provider answer | `tailscale status`, `tailscale serve status`, curl acceptance above, AI logs; check ACL allows the AI host to reach the spare PC; never set `verify_tls = false` |
| Browser shows certificate warning only when visiting by LAN IP | Wrong name: you must browse/API by the MagicDNS name | Always use `<machine>.tailXXX.ts.net` |

Configure a dedicated read-only Proxmox API token with the minimum permissions
the chosen reads require. Keep its value in the protected link-secrets file and
reference its name from the Lua machine profile:

```lua
links = {
  -- Serve path (delegated reads, preferred):
  proxmox = {
    url = "https://spare-pc.tail12345.ts.net",
    token_id = "janus-reader@pam!operations",
    secret = "proxmox_api_token",
    verify_tls = true,
  },
  -- Native path (direct :8006, future governed mutations):
  -- proxmox = {
  --   url = "https://spare-pc.tail12345.ts.net:8006",
  --   token_id = "janus-reader@pam!operations",
  --   secret = "proxmox_api_token",
  --   verify_tls = true,
  --   ca_file = "trust/pve-root-ca.pem",
  -- },
}
```

The token secret stays on the host. `ca_file` is relative to the machine
profile when supplied and is needed only for the native path with a
self-signed or private CA. For the Serve path omit it and rely on system
trust (Let's Encrypt). Keep certificate verification enabled; trust the
Proxmox CA or install a certificate signed by a trusted CA instead of
disabling verification.

The integrated capability set is intentionally smaller than the complete
Proxmox API. VM and container mutations, configuration writes, snapshots,
backups, restores, network changes, access administration and live provider
acceptance remain unimplemented until each action has a separate
exact-target approval and read-back contract. A configured token or a passing
fixture test does not establish that a live cluster is reachable.
