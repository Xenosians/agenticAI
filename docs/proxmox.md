# Proxmox VE integration

Janus reads Proxmox VE and, after the owner confirms each one, starts, shuts
down, reboots or force-stops one exact guest. It uses the configured Proxmox VE
REST API and the shared Lua-gated HTTP client. Provider hosts and credentials
are machine configuration, not model arguments or source constants.

The read capabilities cover cluster membership, node resource counters,
QEMU/LXC inventory and exact-guest status/config reads, per-node storage,
recent per-node task history and the database host inventory. Guest details
require the exact node and VMID from the live inventory. Configuration output
is an allow-list and omits credentials, SSH keys and private network settings.
Results are schema checked, bounded and reduced to fields needed for
operations. A missing provider configuration or non-200/malformed API response
is reported as unavailable or failed with a stable reason
(`provider_unreachable`, `provider_timeout`, `provider_refused`,
`provider_error`); no success data is synthesized and the provider's own answer
is never echoed.

Proxmox does not measure the used disk of a QEMU VM (it reports 0 for "no
information"), and measures a container's only while it runs. Janus reports
those values as unknown, never as an empty disk; the disk size is always shown.

## Database host inventory

`proxmox_database_inventory` lists the guests the operator tagged as database
hosts in Proxmox: tag a VM or container `db`, or `db-<engine>` to name the
engine (for example `db-postgres` or `db-redis`). Janus never decides from a
guest's name that it hosts a database, and a template is never listed. Tags
are set in the Proxmox web UI (guest → Summary → tags) or with
`qm set <vmid> --tags db-postgres` / `pct set <vmid> --tags db-postgres`.

Each tagged guest shows its engine, state, lock, memory, disk size and (for a
running container) disk use, and whether a backup job covers it. What needs
attention is named in a fixed order: `not_running`, `locked`, `no_backup_job`
and `disk_nearly_full` (90 % or more of a measured disk). The storage pools
that hold guest disks on the same nodes are listed with how full they are.
Two reads serve the whole inventory: `GET /cluster/resources` and
`GET /cluster/backup-info/not-backed-up`.

Backup coverage needs `Sys.Audit` on `/` (part of `PVEAuditor`). When the
token cannot read it, or Proxmox does not answer that read, coverage is
reported as unknown with the reason; no guest is shown as unprotected. Coverage
means a backup job includes the guest; it does not prove a backup ran or can
be restored.

The inventory never connects to a database: it has no database credentials,
runs no SQL and cannot back up, restore or change one. A database contract
of that kind needs its own owner-approved design.

The live tool matrix can exercise the guest detail and power calls when the
disposable acceptance profile provides `ACCEPTANCE_PROXMOX_NODE` and
`ACCEPTANCE_PROXMOX_VMID`. Without both values those cases remain skipped for
missing fixture context; they do not invent a node or VMID. Point them only at
a disposable lab guest: the power cases start, reboot, shut down, start and
force-stop it, in an order that leaves it as found.

## Guest power actions

`proxmox_guest_start`, `proxmox_guest_shutdown`, `proxmox_guest_reboot` and
`proxmox_guest_force_stop` act on one exact `node` + `vmid`. Each one:

1. **Freezes the exact target before approval.** Trusted policy reads the
   guest once and records its type and name in the approval. A guest that is
   locked (for example by a backup), is a template, or is not in a state the
   action applies to is refused with no request sent.
2. **Waits for the owner's approval.** The approval names the guest, its node
   and what the action does. Force stop is high risk and is offered only for
   explicit wording ("force", "kill", "hard stop").
3. **Re-checks, then sends one request.** If the guest's type or name changed
   since approval, nothing is sent. Exactly one POST goes to
   `/nodes/{node}/{qemu|lxc}/{vmid}/status/{start|shutdown|reboot|stop}`
   (shutdown and reboot ask the guest's operating system, with a 60-second
   timeout).
4. **Confirms with receipts.** Janus follows the returned Proxmox task (UPID)
   until it finishes, then reads the guest back. Success is reported only when
   the task ended `OK` and the read-back shows the expected state.

A transport failure after the request was sent, an unrecognised task id, or a
task that does not finish within 120 seconds is `outcome_unknown`: the action
is never retried automatically, and the guest must be checked first. Deletion,
disk, network, snapshot, backup and access changes are not available.

The Operations dashboard offers the same four actions per guest (only those
that fit its current state). A button only proposes the action through the
same governed path a chat request uses; nothing runs until the owner confirms
the server's description of the exact target. Phoenix records each action
once per submission, lets only its owner decide it, and never re-sends an
approval whose outcome was not confirmed.

## Deployment shapes

Reads and actions work over either a tailnet name or a lab LAN address. Both
keep certificate verification on; the difference is which certificate and name
the AI host trusts.

- **Tailnet name (preferred; live):** `https://<machine>.<tailnet>.ts.net`
  through `tailscale serve`, with the public Let's Encrypt certificate. The
  name stays the same wherever the Proxmox host is plugged in.
- **LAN address with the cluster CA:** `https://<lan-address>:8006`, with
  `ca_file` naming the CA that signed the Proxmox certificate and
  `server_name` naming the host on that certificate. The URL may be the
  address; TLS still verifies the certificate against `server_name` and the
  CA. Use this only where the tailnet is unavailable; the address changes
  with the network the host is on.

Never disable verification. A certificate-name mismatch is fixed with
`server_name`, not with `verify_tls = false`.

## Tailscale setup

The lab host `hummingbird` (Proxmox VE 9) has joined the tailnet and serves
its API as `https://hummingbird.<tailnet>.ts.net` with
`tailscale serve --bg https+insecure://localhost:8006`. The development stack's
machine profile points at that name with no `ca_file`, so the same address
works at home and at the office. Tailscale on the AI host also drives the
separate tailnet API (devices + DNS), but that does not make a Proxmox host
reachable: each Proxmox host needs its own Tailscale join.

Decision (operator-owned, documented so it is not re-litigated per request):

| Need | Preferred tailnet shape | Why |
| --- | --- | --- |
| Reads and governed power actions (what ships today) | **Tailscale serve delegation** — `https://<machine>.tailXXX.ts.net` proxying to the local Proxmox API at `https://127.0.0.1:8006` | Smallest operator burden: one `tailscale serve` command, public-trust Let's Encrypt cert for the MagicDNS name, no custom CA to distribute, `verify_tls = true` with no `ca_file`. Latency is one in-host hop. |
| Direct daemon access | **Native Proxmox `:8006`** — `https://<machine>.tailXXX.ts.net:8006` | The AI talks to the daemon directly. Requires certificate trust (Step 3) and an explicit operator decision. |

Never use **Tailscale Funnel** for Proxmox: Funnel is public internet, not
tailnet-only. Serve is tailnet-only and honours access-control rules.

### Host checklist (do in order, no shortcuts)

1. **Install + join the tailnet.** Install the Tailscale client on the
   Proxmox host, `tailscale up --hostname=<machine> --accept-dns=false` into
   the same tailnet (keep the host's own DNS), and confirm it owns a MagicDNS
   name (`<machine>.tailXXX.ts.net`). Do not use a LAN IP or a `*.local`
   name. The AI host must `tailscale status` and see the host online.
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
   - *Native path:* `tailscale cert <machine>.tailXXX.ts.net` on the host,
     install the issued cert as Proxmox's own SSL certificate
     (`/etc/pve/pve-ssl.pem` + key, then `systemctl restart pveproxy`), **or**
     keep Proxmox's self-signed CA. Either way copy the CA that signed
     `:8006` to the AI host and point `ca_file` at it. Never set
     `verify_tls = false`.
4. **Least-privilege API token.** In Proxmox, create a `pve` realm user
   (e.g. `janus@pve`) and an API token for it (e.g. `janus@pve!dashboard`)
   with **no** privilege separation. Keep the printed secret in the AI host's
   protected link-secrets file only. Grant only:
   - `PVEAuditor` on `/` for the reads: `GET /cluster/status`,
     `GET /cluster/resources`, `GET /cluster/backup-info/not-backed-up`,
     `GET /nodes/{node}/storage`,
     `GET /nodes/{node}/tasks[/{upid}/status]` and the exact-guest
     `status/current` and `config` reads;
   - a custom role holding only `VM.PowerMgmt` (and `VM.Audit`) on `/vms`
     for the four power actions. Omit it to keep the integration read only;
     the power tools then fail with a permission refusal and change nothing.

   The official
   [Proxmox VE API Viewer](https://pve.proxmox.com/pve-docs/api-viewer/)
   defines each permission. Widen only on a recorded 403 naming the exact
   permission.
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
| `proxmox_read_failed` (502) | DNS/TLS/TCP failure, token 401/403, or malformed provider answer; `reason` names which | `tailscale status`, `tailscale serve status`, curl acceptance above, AI logs; check ACL allows the AI host to reach the Proxmox host; never set `verify_tls = false` |
| A power action is refused before approval | The guest is locked, a template, or not in a state the action applies to | Wait for the lock (for example a backup) to clear, or choose the action that fits the guest's state |
| A power action ends `outcome_unknown` | The request was sent but the task or read-back could not be confirmed | Check the guest (dashboard or `proxmox_guest_status`) before trying again; Janus never retries it |
| Certificate name mismatch when reading by LAN address | The certificate names the host, not the address | Keep the address in `url`, point `ca_file` at the cluster CA, and set `server_name` to the certificate's name |
| Browser shows certificate warning only when visiting by LAN IP | Wrong name for a browser session | For the UI use the MagicDNS name; API reads may use `url` + `server_name` as above |

Configure a dedicated least-privilege Proxmox API token (Step 4). Keep its
value in the protected link-secrets file and reference its name from the Lua
machine profile:

```lua
links = {
  -- Serve path (preferred):
  proxmox = {
    url = "https://spare-pc.tail12345.ts.net",
    token_id = "janus@pve!dashboard",
    secret = "proxmox_api_token",
    verify_tls = true,
  },
  -- Native path (direct :8006):
  -- proxmox = {
  --   url = "https://spare-pc.tail12345.ts.net:8006",
  --   token_id = "janus@pve!dashboard",
  --   secret = "proxmox_api_token",
  --   verify_tls = true,
  --   ca_file = "trust/pve-root-ca.pem",
  -- },
  -- LAN path with a private CA (lab): the URL may be the address, while
  -- server_name is the name on the certificate. Verification stays on.
  -- proxmox = {
  --   url = "https://192.0.2.10:8006",
  --   token_id = "janus@pve!dashboard",
  --   secret = "proxmox_api_token",
  --   verify_tls = true,
  --   ca_file = "trust/pve-root-ca.pem",
  --   server_name = "pve.example.org",
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
Proxmox API. Guest creation and deletion, configuration writes, snapshots,
backups, restores, disk and network changes and access administration remain
unimplemented until each action has its own exact-target approval and
read-back contract. A configured token or a passing fixture test does not
establish that a live cluster is reachable.
