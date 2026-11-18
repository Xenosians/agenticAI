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
attention is named most urgent first: `not_running`, `disk_nearly_full` (90 %
or more of a measured disk), `no_backup_job` and `locked` (mildest, since
every backup run locks its guest). The storage pools
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

## Backups and physical disks

`proxmox_backups` lists every guest's newest backup across the storages that
hold backups (`GET /nodes/{node}/storage?content=backup&enabled=1`, then each
storage's `content?content=backup`; a shared storage is read once). Each guest
shows the age, size and storage of its newest copy, whether that copy is
protected, encrypted and verified, its backup count and whether a backup job
covers it. What needs attention, most urgent first: `never_backed_up`,
`missed_run`, `backup_stale` and `no_backup_job`. A guest in an enabled,
scheduled backup job is judged against that job: Proxmox lists the guests a
job includes (`GET /cluster/backup/{id}/included_volumes`, `Sys.Audit`) and
computes when its schedule fires (`GET /cluster/jobs/schedule-analyze`, any
user; nothing is parsed by Janus), and `missed_run` means the newest copy is
older than the job's last run that was due at least
`missed_run_grace_hours` (12) ago, so a run still in progress is not judged.
Each guest carries that time as `last_scheduled_run`. A guest without such a
job (none, disabled, unschedulable, or guests the token cannot list) is
judged by age: `backup_stale` means the newest copy is older than
`stale_after_days` (8). A guest in two jobs is judged by the more recent due
run. When a backup storage
cannot be read the listing is marked incomplete with the reason, and no guest
is then called never backed up, overdue or late. Volume ids, notes and paths never
leave the AI service. Listing backups needs `Datastore.Audit` on each backup
storage; it cannot start, delete or restore a backup.

`proxmox_node_disks` lists one node's physical disks
(`GET /nodes/{node}/disks/list`): device, type, model, size, SMART health
(`passed`, `failed`, `smart_disabled` or `unknown`), the share of rated life
used (100 − Proxmox's wearout) and what uses the disk. A failed SMART check is
`smart_failed`; 90 % or more of rated life used is `worn_out`. Serial numbers,
WWNs and by-id links are never returned. The read needs `Sys.Audit` on the node.

The Operations dashboard reads both through
`GET /internal/v1/proxmox/health` (`proxmox-health.v1`): the backups and each
node's disks, every part a typed read result so one node's refused disk read
never hides the rest. It is read once a minute while the page is open, apart
from the ten-second overview.

`proxmox_disk_smart` reads one listed disk's SMART data
(`GET /nodes/{node}/disks/smart?disk=<devpath>`, `Sys.Audit`; smartctl runs
on the node, so chat asks for one disk at a time and the dashboard never
does). The device must be one `/nodes/{node}/disks/list` reports on that
node; a bare name such as `sda` means `/dev/sda`. Only fixed numbers leave
the AI service: temperature, power-on hours, power cycles, unsafe shutdowns,
media errors, reallocated, pending and offline-uncorrectable sectors, spare
capacity and its threshold, and wear. Serial numbers, WWNs, vendor rows and
the raw smartctl text never do. Media errors, moved or unreadable sectors
and spare capacity below its threshold are named: they are early signs of
failure even while SMART reports passed. It never runs a self-test.

## Guest snapshots

`proxmox_guest_snapshots` lists one exact guest's snapshots
(`GET /nodes/{node}/{type}/{vmid}/snapshot`, after the guest's type is
confirmed from the inventory; `VM.Audit`), newest first: name, when it was
taken, its parent, whether it saved the guest's RAM (VMs only; a container
snapshot never does) and which snapshot the current state follows. Proxmox's
synthetic `current` entry ("You are here!") is the guest's position, never a
snapshot. A snapshot whose time Proxmox did not record has an unknown age, not
1970. Snapshot descriptions are operator free text and never leave the AI
service. It cannot create, delete or roll back a snapshot. A snapshot lives on
the guest's own storage and is not a backup.

## Node network interfaces

`proxmox_node_network` lists one node's interfaces (`GET /nodes/{node}/network`;
the loopback is never listed): bridges and bonds first, then physical ports,
with each bridge's or bond's ports, which bridge or bond each port belongs to,
the bond mode, MTU, VLAN awareness, whether it starts at boot and how IPv4 is
configured (and only whether an address is set, never the address). A port a
bridge names that is not configured on the node is called out. Addresses,
netmasks, gateways, comments and alternative names never leave the AI service;
an alternative name can embed the adapter's MAC address.

Proxmox fills an interface's `active` flag from the network configuration it
last parsed, so it can lag the real link (Proxmox bug 8148); the reading says
"Proxmox reports it active/inactive", and an interface without a report is
unknown, never down. It cannot change the network.

## High availability

`proxmox_ha_status` reads `GET /cluster/ha/status/current` (`Sys.Audit` on
`/`): whether the cluster has quorum, whether HA is in use (any manager,
node or resource entry), the fencing state, the node running the HA manager,
each node's HA state and each HA resource (`vm:<vmid>` or `ct:<vmid>`) with
its state and the state the operator requested. Proxmox reports the
manager's and nodes' state only inside a status text that also carries the
node's local time ("pve-a (active, watchdog active, Mon …)"); the service
maps the state word to a fixed one (`active`, `idle`, `maintenance`,
`disarmed`, `not_responding`, `clock_drift`, `waiting_for_lock`,
`lost_lock`, `restarting`, `shutting_down`, `unreadable`) and never passes
the text on. A word it does not know is `unknown`, which counts as a
problem; so do a resource in `error`, `fence`, `recovery`, `freeze`,
`deleting` or `not found`. Resource comments are operator free text and
stay in the service. It cannot change HA, migrate or fence anything.

## Node maintenance

`proxmox_node_maintenance` reads one exact node's maintenance state, each
part from its own read (all allowed by `Sys.Audit`):

| Part | Read | Kept |
| --- | --- | --- |
| Version, kernel, boot | `GET /nodes/{node}/status`, `GET /nodes/{node}/apt/versions` | Proxmox VE version, running kernel, newest installed kernel (a newer one means a reboot is due), EFI or legacy boot and Secure Boot |
| Services | `GET /nodes/{node}/services` | Running count; services not running with their state; services turned off; corosync stopped is expected only when `GET /cluster/status` shows no cluster |
| Certificates | `GET /nodes/{node}/certificates/info` | File, role, whether pveproxy serves it, issuer kind (Proxmox CA, self-signed, other CA), key type and size, days left |
| Updates | `GET /nodes/{node}/subscription`, `GET /nodes/{node}/apt/repositories` | Subscription state; which standard repositories are on or off; repository error count |

Issues, most urgent first: `service_not_running`, `certificate_expired` (the
served certificate), `certificate_expiring` (within 30 days),
`enterprise_repository_without_subscription`, `no_update_repository`,
`reboot_for_newer_kernel`, `test_repository_enabled`. A part Proxmox refuses
(or an older release does not have) is listed as unreadable and is unknown,
never healthy. Certificate subjects, issuers, SANs, fingerprints and PEM, the
subscription key and server id, and repository file paths never leave the
service. Pending package updates need `Sys.Modify` (`GET apt/update`), which
Janus does not hold (it can also change node settings), so they are not read;
the reading says so by name on every answer. Nothing is installed or rebooted.

## Node load history

`proxmox_node_load` reads one exact node's own history from Proxmox
(`GET /nodes/{node}/rrddata`, `Sys.Audit`) over the last hour, day, week or
month (a day when the user names none): the `AVERAGE` series for typical
values and the `MAX` series for peaks. It reports average and peak CPU, IO
wait and load average; memory used on average and at peak and the least
memory left available; peak swap; the highest PSI stall shares (CPU, IO and
memory, "some" and "full"); network rates; root disk use. Minutes Proxmox has
no sample for are gaps, never zero, and the reading says how many samples
were present. Issues, most urgent first: `memory_low` (available memory
under 10 % of the total), `memory_stalls` (any full memory stall),
`io_stalls` (full IO stall 10 % or more), `cpu_busy` (peak CPU 90 % or more)
and `swapping` (swap use 10 % or more of the swap space). It is history, not
a forecast.

### How answers show a reading

Every read above declares its formatted reading complete
(`synthesis_evidence: "answer"`), so the answer never reinterprets raw
fields. A one-line reading is the answer as written. A reading of several
lines with six or more figures (load, maintenance, backups and the like) is
shown **exactly as the tool wrote it**, after one or two sentences from the
model that contain no figures (no digits, no number words, no percent sign;
owner's rule of 16 November, after a live answer gave a peak as an average).
A summary that fails that check, or a failed model call, leaves the exact
reading alone; nothing is retried. Other readings are narrowed to the
question by synthesis as before.

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
is never retried automatically, and the guest must be checked first.

The Operations dashboard offers the same four actions per guest (only those
that fit its current state). A button only proposes the action through the
same governed path a chat request uses; nothing runs until the owner confirms
the server's description of the exact target. Phoenix records each action
once per submission, lets only its owner decide it, and never re-sends an
approval whose outcome was not confirmed.

## Guest changes

Eleven governed tools change guests: `proxmox_guest_set_resources` (cores,
memory), `proxmox_guest_resize_disk` (grow only), `proxmox_guest_set_options`
(tags, start at boot), `proxmox_snapshot_create`, `proxmox_snapshot_delete`,
`proxmox_snapshot_rollback`, `proxmox_backup_now`, `proxmox_backup_restore`
(always into a NEW guest), `proxmox_guest_clone`, `proxmox_guest_create` (an
LXC container from a template) and `proxmox_guest_delete` (a stopped guest,
disks purged). They follow the power-action pattern:

1. **Trusted policy freezes the target** (type, name and, for configuration
   changes, the configuration digest) and refuses a locked guest, a template,
   a running guest to delete, or a VMID that is not free. When the owner names
   no VMID for a new guest, Proxmox's own next free VMID is frozen.
2. **The owner approves the exact change.** Delete, snapshot delete, rollback
   and restore are destructive: the approval also needs the frozen guest name
   (for a restore, the new guest's name) typed by the owner. The proposal
   carries that text as `confirmation_text`; the AI approval authority checks
   the typed `confirm_name` before anything runs (dashboard actions are checked
   by Phoenix too). A missing or wrong name leaves the approval pending.
3. **One request, then receipts.** Janus re-reads the guest, refuses a changed
   identity or configuration digest, sends exactly one request, follows the
   Proxmox task and reads the result back (new values, snapshot listed or gone,
   rollback parent, new backup volume, new guest listed).
4. **Unknown is never retried.** Once Proxmox accepted the request,
   `retry_safe` is false. A lost answer, a task that does not finish in time,
   or a read-back that cannot be made is `outcome_unknown`, never `failed`:
   check the guest before deciding again.

Privileges: the token needs each action's own Proxmox privilege (for example
`VM.Config.CPU`/`VM.Config.Memory`, `VM.Config.Disk`, `VM.Snapshot`,
`VM.Snapshot.Rollback`, `VM.Backup`, `VM.Clone`, `VM.Allocate`,
`Datastore.AllocateSpace` on the target storage); a refusal is reported as
`denied` with nothing changed.

The live tool matrix runs the change cases when the acceptance profile also
provides `ACCEPTANCE_PROXMOX_TEMPLATE` (an existing container template
volume), `ACCEPTANCE_PROXMOX_DISK_STORAGE` and
`ACCEPTANCE_PROXMOX_BACKUP_STORAGE`. Reversible changes run on the lab guest
and are undone in the same run; everything that makes a guest works on a
disposable container the run creates, and every guest the run made is
deleted again. Destructive approvals send the name the proposal shows.

## Whose Proxmox and Tailscale

Each Janus account connects its own Proxmox and Tailscale in **Plugins >
Servers** (owner decision, 17 November: "user dependant not machine
dependant"). The AI service stores the link in the account's extension store
with its secret encrypted, lists it without the secret, and uses it for that
account's chat, tools, dashboards and pre-approval checks
(`services/provider_links.py`). An account without a link uses this machine's
profile link as its default; with neither, the read says to add a link in
Connections.

The account crosses every hop as data the browser cannot choose: Phoenix
takes it from the session and passes `owner` to the dashboard reads; the AI
sends it to the MCP subprocess as request metadata (`janus.itsm/owner`, never
a model argument), where each tool call builds that account's client and
closes it afterwards; an approval records the proposing account, and the
approved action runs with that account's link, whichever request decides it.

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
