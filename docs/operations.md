# Platform operations

The operations specialist covers the ITSM platform's own runtime: its Docker
containers, the health endpoints of the backend and AI services, and
notifications to registered application users. It is separate from the
developer specialist, which inspects a developer workspace compose project and
runs git, build and test commands.

## Read tools

| Tool | Reads | Notes |
|---|---|---|
| `container_status` | Docker Engine `GET /containers/{name}/json` | State, health, start time, uptime, exit code, restart count and image of every allow-listed container, or of one named container. A missing allow-listed container is reported as `not_found`, never hidden. |
| `container_logs` | Docker Engine `GET /containers/{name}/logs` | The last 1-200 lines (default 50) of one allow-listed container, stdout and stderr together, demultiplexed, ANSI-stripped and redacted with the shared rules in `services/log_redaction.py`. At most 16 KiB of text is returned; older output is cut at a line boundary and the result says so. |
| `service_health` | `GET` on each configured health URL | Up/down, HTTP status and latency per service. Only the status line is read; redirects are not followed; no credentials, cookies, proxy settings or internal tokens are sent. |

All three are `risk: read` without approval. They never start, stop, restart or
exec into containers and never change configuration.

### Why the Docker Engine API

The tools talk to the Engine API over the configured unix socket with `httpx`
instead of running the `docker` CLI through the trusted process runner:
responses are structured JSON (no output parsing), there is no subprocess, and
the code issues only fixed GET requests for names that are already on the
allowlist. Containers are inspected one by one by exact name, so containers
outside the allowlist are never listed or returned.

Access to the Docker socket is powerful: the AI service user must be in the
`docker` group, and only these fixed reads are implemented.

## Notifications

`notify_user` (`risk: medium`, approval required) sends one plain-text email to a
registered application user through the backend's durable mail outbox.

- Arguments: `email` and `message` are exact values from the user; `subject` is
  optional. The message is never rewritten.
- Before approval, trusted policy validates the request and freezes a
  `dedupe_key` (a hash of recipient, subject, message and a per-request nonce).
  The model never supplies it. Replaying the same approval reuses the key, so the
  backend queues the message at most once; a later identical request is a new
  notification.
- The approval text shows the recipient and the exact message.
- The AI posts to `POST {PHOENIX_BASE_URL}/api/internal/v1/notifications` with
  `x-internal-token` (HTTPS, or HTTP on localhost only). Phoenix resolves the
  recipient from its own user store (only active, verified users; an unknown,
  unverified or disabled address is `404 recipient_not_found`), takes the stored
  address and display name, sanitizes the text, and enqueues one outbox record of
  kind `user_notification`. It returns `202 {queued: true, outbox_id}`.
- The result says "queued for delivery", never "delivered": the outbox worker
  owns delivery and retries, and a delivery failure never fails the job.
- A backend timeout or 5xx after the request was sent returns
  `status: outcome_unknown` (the approval stays in reconciliation instead of
  retrying). A connection refused before sending is a plain error. A reused key
  with different content is refused (`409 dedupe_conflict`).

## Configuration

| Variable | Meaning | Lab example |
|---|---|---|
| `OPERATIONS_CONTAINERS` | Comma-separated exact container names the tools may read. Empty disables the container tools ("not configured"). | `itsm-git-lab,itsm-mailbox-lab,agentic-openwrt-lab,itsm-surrealdb,itsm-samba-ad` |
| `OPERATIONS_DOCKER_SOCKET` | Docker Engine unix socket. | `/var/run/docker.sock` (default) |
| `OPERATIONS_HEALTH_URLS` | Comma-separated `name=url` pairs (at most 10, http/https, no credentials, query or fragment). Empty derives `phoenix` = `PHOENIX_BASE_URL` + `/api/health` and `ai` = `http://AI_HOST:AI_PORT/health` (a wildcard bind address is probed through loopback). | `phoenix=http://127.0.0.1:4000/api/health,ai=http://127.0.0.1:8000/health` |
| `OPERATIONS_TIMEOUT_SECONDS` | Per-request timeout for Docker and health probes (max 10). | `3` (default) |

## Limits

- `notify_user` cannot reach unregistered addresses and does not create
  mailboxes; in the lab, `.test` recipients are delivered by the local mail relay.

- Health is what each endpoint reports at that moment; a 2xx status means "up"
  and nothing deeper is verified.
- Redaction is best effort for common secret forms (assignments such as
  `password=...`, bearer tokens, credentials in URLs); it is not a complete
  secret scanner.
