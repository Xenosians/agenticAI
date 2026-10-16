# Local SMTP/IMAP mailbox lab

The account specialist has two separate mailbox tools: `mailbox_create(email)`
creates a real Maddy SMTP/IMAP account after approval; `mailbox_status(email)`
checks its credential and storage presence. Directory `account_create` still
creates a directory identity. Brevo remains the application's authentication,
completion notification, and project report mailer.

Start the isolated lab from the AI repository:

```sh
docker compose -f docker/mailbox-lab/compose.yaml up -d
```

The image is pinned to Maddy **0.9.6** and its verified image digest. The two
ports bind only to `127.0.0.1`: SMTP submission `1587`, IMAP `1143`. A dedicated
Docker bridge supports the loopback mappings; mail routing rejects public relay
and defines no remote delivery target. This lab uses plaintext
authentication only on these local test endpoints; it is not a public deployment.

Configure the shared local `.env`, then restart the AI service:

```dotenv
MAILBOX_LAB_ENABLED=true
MAILBOX_LAB_DOMAIN=itsm.test
MAILBOX_SMTP_PORT=1587
MAILBOX_IMAP_PORT=1143
```

Use the same domain and ports when starting Compose. The existing Phoenix
internal token, base URL, and credential encryption key must be configured.
No passwords belong in requests to the LLM or mailbox tool arguments.

Example request: **Create a local mailbox for alice@itsm.test.** After normal
approval, trusted code checks that neither credential nor storage exists,
generates a password, passes it to Maddy only through standard input, creates
IMAP storage, and hands the credential to Phoenix for encrypted persistence.
It then authenticates to SMTP, sends one message to the same address, and reads
its unique Message-ID through authenticated IMAP. Only that successful readback
produces a verified result. The encrypted vault record remains marked
`provisioned_unverified` because it was safely persisted before this readback;
the tool result carries the final verification evidence.

Creation never resets or replaces an existing mailbox. A timeout, partial
creation, persistence failure, or failed readback produces `outcome_unknown`
and disables retry. Inspect status and use an operator-led recovery; do not
retry creation automatically. The process lock serializes AI service creates
sharing the same runtime path; do not run competing manual provisioning.

`.test` addresses cannot receive public GitHub or Atlassian verification email.
Using this mailbox for those public account registrations needs a separately
configured public mail domain, DNS/MX, reachable TLS mail service, and provider
registration access. No public account registration or dependent workflow
result binding is implied by the local mailbox test.

## Verification evidence

The local provider acceptance on 2026-10-16 used the actual gateway, durable
approval store, MCP subprocess, Maddy server, and Phoenix endpoint. It verified
absence before creation, encrypted handoff, authenticated SMTP self-delivery
read back with IMAP, presence afterward, and replay after reopening the approval
store without a second create call. The private report is
`.runtime/mailbox-live-acceptance.json`; it explicitly marks LLM generation as
untested. This provider proof does not establish local-model or harness routing
accuracy, public delivery, or a public service registration workflow.

Upstream contracts:
[security release](https://github.com/foxcpp/maddy/releases/tag/v0.9.6),
[setup](https://maddy.email/tutorials/setting-up/),
[password stdin implementation](https://github.com/foxcpp/maddy/blob/v0.9.6/internal/cli/ctl/users.go),
[Docker management](https://github.com/foxcpp/maddy/blob/v0.9.6/docs/docker.md).
