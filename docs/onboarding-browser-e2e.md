# Browser onboarding acceptance

`scripts/onboarding_browser_e2e.py` checks one declared adapter through the real
frontend: registration, verification email, sign-in, exact account approval,
a second exact Jira approval on the same durable job, saved history and a
completion message read independently through IMAP. Source/profile drift or any
unverified step leaves the report **incomplete** (exit 2). Admission and HTTP
acknowledgements never establish ITSM completion.

Use a separate run for local defaults and each outbound adapter (`http`, `pi`,
`opencode`). This workflow covers account/Jira/mail; it is not the all-tools
matrix or a full MVP certificate. The model/profile fingerprint binds declared
selection; retain actual inference telemetry separately for adapter evidence.

## Prerequisites

- The current backend, AI and rebuilt frontend are running. The configured Jira
  project is explicitly authorized for a disposable ticket, and directory account
  creation uses a real configured provider.
- The Maddy lab is running with an explicitly enabled lowercase `.test` domain in
  `--lab-env` (default: the AI `.env`). The runner reads `MAILBOX_LAB_ENABLED`,
  `MAILBOX_LAB_DOMAIN` and optional `MAILBOX_IMAP_PORT`.
- Phoenix's non-production lab-mail relay uses the same lab domain and has its
  SMTP sender credentials configured. Public mail retains the existing Brevo
  transport; this test never sends a reserved `.test` address through Brevo.
- Playwright Chromium is installed in the selected project Python runtime.

The driver creates a fresh lab mailbox using the existing Maddy admin allowlist
and provisioning lock, after recording intent and confirming neither credentials
nor IMAP storage exists. This bootstrap happens before an app owner exists and
is explicitly **setup, not mailbox-tool execution credit**. It does not download,
start or reconfigure the mail service. A `.test` mailbox cannot receive public
GitHub/Atlassian verification messages.

## Run

Every attempt requires two distinct, new paths: a non-secret JSON report and a
private reconciliation `.env`. Use an ignored directory with appropriate local
access control. Files are exclusively created with mode 0600 where supported;
Windows/filesystem access controls still determine who can read them.

```sh
python scripts/onboarding_browser_e2e.py \
  --adapter local --project "$JIRA_PROJECT" \
  --report .runtime/acceptance/local-onboarding.json \
  --private-env .runtime/acceptance/private/local-login.env

python scripts/onboarding_browser_e2e.py \
  --adapter http --project "$JIRA_PROJECT" \
  --connection-env .runtime/acceptance/private/model.env \
  --report .runtime/acceptance/outbound-onboarding.json \
  --private-env .runtime/acceptance/private/outbound-login.env
```

Optional arguments: `--frontend-url`, `--backend-url`, `--lab-env`, `--headed`,
`--reviewed-workflow`, `--timeout` (default 300 seconds per workflow/mail phase). The timeout is a
stated budget, not a retry-until-green setting. Measure local latency and record
failure rather than repeatedly increasing it.

Use `--reviewed-workflow` to cover the composer's **Required operations** path.
The driver selects account creation and ticket creation through visible controls,
fills the exact fixture targets and chooses the supported success dependency.
It verifies the actual submitted contract, server acknowledgement/hash, durable
job after reload, both approvals and the final history binding. A missing or
changed binding stops before the next approval. Without this flag the original
free-text path is checked; it does not establish reviewed-plan completeness.
Neither mode inserts a missing model operation or bypasses any approval.

For outbound scope the private connection file explicitly declares:

```dotenv
MODEL_BACKEND=http
MODEL_URL=https://your-configured-provider.example/v1
MODEL_ID=your-model-id
MODEL_API_KEY=your-private-key-if-required
MODEL_API_FORMAT=openai
MODEL_PROVIDER_ID=your-configured-provider-id
```

`MODEL_BACKEND` must match `--adapter`. For HTTP, the URL is required and API
format may be `openai` or `anthropic`. Pi/OpenCode require their admitted
provider/model identifier and installed harness. Optional `MODEL_OPTIONS_JSON`
contains only the product's supported inference options, such as admission
limits; it cannot replace backend, endpoint or credential fields. Local scope
uses configured defaults and accepts no connection file.

The new owner admits this connection through its authenticated public extensions
API. No other account's credentials are read, decrypted or copied. An
`AcceptanceProfileScope` pins the declared adapter, clears conflicting
non-networking overrides, preserves the existing networking policy and restores
prior selection only when safe. No probe is necessary: the actual governed
workflow must establish successful inference and tool execution.

## Evidence and reconciliation

The report is created **before** mailbox setup or any product mutation and is
checkpointed before registration, connection selection, request dispatch and
each exact approval. Observed chat/submission/job/approval identifiers survive
failures. The separate private file contains the generated app/mailbox login
credentials; the JSON report contains no password or recipient address.

Pass requires the exact requested account fields, verified/persisted account
receipt, distinct Jira approval, matching verified Jira receipt, both saved
approval executions, history round-trip, WebSocket frames and an actual inbox
message with RFC Message-ID and matching job/account/ticket identifiers. An
outbox `sent` state or provider acceptance alone cannot pass inbox delivery.

There is no automatic resume, re-registration, replacement ticket, credential
copy, retry of an unknown side effect or destructive fixture cleanup. The new
mailbox, application account, admitted encrypted connection and any real ITSM
results remain available for operator inspection. Reconcile an interrupted run
using its original report/private file before deciding on further actions.

Focused offline safeguards:

```sh
python -m pytest tests/unit/scripts/test_onboarding_browser_e2e.py \
  tests/unit/scripts/test_workflow_acceptance.py \
  tests/unit/scripts/test_acceptance_profiles.py
```
