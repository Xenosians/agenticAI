# Cross-Repository End-to-End Acceptance

This suite verifies authenticated requests through Phoenix, the AI service and
governed tools. It checks real tool results and persisted job state; browser
rendering and exhaustive tool coverage need separate evidence.

## Path under test

Browser-equivalent HTTP client
→ Phoenix authentication / CSRF
→ Phoenix chat
→ Phoenix durable job
→ AI `/v1/jobs`
→ Hub / specialist
→ SemanticGuard / ToolGateway
→ provider
→ AI completion callback
→ Phoenix durable job state.

The live runner uses the same Phoenix endpoints used by the Nim/Karax frontend:

- `POST /api/v1/auth/login`
- `POST /api/v1/chats`
- `POST /api/v1/jobs`
- `GET /api/v2/jobs/:id`
- `GET /api/v2/chats/:id/history`
- `POST /api/v2/jobs/:id/approve` only when explicit execution is enabled.
- `POST /api/v2/jobs/:id/reject` for the shipped safe human-rejection case.

Reads and approval responses must satisfy the canonical backend JSON schemas.
Use `--job-api-version 1` only for legacy contract checks; job creation continues
to use the v1 endpoint with either option.

Every selected live case must also appear in history under the canonical chat
identifier. The submitted user turn, visible answer or error, terminal state and
persisted proposal must match the durable job. A successful job GET with missing
or stale chat history cannot pass. History checks validate the live response
against the job; there is currently no separate backend history JSON schema.

## Safety

The shipped mutation proposal case verifies that the job reaches
`waiting_approval`, but has `allow_execute=false`. It is therefore never
executed by the shipped suite.

A separate shipped rejection case creates another pending proposal and sends the
explicit v2 reject decision. It requires `allow_reject=true`, verifies the durable
`rejected -> denied` projection, verifies that no tool result exists, reopens the
terminal job, and proves both later approve and reject submissions receive HTTP 409.
Rejection is therefore exercised without executing the proposed mutation.

For a separately reviewed disposable fixture, copy the case manifest, set only
the intended case's `allow_execute=true`, and pass both `--case` and
`--execute-approvals`. Set `expected_post_approval_status` explicitly if testing a
failure or unresolved outcome; successful execution defaults to `completed`.
Review fixture cleanup separately: `--cleanup-chat` archives the chat and does
not undo a tool's effects.

An enabled execution case checks the original job, attempt, owner, chat,
specialist and exact proposal binding against the approval response and durable
GET. The stored `approval_execution` must identify the original approval and
contain a consistent processed protocol/authority/execution outcome. Completed
actions additionally require a successful structured tool result and visible
answer. The runner reopens the terminal job, submits one deliberate approval
replay, requires the public API's HTTP 409 terminal-state rejection, and verifies
that another GET preserves the terminal result. It does not retry ambiguous
approvals.

Reports retain the original job and approval identifiers and a canonical
proposal SHA-256 instead of raw arguments or tool results. A rejected replay and
unchanged public state do not independently prove the number of external side
effects or that inference was never rerun. Those evidence flags remain false;
provider/audit tracing is required to establish them. Mock runner tests verify
these acceptance checks, not model quality or live execution.

Provider-specific read cases are skipped unless the corresponding environment
input or integration is configured. Skips make the selected suite incomplete
and return a failure status, so missing fixtures cannot produce a passing gate.

Passwords, cookies, and CSRF tokens are not written to the acceptance report.

Reports are written under:

`.runtime/acceptance/cross-repo/`

and are not intended for Git.

## Repo gate

```bash
python scripts/cross_repo_acceptance.py --mode repo
```

This compiles and checks the AI repository, runs the explicit capability
contract tests and audit, compiles/tests the Phoenix API contract, and compiles
the Nim/Karax frontend into `.runtime/cross-repo-acceptance/app.js`.

## Start services for live acceptance

AI:

```bash
conda activate qwen-infra
cd /mnt/c/project/agenticaiPersonal/itsm-agent
set -a
source .env
set +a
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Phoenix:

```bash
cd /mnt/c/project/agenticBackend/itsm_backend
set -a
source .env
set +a
iex -S mix phx.server
```

The frontend server is not required for the automated HTTP-path runner because
the runner deliberately drives the same backend endpoints used by the browser.
The repo gate still compiles the actual Nim/Karax application.

## Live run

Set the test account email. The password can be entered securely at the prompt:

```bash
export ACCEPTANCE_EMAIL='your-test-login@example.com'

cd /mnt/c/project/agenticaiPersonal/itsm-agent

python scripts/cross_repo_acceptance.py \
  --mode live
```

For unattended runs, set `ACCEPTANCE_PASSWORD` in private local configuration and
pass `--non-interactive`. Do not commit credentials. Existing `V18_03_*` variables
remain supported as fallback aliases; `ACCEPTANCE_*` takes precedence.

Repository paths can be overridden with `ACCEPTANCE_BACKEND_REPO` and
`ACCEPTANCE_FRONTEND_REPO`.

Optional read cases become active when these are present:

```bash
export ACCEPTANCE_ACCOUNT_USER_ID='...'
export ACCEPTANCE_TICKET_KEY='...'
export ACCEPTANCE_JIRA_PROJECT_KEY='...'
```

OpenWrt and Palo Alto read cases require their corresponding integration to be
reported as configured by `/api/v1/integrations`.

## One case only

```bash
python scripts/cross_repo_acceptance.py \
  --mode live \
  --case workspace-project-info-read
```

## Combined gate

```bash
python scripts/cross_repo_acceptance.py \
  --mode all
```

Do not pass `--execute-approvals` unless an acceptance manifest has been
reviewed and a specific case is explicitly marked `allow_execute=true`.

## Ambiguous mutation security fail-closed proof

An ambiguous mutation passes this security case only when durable state proves
that authority was never created:

```text
status = failed
selected_agent = null
proposed_tool = null
result = null
durable error = governed-plan refusal
```

The runner does not treat arbitrary runtime failures as security success.
Provider timeouts, delegated specialists, tool proposals, approvals, or
application results fail this case.

Security invariant:

```text
ambiguous required mutation target
-> no unique trusted grounding
-> no valid governed execution plan
-> no authority delegation
-> no proposal
-> no approval
-> no tool execution
```

The `failed` label is state-machine semantics; the security property is the
absence of delegated authority and executable evidence.
