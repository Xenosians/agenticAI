# V18-03 — Cross-Repository End-to-End Acceptance

V18-03 proves the real application path rather than treating direct tool calls
as end-to-end success.

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
- `GET /api/v1/jobs/:id`
- `POST /api/v1/jobs/:id/approve` only when explicit execution is enabled.

## Safety

The shipped mutation case verifies that the job reaches
`waiting_approval`, but has `allow_execute=false`. It is therefore never
executed by the shipped suite.

Provider-specific read cases are skipped unless the corresponding environment
input or integration is configured.

Passwords, cookies, and CSRF tokens are not written to the acceptance report.

Reports are written under:

`.runtime/acceptance/v18-03/`

and are not intended for Git.

## Repo gate

```bash
python scripts/v18_03_acceptance.py --mode repo
```

This compiles and checks the AI repository, runs the explicit capability
contract tests and audit, compiles/tests the Phoenix API contract, and compiles
the Nim/Karax frontend into `.runtime/v18_03/app.js`.

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
export V18_03_EMAIL='your-test-login@example.com'

cd /mnt/c/project/agenticaiPersonal/itsm-agent

python scripts/v18_03_acceptance.py \
  --mode live
```

Optional read cases become active when these are present:

```bash
export V18_03_ACCOUNT_USER_ID='...'
export V18_03_TICKET_KEY='...'
export V18_03_JIRA_PROJECT_KEY='...'
```

Palo Alto runs automatically only when `/api/v1/integrations` reports the
`palo_alto` integration as configured.

## One case only

```bash
python scripts/v18_03_acceptance.py \
  --mode live \
  --case workspace-project-info-read
```

## Combined gate

```bash
python scripts/v18_03_acceptance.py \
  --mode all
```

Do not pass `--execute-approvals` unless an acceptance manifest has been
reviewed and a specific case is explicitly marked `allow_execute=true`.
