# Agentic AI

Local-first AI runtime for the Agentic Developer Hub / ITSM platform.

This repository contains the persistent FastAPI AI service, conversational Primary Assistant, specialist runtime, local model backends, ToolGateway policy boundary, MCP integration, governed local process execution, LDAP/Samba AD integration, durable completion delivery, and durable approval execution state.

> Development prototype. Model output is never authorization.

## Current baseline

Implemented and proven in the current stack:

- FastAPI `/health` and `/ready`
- Phoenix durable-job execution through `/v1/jobs/execute`
- durable SQLite/WAL completion outbox
- conversational Primary Assistant for ordinary non-tool requests
- registered specialist routing
- Account and Access specialist/tool paths
- lazy/cached specialist model loading through `ModelRegistry`
- Qwen2.5-Coder developer specialist
- ToolGateway allowlists, schema checks, grounded-argument validation, and deterministic policy resolvers
- persistent MCP execution boundary
- Samba AD / LDAP development integration
- governed local process execution with `shell=False`
- read-only `process_exec` support for `pwd`
- approval-gated `workspace_mkdir`
- Phoenix job-scoped browser approval flow
- durable SQLite/WAL approval state
- idempotent approval replay
- pending approvals surviving FastAPI restart
- fail-closed handling for approvals left in an ambiguous `executing` state

The full browser path has been exercised as:

```text
Karax browser
    -> Phoenix durable job
    -> FastAPI AI runtime
    -> Primary Assistant / specialist
    -> ToolGateway
    -> governed process runner or MCP
    -> Phoenix durable result
    -> Karax browser
```

For mutations:

```text
browser request
    -> Phoenix job
    -> AI proposes validated mutation
    -> job enters waiting_approval
    -> browser approves by job_id
    -> Phoenix resolves trusted persisted approval_id
    -> AI executes exact persisted proposal
    -> result is durably recorded
```

The browser never supplies or authorizes the internal approval ID directly.

## Architecture

```text
Phoenix durable job
       |
       v
FastAPI
       |
       v
Orchestrator
   /         \
  v           v
Primary     Specialist
Assistant    Router
                 |
                 v
             Worker model
                 |
                 v
             ToolGateway
          /        |        \
         v         v         v
   ApprovalStore  MCP   Process Runner
       SQLite      |     pwd / mkdir
                   v
             DirectoryService
                   |
                   v
               LDAP/Samba

AI completion
    -> SQLite completion outbox
    -> authenticated Phoenix callback
    -> ACK
```

The Hub / Primary Assistant can answer ordinary requests directly. Specialists are used only when a domain-specific reasoning role or governed capability is useful.

## Models and specialists

Current model/runtime design is role-oriented rather than API-coupled to one checkpoint.

- Hub / Primary Assistant: local Ministral profile
- Account specialist: Qwen function-calling worker
- Developer specialist: Qwen2.5-Coder worker
- Access specialist: registered capability path, environment-dependent

`ModelRegistry` supports eager or lazy registration. Lazy specialist backends are loaded on first use, protected against duplicate concurrent first-load, and cached for later requests.

The broader target remains a generic `ModelProfile` / provider configuration layer so local, company-specific, or future remote models can be swapped without changing the frontend or Phoenix public API.

## Setup

Recommended:

- Linux / WSL2
- Python 3.12
- Conda
- Docker
- PyTorch / Transformers
- sufficient RAM / VRAM for the configured local models

```bash
conda activate qwen-infra
python -m pip install -r requirements.txt
cp .env.example .env
```

Configure the model paths and runtime settings in `.env` for your machine. Model weights should remain outside the source repository.

Example Hub settings:

```env
AGENTS_DIR=./subagents/agents

HUB_BACKEND=ministral
HUB_MODEL_PATH=/absolute/path/to/Ministral-3-3B-Instruct-2512
HUB_DEQUANTIZE_FP8=true

ACCOUNT_BACKEND=qwen-funccall
ACCOUNT_MODEL_KEY=qwen2.5-0.5b-funccall
ACCOUNT_MODEL_PATH=/absolute/path/to/qwen2.5-0.5b-funccall

ACCESS_ENABLED=false
```

Optional:

```env
HUB_OFFLOAD_FOLDER=/tmp/itsm-ministral-offload
PROCESS_WORKSPACE_ROOT=/approved/workspace/root
APPROVAL_STORE_PATH=.runtime/approvals.sqlite3
```

## Samba AD

```bash
docker compose up -d samba-ad
docker compose ps
python scripts/ldap_preflight.py
```

Development ports:

```text
LDAP  1389
LDAPS 1636
```

## Start AI

```bash
python -m uvicorn api.app:app \
  --host 127.0.0.1 \
  --port 8000
```

Healthy startup includes:

```text
[OUTBOX] Delivery worker started pending=0
[MCP] Persistent server started.
[API] AI runtime ready.
```

Readiness check:

```bash
curl -s http://127.0.0.1:8000/ready | python -m json.tool
```

Expected shape:

```json
{
  "status": "ready",
  "service": "itsm-ai"
}
```

## Durable completion outbox

Default database:

```text
.runtime/completion_outbox.sqlite3
```

Flow:

```text
AI result
  -> persist immutable completion payload
  -> callback Phoenix
  -> Phoenix validates job / attempt / terminal state
  -> ACK
  -> remove outbox entry
```

This protects the Python -> Phoenix completion boundary from temporary callback failure.

## Durable approvals

Default database:

```text
.runtime/approvals.sqlite3
```

Override with:

```env
APPROVAL_STORE_PATH=/path/to/approvals.sqlite3
```

Approval execution state machine:

```text
pending
   |
   v
executing
  /     \
 v       v
approved failed
```

Important semantics:

- the exact validated tool and arguments are persisted before execution
- `pending -> executing` is committed before MCP / OS side effects
- approved results are replayed instead of executing again
- failed approvals are terminal and are not automatically retried
- concurrent requests for the same approval are serialized
- pending approvals survive an AI restart
- an approval left `executing` after an ambiguous failure or crash is not reset to `pending`

That last rule is deliberate: the side effect may already have happened, so blind retry could duplicate a mutation.

## Tool safety

```text
Models propose.
Deterministic Python policy validates.
Trusted MCP / runner code executes.
```

Current enterprise and developer tools include:

| Tool | Type | Approval |
| --- | --- | --- |
| `account_status` | read | no |
| `check_access` | read | no |
| `unlock_user` | mutation | yes |
| `reset_password` | mutation | yes |
| `process_exec` with `pwd` | local read | no |
| `workspace_mkdir` | local mutation | yes |

The local process runner currently uses a narrow executable allowlist.

`pwd` policy:

```text
executable: pwd
args: []
risk: read
approval: no
```

`mkdir` policy is exposed through the typed `workspace_mkdir(directory_name)` tool. The deterministic runner only accepts one direct-child directory name under the approved workspace. Paths, flags, `..`, workspace escape, and unsupported characters are denied.

The execution primitive uses structured arguments:

```python
subprocess.run(
    [executable_path, *args],
    cwd=resolved_cwd,
    timeout=timeout_seconds,
    shell=False,
    capture_output=True,
    text=True,
)
```

Never execute raw model text using `os.system`, model-generated `shell=True`, `eval`, or equivalent unrestricted paths.

## Important files

```text
api/app.py

agent/approvals.py
agent/approval_store.py
agent/completion_outbox.py
agent/mcp_client.py

subagents/hub.py
subagents/core/orchestrator.py
subagents/core/primary_assistant.py
subagents/core/llm_router.py
subagents/core/runtime.py
subagents/core/tool_gateway.py
subagents/llm/registry.py
subagents/llm/qwen_coder_worker.py
subagents/agents/
subagents/prompts/

services/process_runner.py
services/directory/

tools/process.py
tools/workspace.py
tools/registry.py

mcp_server.py
```

## Tests and smoke checks

Compile the Python packages:

```bash
python -m compileall api agent subagents services tools
```

Run the test suite:

```bash
python -m pytest subagents/test -v
```

Useful integration checks include:

```text
hello
Is jdoe locked?
What is the current working directory?
Create a directory named demo_workspace
```

For the mutation case, verify that the directory does not exist before approval and appears only after the browser/Phoenix approval path completes.

Approval reliability checks should cover:

```text
same approval ID twice
    -> first executes
    -> later call replays stored result
    -> no duplicate mutation

pending approval
    -> stop FastAPI
    -> restart FastAPI
    -> approve same ID
    -> execution succeeds
```

## Current gaps

The current baseline is functional, but several architectural targets remain open:

- durable Phoenix-owned multi-turn conversation/history context
- generic `ModelProfile` / broader `ModelManager` provider configuration
- additional narrow developer read capabilities such as directory listing, file reads, and Git inspection
- explicit durable recovery/requeue of expired `processing` Phoenix jobs
- reconciliation workflow for approvals left in ambiguous `executing` state
- runtime/build configuration for frontend service URLs
- WebSocket / PubSub user delivery; polling is the current working fallback
- broader plugin/provider integrations such as GitHub, Jira, Slack, RAG, and endpoint tooling

The highest-priority reliability caveat is still Phoenix processing-lease recovery: clearing local worker state alone does not durably requeue an expired `processing` job.

See the cross-repository project handoff / SRS for the full architecture baseline.
