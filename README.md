# Agentic AI

Local-first AI runtime for the Agentic Developer Hub / ITSM platform.

This repository contains the Python/FastAPI AI service used behind the Phoenix application boundary.

The runtime provides:

- conversational Hub / Primary Assistant inference
- specialist-agent routing
- configurable logical model profiles
- centralized model lifecycle management
- serialized GPU admission
- deterministic ToolGateway policy enforcement
- persistent MCP execution
- LDAP / Samba AD integration
- governed developer-workspace operations
- durable approval execution state
- durable completion callback delivery

> Model output is never authorization.

---

# Runtime boundary

The browser does not call this service directly.

The intended application path is:

```text
Browser
   |
   v
Phoenix
   |
   | authenticated internal request
   v
FastAPI AI service
   |
   v
Hub / specialists
```

Phoenix owns the public application boundary and durable user-facing job state.

FastAPI owns AI runtime state, model execution, approval execution safety, and the durable completion outbox.

---

# Current runtime architecture

```text
FastAPI lifespan
    |
    +-- Settings
    |
    +-- CompletionOutbox
    |
    +-- Phoenix HTTP client
    |
    +-- MCPRuntime
    |
    +-- ApprovalStore
    +-- ApprovalManager
    |
    +-- ModelManager
    +-- GpuScheduler
    +-- InferenceCoordinator
    |
    +-- ToolGateway
    |
    +-- build_hub(...)
            |
            +-- AgentRegistry
            |
            +-- LLMRouter
            |
            +-- PrimaryAssistant
            |
            +-- AgentRuntime
            |
            +-- Orchestrator
```

Runtime resources are explicitly owned by the FastAPI lifespan.

`build_hub(...)` only assembles the reasoning graph. It does not create process-wide runtime resources.

---

# Request flow

Ordinary conversational requests:

```text
Phoenix
   |
   v
FastAPI
   |
   v
Orchestrator
   |
   v
LLMRouter
   |
   +-- no specialist needed
           |
           v
     PrimaryAssistant
```

Specialist requests:

```text
Phoenix
   |
   v
FastAPI
   |
   v
Orchestrator
   |
   v
LLMRouter
   |
   v
AgentRuntime
   |
   v
specialist model
   |
   v
structured tool proposal
   |
   v
ToolGateway
   |
   +-------------------------+
   |                         |
   v                         v
read / allowed          approval required
   |                         |
   v                         v
MCPRuntime              ApprovalManager
   |                         |
   v                         |
MCP subprocess              |
   |                         |
   +-------------+-----------+
                 |
                 v
          trusted execution
```

---

# Model architecture

Models are configured through logical profiles.

Public APIs and specialist roles do not depend directly on model brands or checkpoint paths.

Example:

```text
hub-main
    |
    v
MODEL_PROFILES
    |
    v
ministral backend
    |
    v
local checkpoint
```

A specialist definition references only a logical model key:

```yaml
---
name: account-specialist
description: Handles Active Directory account operations.
model: qwen2.5-0.5b-funccall
tools:
  - account_status
  - unlock_user
  - reset_password
---
```

The model profile itself is configured separately.

Current model runtime:

```text
logical model key
      |
      v
ModelManager
      |
      v
ModelRegistry
      |
      v
backend factory
      |
      v
LLM backend
```

`ModelManager` handles:

- configured profile discovery
- lazy backend creation
- backend caching
- explicit loading
- explicit unloading
- model diagnostics

---

# GPU scheduling

All current model inference goes through:

```text
InferenceCoordinator
        |
        v
GpuScheduler
        |
        v
ModelManager
        |
        v
LLM backend
```

The current MVP scheduler provides:

- one admitted model operation at a time
- priority ordering for queued work
- FIFO behavior within equal priority
- non-preemptive execution
- blocking inference outside the asyncio event loop
- cancellation safety that does not release GPU admission while native inference is still executing

Current priorities include:

```text
STARTUP
SPECIALIST
HUB_ROUTING
PRIMARY_RESPONSE
```

The scheduler is intentionally a stable seam for later:

- VRAM observation
- HOT / WARM / COLD model residency
- measured eviction
- load latency tracking
- queue-wait metrics
- TTFT
- tokens/sec
- concurrency tuning
- validated inference overlap

Those are feature/scaling improvements, not reasons to change orchestration APIs.

---

# Specialists

Specialists are Markdown definitions with YAML front matter.

Current specialists:

```text
account-specialist
access-specialist
developer-specialist
```

Every specialist must explicitly define:

- name
- description
- logical model profile
- allowed tools

There is no implicit fallback model.

## Account specialist

Current responsibilities include:

- account status
- lock state
- account unlock proposal
- password reset proposal

Tools:

```text
account_status
unlock_user
reset_password
```

## Access specialist

Current responsibility:

- resource / authorization checks

Tool:

```text
check_access
```

## Developer specialist

Current governed workspace capabilities include:

```text
process_exec
workspace_mkdir
workspace_read_text
workspace_git_status
```

The developer specialist does not receive unrestricted shell access.

---

# Tool security

The core rule is:

```text
Models propose.
Trusted Python validates.
MCP / trusted providers execute.
```

The model does not authorize itself.

`ToolGateway` validates:

- specialist tool allowlists
- tool existence
- grounded arguments
- deterministic policy results
- risk classification
- approval requirements

Identity-sensitive fields such as `user_id` must originate literally from the original user request when required by tool policy.

Example:

```text
User:
Is jdoe locked?

Model proposal:
account_status(user_id="jsmith")

Result:
denied
```

---

# Tool registry

`tools/registry.py` is trusted capability and policy metadata.

It does not execute tools directly.

It owns information such as:

- description
- parameter schema
- risk
- approval requirement
- grounded arguments
- deterministic policy resolver
- trusted result formatter
- trusted approval formatter

Actual execution crosses MCP.

This keeps the generic `AgentRuntime` free of tool-name-specific branches.

Adding a new tool should not require editing `AgentRuntime`.

---

# MCP boundary

FastAPI owns one `MCPRuntime`.

`MCPRuntime` owns the persistent MCP subprocess connection.

Inside the MCP subprocess:

```text
MCPServer
   |
   +-- DirectoryService
   |
   +-- process runner
   |
   +-- workspace tools
```

The MCP subprocess explicitly owns one `DirectoryService`.

There is no hidden global directory-service cache.

Directory backend selection is configuration-driven:

```env
DIRECTORY_BACKEND=mock
```

or:

```env
DIRECTORY_BACKEND=ldap
```

Current implementations:

```text
MockDirectoryService
LdapDirectoryService
```

---

# LDAP / Samba AD

Current LDAP support includes:

- account lookup
- enabled state
- lock state
- direct group-based access checks
- approved account unlock

Separate read and mutation credentials are supported.

Password reset against real Active Directory remains disabled.

Local Samba AD can be used as the development LDAP environment.

Example preflight:

```bash
python scripts/ldap_preflight.py
```

---

# Governed process execution

Native process execution is controlled by deterministic Python policy.

The current executor uses:

```python
subprocess.run(
    [executable_path, *args],
    cwd=str(resolved_cwd),
    capture_output=True,
    text=True,
    timeout=timeout_seconds,
    shell=False,
    check=False,
)
```

The model cannot submit raw shell strings.

Current policy is intentionally narrow.

Examples include:

```text
pwd
ls
git status --short --branch
mkdir <direct-child-name>
```

Unsupported executables, arguments, paths, and working directories are denied.

Never introduce:

```text
os.system(model_output)
shell=True with model output
eval(model_output)
exec(model_output)
```

or equivalent unrestricted execution paths.

---

# Durable approvals

Mutation proposals are persisted before execution.

Default database:

```text
.runtime/approvals.sqlite3
```

State machine:

```text
pending
   |
   v
executing
  /     \
 v       v
approved failed
```

Important properties:

- exact tool and arguments are persisted before execution
- `pending -> executing` is committed before side effects
- approved results are replayed
- failed approvals are terminal
- concurrent approval attempts are serialized
- pending approvals survive FastAPI restart
- ambiguous `executing` approvals are never automatically retried

This prevents blind duplicate side effects after crashes or transport failures.

---

# Phoenix completion handshake

Phoenix dispatches durable work to FastAPI:

```text
Phoenix
   |
   | POST /v1/jobs/execute
   v
FastAPI
   |
   | 202 Accepted
   v
background AI execution
```

When AI work completes:

```text
AI result
   |
   v
CompletionOutbox
   |
   v
authenticated Phoenix callback
   |
   v
Phoenix ACK
   |
   v
outbox deletion
```

Default completion outbox:

```text
.runtime/completion_outbox.sqlite3
```

The durable outbox protects completion delivery from temporary Phoenix callback failures.

---

# Configuration

Create the local environment file:

```bash
cp .env.example .env
```

Do not commit `.env`.

Important model configuration:

```env
HUB_MODEL_KEY=hub-main

MODEL_PROFILES='{
  "hub-main": {
    "backend": "ministral",
    "model_path": "/path/to/Ministral",
    "enabled": true
  },
  "qwen2.5-0.5b-funccall": {
    "backend": "qwen-funccall",
    "model_path": "/path/to/account-model",
    "enabled": true
  },
  "qwen3-0.6b": {
    "backend": "qwen3",
    "model_path": "/path/to/access-model",
    "enabled": true
  },
  "qwen2.5-coder-0.5b": {
    "backend": "qwen-coder",
    "model_path": "/path/to/developer-model",
    "enabled": true
  }
}'
```

The real `.env` may keep the JSON on one line.

Other relevant configuration includes:

```text
PHOENIX_BASE_URL
ITSM_INTERNAL_JOB_TOKEN

COMPLETION_OUTBOX_PATH
OUTBOX_RETRY_SECONDS
OUTBOX_BATCH_SIZE

APPROVAL_STORE_PATH

PROCESS_WORKSPACE_ROOT

AGENTS_DIR

DIRECTORY_BACKEND

AD_HOST
AD_PORT
AD_USE_SSL
AD_BASE_DN

AD_BIND_USER / AD_BIND_DN
AD_BIND_PASSWORD

AD_WRITE_BIND_USER / AD_WRITE_BIND_DN
AD_WRITE_BIND_PASSWORD

AD_ACCESS_GROUPS
```

Deployment-specific values belong in centralized validated Settings rather than being scattered throughout runtime code.

---

# Start the AI service

```bash
python -m uvicorn api.app:app \
  --host 127.0.0.1 \
  --port 8000
```

The host and port shown above are development launch values. Deployment tooling may use `AI_HOST` and `AI_PORT`.

Readiness:

```bash
curl -s \
  http://127.0.0.1:8000/ready \
  | python -m json.tool
```

Expected shape:

```json
{
  "status": "ready",
  "service": "itsm-ai"
}
```

---

# Tests

Current fast baseline:

```bash
python -m pytest -q \
  subagents/test \
  test/test_mcp.py
```

Current cleanup baseline:

```text
53 passed
```

Integration tests live under:

```text
subagents/integration/
```

They include real-model and MCP vertical slices and are intentionally more expensive than the fast unit baseline.

---

# Important files

```text
api/app.py
api/runtime.py

config/settings.py

agent/approval_store.py
agent/approvals.py
agent/completion_outbox.py
agent/mcp_client.py

subagents/hub.py

subagents/core/orchestration/router.py
subagents/core/orchestration/orchestrator.py
subagents/core/orchestration/primary_assistant.py
subagents/core/orchestration/runtime.py
subagents/core/tooling/gateway.py
subagents/core/tooling/parser.py
subagents/core/tooling/prompt.py

subagents/llm/runtime/model_manager.py
subagents/llm/runtime/registry.py
subagents/llm/runtime/inference.py
subagents/llm/runtime/scheduler.py
subagents/llm/runtime/factory.py

subagents/agents/
subagents/prompts/

tools/registry.py
tools/presentation.py
tools/workspace.py

services/process_runner.py
services/directory/

mcp_server.py
```

---

# Current deliberate limitations

These are feature work, not cleanup blockers:

- one tool proposal per specialist task
- simple multi-specialist result composition
- no direct worker-to-worker communication
- real AD password reset disabled
- direct LDAP group membership checks
- process-local GPU scheduling
- no measured HOT / WARM / COLD residency policy yet
- no production continual-learning promotion pipeline yet
- no machine-wide GPU authority across multiple Python worker processes
- polling may still be used on the browser-facing Phoenix side

The architecture now exposes stable seams for those capabilities without requiring another foundational runtime rewrite.