# Agentic AI

Local-first AI runtime for the Agentic Developer Hub.

This repository contains the FastAPI execution service, local language models, primary conversational assistant, specialist routing, ToolGateway policy, MCP integration, LDAP/Samba AD integration, approvals, and a durable completion outbox.

> Development prototype. Model output is never authorization.

## Current capabilities

- FastAPI `/health` and `/ready`
- Phoenix durable-job execution
- SQLite completion outbox
- local Ministral Hub / primary assistant
- specialist routing
- Qwen2.5 function-calling Account worker
- ToolGateway allowlists
- identifier validation
- MCP execution boundary
- Samba AD development environment
- LDAP/LDAPS account lookup
- human approval flow for mutations
- user-facing natural tool answers

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
                |
          +-----+-----+
          |           |
          v           v
       Approval      MCP
                      |
                      v
                DirectoryService
                      |
                      v
                  LDAP/Samba
```

The Hub model currently handles both routing and normal conversation.

## Setup

Recommended:

- Linux/WSL2
- Python 3.12
- Conda
- Docker
- PyTorch/Transformers
- sufficient RAM/VRAM

```bash
conda activate qwen-infra
python -m pip install -r requirements.txt
cp .env.example .env
```

Configure model paths in `.env`:

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
```

## Samba AD

```bash
docker compose up -d samba-ad
docker compose ps
python scripts/ldap_preflight.py
```

Ports:

```text
LDAP  1389
LDAPS 1636
```

## Start AI

```bash
python -m uvicorn api.app:app   --host 127.0.0.1   --port 8000
```

Healthy startup:

```text
[OUTBOX] Delivery worker started pending=0
[MCP] Persistent server started.
[API] AI runtime ready.
```

## Completion outbox

Default:

```text
.runtime/completion_outbox.sqlite3
```

Flow:

```text
AI result
  -> persist completion
  -> callback Phoenix
  -> ACK
  -> remove outbox entry
```

## Tool safety

```text
Models propose.
Python validates.
MCP or trusted runner executes.
```

Current directory tools:

| Tool | Type | Approval |
| --- | --- | --- |
| `account_status` | read | no |
| `check_access` | read | no |
| `unlock_user` | mutation | yes |
| `reset_password` | mutation | yes |

Structured tool results stay internal.

Example internal data:

```python
{"ok": True, "user_id": "jdoe", "enabled": True, "locked": False}
```

User-facing answer:

```text
jdoe is enabled and is not locked.
```

## Next capability: local process execution

Add a governed process tool, not arbitrary shell text.

Suggested request:

```json
{
  "executable": "pwd",
  "args": [],
  "cwd": "/approved/workspace",
  "timeout_seconds": 10
}
```

Suggested execution:

```python
subprocess.run(
    [executable, *args],
    cwd=cwd,
    timeout=timeout_seconds,
    shell=False,
    capture_output=True,
    text=True,
)
```

First allow `pwd`; then add `mkdir` as approval-required.

## Important files

```text
api/app.py
subagents/hub.py
subagents/core/orchestrator.py
subagents/core/primary_assistant.py
subagents/core/llm_router.py
subagents/core/runtime.py
subagents/core/tool_gateway.py
subagents/llm/
subagents/agents/
subagents/prompts/
agent/completion_outbox.py
agent/mcp_client.py
mcp_server.py
services/directory/
tools/
```

## Tests

```bash
python -m compileall api agent subagents services tools
python -m pytest subagents/test -v
```

Then test through the full UI:

```text
hello
Is jdoe locked?
```

## Known issues

- General chat may use two sequential Hub generations.
- CPU/disk offload can make Ministral slow.
- `active_jobs` uses `job_id` as its current key; retry-attempt handling needs improvement.
- Fault tests must be isolated from the normal outbox and live service ports.
- Old callback entries can retry indefinitely when Phoenix permanently returns errors such as an unknown job.

See the cross-repository `PROJECT_HANDOFF.md`.
