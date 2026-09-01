# ITSM Agent

A local-first, multi-agent IT service management prototype designed to route natural-language requests through a small local hub model and specialized worker models before executing directory-service operations through MCP.

The project currently focuses on safe, auditable Active Directory-style workflows using local language models, Python-enforced security boundaries, and a Samba AD development environment.

> This repository is a development prototype and is not production-ready.

## Current Development State

The project is transitioning from a single-agent architecture into a hub/worker multi-agent architecture.

### Working infrastructure

* Local Qwen3-0.6B inference
* Persistent MCP subprocess
* `DirectoryService` abstraction
* Mock directory backend
* LDAP/Samba AD backend
* Local Samba Active Directory domain in Docker
* LDAPS connectivity
* LDAP read-only preflight
* Account status lookup
* Group-based access checks
* Human approval flow for mutation tools
* Identifier validation before tool execution
* Experimental approved account unlock

### Hub/worker system

Implemented and tested:

* Markdown-defined specialist agents
* `AgentDefinition`
* `AgentTask`
* `AgentResult`
* `HubResult`
* `AgentRegistry`
* Agent-definition loader
* `ModelRegistry`
* Deterministic specialist router
* Qwen3-0.6B hub backend
* LLM-based hub router
* Orchestrator
* Worker runtime
* Tool-call JSON parser
* Worker-specific tool allowlists
* Central `ToolGateway`
* Schema-driven worker tool prompts
* Qwen2.5-0.5B function-calling Account worker
* Read-only Account worker end-to-end test through real MCP and LDAP
* Deterministic Hub → Account Worker → MCP → LDAP end-to-end test

### Current model roles

#### Hub

Current target:

```text
Qwen3-0.6B
```

The hub is responsible for deciding **which specialist worker should handle a request**.

It is not intended to directly authorize or execute privileged directory operations.

The Qwen3 hub backend and LLM router are implemented and unit-tested. Real-model routing validation and replacement of the temporary deterministic router are the next integration step.

#### Account specialist

Current model:

```text
Qwen2.5-0.5B FuncCall
```

The Account worker specializes in converting account-management requests into structured function-call proposals such as:

```json
[
  {
    "name": "account_status",
    "arguments": {
      "user_id": "jdoe"
    }
  }
]
```

The Account worker has successfully completed a real end-to-end read request:

```text
User request
    ↓
Account worker
    ↓
Tool-call parser
    ↓
ToolGateway
    ↓
MCP
    ↓
LDAP
    ↓
Samba AD
```

Tested request:

```text
Is jdoe locked?
```

Result:

```text
account_status(user_id="jdoe")
```

with the real LDAP result returned successfully.

#### Access specialist

The Access specialist definition exists, but its final worker model and end-to-end implementation are still under development.

Additional specialists may later cover networking, endpoint management, security, incidents, software, and other ITSM domains.

## Architecture

```text
                         User
                          |
                          v
                  +---------------+
                  |   Hub Agent   |
                  | Qwen3-0.6B    |
                  +-------+-------+
                          |
                    delegation
                          |
             +------------+------------+
             |                         |
             v                         v
   Account Specialist          Access Specialist
   Qwen2.5-0.5B FuncCall       worker model TBD
             |                         |
             +------------+------------+
                          |
                          v
                    Tool Parser
                          |
                          v
                    ToolGateway
                          |
          +---------------+---------------+
          |                               |
          | read                          | mutation
          v                               v
         MCP                      Approval Manager
          |                               |
          |                         Human Approval
          |                               |
          +---------------+---------------+
                          |
                          v
                  Persistent MCP
                          |
                          v
                  DirectoryService
                     /        \
                    /          \
                 Mock          LDAP
                                |
                                v
                              LDAPS
                                |
                                v
                            Samba AD
```

The architecture follows four distinct responsibilities:

```text
Hub
= Who should handle the request?

Worker
= What operation should be proposed?

Python / ToolGateway
= Is that operation allowed?

MCP / DirectoryService
= Execute the permitted operation.
```

The language models are not security boundaries.

## Security Model

Worker output is treated only as a proposal.

Before execution, application code enforces:

1. The selected agent exists.
2. The worker model exists in the `ModelRegistry`.
3. Worker output is valid structured JSON.
4. The proposed tool exists.
5. The tool is explicitly allowed for that specialist.
6. Identity-sensitive arguments match identifiers in the original request.
7. Mutation tools require human approval.
8. Permitted operations cross the MCP boundary.
9. Directory permissions remain an additional enforcement layer.

For example:

```text
Worker proposes:
unlock_user(user_id="jdoe")
        |
        v
ToolGateway
        |
        +-- Is unlock_user allowed for this worker?
        +-- Does jdoe appear in the user's request?
        +-- Does unlock_user require approval?
        |
        v
approval_required
```

The mutation is not sent directly to MCP.

## Repository Layout

The current development structure includes:

```text
itsm-agent/
├── agent/
│   ├── agent.py
│   ├── approvals.py
│   ├── llm.py
│   └── mcp_client.py
│
├── subagents/
│   ├── agents/
│   │   ├── account-specialist.md
│   │   └── access-specialist.md
│   │
│   ├── core/
│   │   ├── types.py
│   │   ├── loader.py
│   │   ├── registry.py
│   │   ├── router.py
│   │   ├── llm_router.py
│   │   ├── orchestrator.py
│   │   ├── runtime.py
│   │   ├── tool_parser.py
│   │   ├── tool_prompt.py
│   │   └── tool_gateway.py
│   │
│   ├── llm/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── qwen_funcall.py
│   │   └── qwen_hub.py
│   │
│   ├── test/
│   ├── integration/
│   └── hub.py
│
├── services/
│   └── directory/
│       ├── base.py
│       ├── factory.py
│       ├── ldap.py
│       └── mock.py
│
├── tools/
│   ├── account.py
│   ├── access.py
│   └── registry.py
│
├── scripts/
│   └── ldap_preflight.py
│
├── docker/
│   └── samba-ad/
│
├── main.py
├── mcp_server.py
├── config.py
└── docker-compose.yaml
```

The existing single-agent CLI remains available while the new hub/worker path is developed independently.

The main CLI should not be switched to the hub runtime until the real Qwen3 routing path is validated end-to-end.

## Models

Models are loaded locally.

Current development models:

```text
Models/
├── Qwen3-0.6B
└── qwen2.5-0.5b-funccall
```

Current intended responsibilities:

| Model                 | Role                                       |
| --------------------- | ------------------------------------------ |
| Qwen3-0.6B            | Hub / specialist routing                   |
| Qwen2.5-0.5B FuncCall | Account worker / structured tool selection |
| TBD                   | Access worker                              |
| TBD                   | Future specialist workers                  |

Specialists may use different base models or separately fine-tuned copies of the same small model family.

## Requirements

Recommended development environment:

* Linux or WSL2
* Python 3.12
* Conda or another Python virtual environment
* Docker Engine
* Docker Compose
* PyTorch
* Transformers
* enough RAM/VRAM to load the local models

Activate the current environment:

```bash
conda activate qwen-infra
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Directory Backends

### Mock

```env
DIRECTORY_BACKEND=mock
```

Useful for testing orchestration and approval behavior without LDAP.

### LDAP

```env
DIRECTORY_BACKEND=ldap
```

The LDAP backend supports separate read and write credentials.

Example:

```env
AD_HOST=127.0.0.1
AD_PORT=1636
AD_USE_SSL=true

AD_BIND_DN=CN=svc_itsm_reader,CN=Users,DC=itsm,DC=local
AD_BIND_PASSWORD=change-me

AD_WRITE_BIND_DN=CN=svc_itsm_writer,CN=Users,DC=itsm,DC=local
AD_WRITE_BIND_PASSWORD=change-me

AD_BASE_DN=DC=itsm,DC=local
AD_TEST_USER=jdoe
```

Use narrowly delegated service accounts in real environments.

Do not use Domain Administrator credentials in production.

## Samba AD Development Environment

Start Samba AD:

```bash
docker compose up -d samba-ad
```

Check status:

```bash
docker compose ps
```

Current development ports:

```text
LDAP   localhost:1389 -> container:389
LDAPS  localhost:1636 -> container:636
```

The Python LDAP development configuration currently uses LDAPS.

Run the LDAP preflight:

```bash
python scripts/ldap_preflight.py
```

Healthy output should include:

```text
[OK] Configuration present
[OK] TCP connection succeeded
[OK] LDAP adapter initialized
[OK] LDAP bind succeeded
[OK] Test account lookup

LDAP read-only preflight passed.
```

## Tool Policy

Current tools:

| Tool             | Operation | Approval |
| ---------------- | --------- | -------- |
| `account_status` | Read      | No       |
| `check_access`   | Read      | No       |
| `unlock_user`    | Mutation  | Yes      |
| `reset_password` | Mutation  | Yes      |

`reset_password` remains disabled for real LDAP execution until password policy, delegated permissions, secret handling, and verification are implemented safely.

## Testing

Run the sub-agent unit suite:

```bash
python -m pytest subagents/test -v
```

Run the Account worker model integration tests:

```bash
python -m pytest \
    subagents/integration/test_account_worker_model.py \
    -v -s
```

Run the real Account worker E2E test:

```bash
python -m pytest \
    subagents/integration/test_account_worker_e2e.py \
    -v -s
```

Run the current Hub E2E test:

```bash
python -m pytest \
    subagents/integration/test_hub_e2e.py \
    -v -s
```

The real Account worker and deterministic Hub path have both successfully reached the LDAP-backed `account_status` operation through MCP.

## Current Limitations

* Qwen3 real-model hub routing is not yet connected to the main hub runtime.
* The deterministic router is still used for the proven Hub E2E path.
* Access worker implementation is incomplete.
* Only one worker tool call is currently permitted per `AgentTask`.
* Worker-to-worker communication is intentionally unsupported.
* Password reset remains disabled in the LDAP backend.
* Approvals are not durably persisted.
* No durable audit database exists.
* CLI remains the primary interface.
* No production authentication/authorization layer exists.
* Access checks currently use direct group membership.
* Local Samba AD uses development-only configuration.
* Production certificate validation and identity management are not complete.

## Near-Term Roadmap

1. Benchmark real Qwen3-0.6B hub routing.
2. Replace deterministic hub routing with validated `LLMRouter`.
3. Run full Qwen3 Hub → Account Worker → MCP → LDAP E2E.
4. Implement and benchmark the Access specialist worker.
5. Support multi-specialist delegation safely.
6. Integrate `run_hub()` into the main CLI.
7. Expand specialist domains.
8. Add durable audit and approval persistence.
9. Add production-grade authentication and authorization.
10. Later expose the Python agent runtime through a service API, potentially with a separate Elixir/Phoenix REST layer.

## Development Principle

The central rule of the system is:

```text
Models propose.
Python validates and authorizes.
MCP executes.
Directory permissions enforce.
```

No language-model output should ever be treated as authorization.

## License

Licensed under the MIT License.
