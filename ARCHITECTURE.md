# Architecture

## Overview

ITSM Agent is evolving from a single local-agent prototype into a hub/worker multi-agent system.

The architecture separates:

* natural-language routing
* specialist reasoning
* structured tool selection
* authorization
* privileged execution
* directory-service implementation

The primary design principle is:

```text
Hub decides who should handle the request.

Worker proposes what operation should be performed.

Application code decides whether the operation is allowed.

MCP and DirectoryService perform permitted execution.
```

Language-model output is never treated as authorization.

## Target Runtime Flow

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
      Account Specialist        Access Specialist
      Qwen2.5-0.5B              worker model TBD
      FuncCall
               |                         |
               +------------+------------+
                            |
                            v
                     Tool-call Parser
                            |
                            v
                       ToolGateway
                            |
             +--------------+--------------+
             |                             |
           read                         mutation
             |                             |
             v                             v
            MCP                     Approval Manager
             |                             |
             |                       Human Approval
             |                             |
             +--------------+--------------+
                            |
                            v
                    Persistent MCP Server
                            |
                            v
                    DirectoryService
                       /          \
                      /            \
                 Mock backend    LDAP backend
                                   |
                                   v
                                 LDAPS
                                   |
                                   v
                               Samba AD
```

## Hub Layer

### Responsibility

The hub determines which registered specialist or specialists should receive a user request.

The hub does not directly authorize privileged execution.

Current target hub model:

```text
Qwen3-0.6B
```

Current development components:

* `QwenHubBackend`
* `LLMRouter`
* deterministic `Router`
* `Orchestrator`

The deterministic router remains useful as:

* a development baseline
* a predictable unit-test implementation
* a potential fallback for known request classes

The Qwen3-based router is implemented and unit-tested but still requires real-model routing validation before becoming the default production path.

## Specialist Layer

Specialists are defined through Markdown files containing metadata and system prompts.

Example concept:

```yaml
---
name: account-specialist
description: Handles account-management operations.
tools:
  - account_status
  - unlock_user
  - reset_password
model: qwen2.5-0.5b-funccall
max_steps: 3
---
```

This allows specialist behavior to be defined without hardcoding every agent as a dedicated Python class.

### Current Account worker

Model:

```text
Qwen2.5-0.5B FuncCall
```

Responsibilities:

* account-status lookup
* account lock-state lookup
* account-unlock proposal
* password-reset proposal

The model produces structured JSON function calls rather than directly executing tools.

Example:

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

A real Account-worker read path has been validated end-to-end against MCP and LDAP.

### Access worker

The Access specialist definition exists.

Its final model and real execution path remain under development.

## Agent Registry

`AgentRegistry` stores available specialist definitions.

The hub and runtime resolve agents by registered names rather than constructing hardcoded specialist classes.

This allows additional workers to be added without changing orchestration logic.

## Agent Loader

Specialist definitions are loaded from Markdown files with YAML front matter.

The loader converts these definitions into `AgentDefinition` instances.

This keeps specialist configuration data-driven.

## Model Registry

`ModelRegistry` maps logical model identifiers to actual loaded inference backends.

Example:

```text
qwen2.5-0.5b-funccall
        |
        v
QwenFuncCallBackend
        |
        v
local model checkpoint
```

The runtime therefore does not assume every worker uses the same model.

Future specialists may use:

* different base models
* different fine-tunes of the same base model
* different inference backends

without changing the worker runtime contract.

## Agent Runtime

`AgentRuntime` executes an `AgentTask`.

Current flow:

```text
AgentTask
    |
    v
AgentRegistry
    |
    v
AgentDefinition
    |
    v
ModelRegistry
    |
    v
Worker model
    |
    v
raw structured output
    |
    v
Tool Parser
    |
    v
ToolGateway
    |
    v
MCP / Approval
    |
    v
AgentResult
```

The runtime currently permits exactly one tool call per worker task.

This is an intentional V1 restriction that simplifies validation and prevents small worker models from creating uncontrolled multi-step execution chains.

## Tool Prompt Construction

Worker prompts are generated from the worker's explicit tool allowlist.

An Account worker therefore receives only Account tools.

For example:

```text
account_status
unlock_user
reset_password
```

It does not automatically receive unrelated tools such as `check_access`.

Tool metadata includes structured parameter descriptions.

This creates two independent boundaries:

```text
Prompt:
"Only use these tools."
        |
        v
soft behavioral constraint

ToolGateway:
tool_name must exist in agent.tools
        |
        v
hard application constraint
```

## Tool-call Parser

Worker output is expected to be structured JSON.

The parser validates:

* valid JSON
* array format
* object format
* tool name
* argument object
* non-empty call list

Malformed output does not reach the execution gateway.

## ToolGateway

`ToolGateway` is the primary application-side execution boundary for sub-agents.

It performs:

1. worker tool-allowlist validation
2. tool-registry validation
3. identifier validation
4. approval checks
5. MCP dispatch for permitted reads

Example:

```text
Worker:
unlock_user(user_id="jdoe")
        |
        v
ToolGateway
        |
        +-- account worker may use unlock_user?
        |
        +-- unlock_user exists?
        |
        +-- jdoe exists in original user request?
        |
        +-- mutation requires approval?
        |
        v
approval_required
```

The mutation is not executed immediately.

## Identifier Boundary

Identity-sensitive arguments proposed by a model are validated against the original user request.

Example:

```text
User:
"Is jdoe locked?"

Worker:
account_status(user_id="jsmith")
```

The gateway rejects the operation because `jsmith` was not present in the original request.

This prevents a model from silently substituting a different account identifier.

## Approval Boundary

Mutation operations remain protected by human approval.

Current mutation tools include:

```text
unlock_user
reset_password
```

The execution sequence is:

```text
Worker proposes mutation
        |
        v
ToolGateway
        |
        v
Approval record
        |
        v
Human approval
        |
        v
MCP mutation
        |
        v
Directory permissions
        |
        v
Result verification
```

No worker or hub model may bypass this process.

## MCP Boundary

MCP remains the separation between agent reasoning and directory execution.

Both read operations and approved mutation operations ultimately cross the MCP boundary.

This keeps:

```text
agent orchestration
```

separate from:

```text
directory implementation
```

and makes it possible to change backend infrastructure without rewriting specialist-agent reasoning.

## DirectoryService

`DirectoryService` is the backend abstraction consumed by MCP tools.

Current implementations:

* `MockDirectoryService`
* `LdapDirectoryService`

Backend selection uses:

```env
DIRECTORY_BACKEND=mock
```

or:

```env
DIRECTORY_BACKEND=ldap
```

## LDAP Backend

Current LDAP functionality includes:

* account status
* enabled-state lookup
* lock-state lookup
* direct group-based access checks
* experimental approved account unlock

Password reset remains disabled.

Separate read and mutation credentials are supported.

## Tested Integration Paths

### Account worker E2E

Validated:

```text
"Is jdoe locked?"
        |
        v
Qwen2.5-0.5B Account worker
        |
        v
account_status(user_id="jdoe")
        |
        v
ToolParser
        |
        v
ToolGateway
        |
        v
MCP
        |
        v
LDAP
        |
        v
Samba AD
```

The test successfully returned:

```text
enabled=True
locked=False
```

for the development `jdoe` account.

### Current Hub E2E

Validated with deterministic routing:

```text
User
    |
    v
Orchestrator
    |
    v
Deterministic Router
    |
    v
Account worker
    |
    v
ToolGateway
    |
    v
MCP
    |
    v
LDAP
```

The next integration milestone is replacing the deterministic routing decision with validated Qwen3-0.6B routing.

## Security Boundaries

The complete intended enforcement stack is:

```text
User input
    |
    v
Hub routing
    |
    v
Registered specialist validation
    |
    v
Worker inference
    |
    v
Structured-output validation
    |
    v
Worker tool allowlist
    |
    v
Tool registry
    |
    v
Identifier validation
    |
    v
Approval policy
    |
    v
MCP
    |
    v
Directory permissions
    |
    v
Execution/result verification
```

No individual layer should be considered sufficient on its own.

## Current Limitations

* Qwen3 real-model routing has not yet replaced deterministic Hub routing.
* Access specialist execution is incomplete.
* One tool call is allowed per `AgentTask`.
* Workers do not communicate directly with one another.
* Multi-worker result composition is currently basic.
* Main CLI still uses the older single-agent flow.
* Approval persistence is in-memory.
* No durable audit store exists.
* Password reset is disabled.
* Direct group membership is used for access checks.
* Samba AD configuration is development-only.
* Production authentication and authorization are not implemented.
* Production API/microservice boundaries are not implemented.

## Future Service Boundary

A later service architecture may separate the agent runtime from the public API.

Potential structure:

```text
Client
    |
    v
Elixir / Phoenix REST layer
    |
    v
Python Agent Service
    |
    v
Qwen Hub
    |
    v
Specialist workers
    |
    v
ToolGateway
    |
    v
MCP / Directory infrastructure
```

This service split is intentionally deferred until the core hub/worker runtime is stable.
