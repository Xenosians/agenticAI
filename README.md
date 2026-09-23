# Agentic AI

Local-first AI runtime for the **Agentic Developer Hub / ITSM Platform**.

This repository contains the Python/FastAPI AI service that sits behind the Phoenix application boundary. It provides conversational orchestration, specialist routing, governed tool execution, durable approval handling, local model lifecycle management, and trusted integrations for developer and ITSM workflows.

> **Core rule:** model output is a proposal, never authorization.

The runtime is designed around one principle:

```text
RUSH CAPABILITIES.
DO NOT RUSH AUTHORITY.
```

---

## System role

The browser does not call this service directly.

The intended application path is:

```text
Browser / Nim + Karax
        |
        v
Phoenix / Elixir backend
        |
        | durable job dispatch
        v
FastAPI AI service
        |
        v
Primary Assistant / Hub
        |
        v
Specialist
        |
        v
SemanticGuard
        |
        v
ToolGateway
        |
        v
MCP / trusted provider
        |
        v
Verified result
        |
        v
Phoenix durable result
        |
        v
Frontend
```

Repository boundaries:

```text
agenticFrontend
    Browser UI only.
    Calls Phoenix, never FastAPI directly.

agenticBackend
    Public API, durable jobs, leases, approval binding,
    callback validation, and browser-facing lifecycle.

agenticAI
    Model runtime, orchestration, semantic validation,
    ToolGateway, approval execution safety, providers,
    completion outbox, and runtime evidence.
```

---

## Security and authority model

The AI runtime deliberately separates **semantic intent** from **execution authority**.

```text
User request
    |
    v
Hub / Primary Assistant
    |
    | SemanticIntent
    v
Specialist
    |
    | one typed tool proposal
    v
SemanticGuard
    |
    | semantic consistency only
    v
ToolGateway
    |
    | deterministic authorization boundary
    v
Trusted provider / MCP
```

`SemanticIntent` may describe:

```text
effect
allowed_tools
allowed_arguments
forbidden_tools
forbidden_arguments
max_tool_calls
clarification_required
summary
```

It is not an authorization token.

`ToolGateway` and trusted application code own:

- tool existence and specialist allowlists;
- typed argument validation;
- exact user-grounding and provenance;
- trusted enum/resource resolution;
- risk classification;
- approval requirements;
- trusted preconditions;
- hidden execution snapshots;
- provider dispatch;
- final execution policy.

Unknown, malformed, unsupported, or unsafe requests fail closed.

---

## Exact approval execution

Governed mutations are persisted before execution.

A successful approval path is:

```text
model proposes exact capability + exact user arguments
        |
        v
trusted policy validates preconditions
        |
        v
trusted policy appends declared execution snapshot
        |
        v
durable approval
        |
        | user approval
        v
execute exact persisted action
        |
        v
provider read-back / reconciliation
```

The model is **not rerun** after approval to regenerate mutation arguments.

Trusted policy arguments may append data such as immutable provider IDs, current resource names, repository HEADs, remote refs, or account state. They may not silently rewrite model/user-controlled arguments.

For remote mutations, transport success alone is not treated as sufficient proof. Provider state is reconciled after execution.

Ambiguous side effects are never blindly retried.

---

## Runtime architecture

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
            +-- LLMRouter
            +-- PrimaryAssistant
            +-- AgentRuntime
            +-- Orchestrator
```

Runtime resources are owned by the FastAPI lifespan.

`build_hub(...)` assembles the reasoning graph; it does not own process-wide runtime resources.

---

## Conversational and specialist flow

For normal non-tool conversation:

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
   +-- no specialist required
           |
           v
     PrimaryAssistant
```

For specialist work:

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
Specialist
   |
   v
Structured tool proposal
   |
   v
SemanticGuard
   |
   v
ToolGateway
   |
   +--------------------------+
   |                          |
   v                          v
read / allowed          approval required
   |                          |
   v                          v
MCP / provider          ApprovalManager
   |                          |
   +------------+-------------+
                |
                v
        trusted execution
```

Specialists are domain roles and policy/tool definitions. They are not required to have separate model weights.

---

## Current specialists

Current repository definitions include:

```text
account-specialist
access-specialist
developer-specialist
ticket-specialist
knowledge-specialist
atlassian-specialist
jira-specialist
```

### Account

Current governed account operations include:

```text
account_status
unlock_user
reset_password
enable_user
disable_user
```

### Access

Current access operations include:

```text
check_access
grant_access
revoke_access
```

Logical resources are resolved by trusted configuration rather than model-generated directory identifiers.

### Developer workspace

The developer specialist exposes governed workspace discovery, execution, service inspection, and Git operations.

Examples include:

```text
process_exec
workspace_mkdir
workspace_read_text
workspace_list
workspace_search
workspace_file_info
workspace_project_info
workspace_run_tests
workspace_run_build
workspace_process_snapshot
workspace_service_status
workspace_service_logs
```

### Ticketing

Ticket-oriented capabilities include:

```text
ticket_get
ticket_search
ticket_history
ticket_comments
ticket_add_comment
ticket_create
ticket_assign
ticket_transition
```

### Knowledge

Knowledge and runbook capabilities are read-only:

```text
knowledge_search
knowledge_get
runbook_get
```

Retrieving a runbook does not authorize execution of the actions described by that runbook.

---

## Governed Git

Model-facing Git tools operate on logical repository identifiers such as:

```text
ai
backend
frontend
```

Physical repository paths are resolved by trusted application configuration.

The model never receives authority to choose arbitrary host repository roots.

### Read capabilities

```text
workspace_git_status
workspace_git_branches
workspace_git_log
workspace_git_diff
workspace_git_staged_diff
workspace_git_changed_files
```

### Mutation capabilities

```text
workspace_git_stage_files
workspace_git_unstage_files
workspace_git_stage_all
workspace_git_create_branch
workspace_git_switch_branch
workspace_git_commit
workspace_git_push
```

Git mutations use trusted fixed commands rather than generic model-provided shell execution.

Important boundaries include:

- exact repository selection;
- exact path grounding where paths are model-visible;
- no unrestricted shell;
- no force push;
- no model-selected arbitrary remote or refspec;
- post-approval state revalidation;
- post-execution verification;
- fail-closed handling when remote side effects are uncertain.

`workspace_git_push` is a high-risk operation and binds trusted branch/HEAD/upstream/remote state before approval.

---

## Atlassian foundation

The Atlassian administration layer is separate from Jira project operations.

Current read-only capabilities include:

```text
atlassian_credential_status
atlassian_org_list
atlassian_org_get
atlassian_workspace_list
atlassian_api_token_metadata
```

The provider owns trusted Atlassian URLs and credentials.

Secret token values are never returned to the model.

Workspace discovery means discovering existing Atlassian sites/product workspaces. It is not workspace creation.

---

## Jira project administration

The Jira integration uses typed semantic capabilities rather than arbitrary Jira HTTP access.

The model does **not** receive:

```text
arbitrary HTTP method
arbitrary Atlassian URL
arbitrary REST path
raw Authorization header
provider credentials
unrestricted JQL
```

Current project capabilities are:

```text
jira_project_list
jira_project_get
jira_project_create
jira_project_update
jira_project_archive
```

### Project reads

`jira_project_list` performs bounded provider-side project discovery.

`jira_project_get` retrieves exactly one project using the exact user-supplied project ID or key.

### Project creation

Model-facing input:

```text
jira_project_create(
    project_key,
    project_name,
    template
)
```

Trusted code resolves:

```text
project type
provider-native template key
authenticated Jira lead identity
REST path
HTTP method
credentials
```

Creation is approval-gated and verified by provider read-back.

### Project rename

Model-facing input:

```text
jira_project_update(
    project_id_or_key,
    new_name
)
```

The current capability changes **only the project name**.

It does not mutate:

```text
project key
lead
category
workflow scheme
permission scheme
project type
template
```

Trusted policy binds the exact pre-approval project identity and current name, revalidates them after approval, and verifies the renamed state after the provider mutation.

### Project archive

Model-facing input:

```text
jira_project_archive(
    project_id_or_key
)
```

Archive is a **high-risk** administrative mutation.

Trusted policy binds:

```text
expected_project_id
expected_project_key
expected_project_name
```

Execution proves the project is still the approved live project, archives through the trusted provider, and then verifies archived lifecycle state.

Project deletion is intentionally not part of the current project tool surface.

---

## Model architecture

Models are selected through logical profiles rather than hardcoded public contracts.

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

A specialist definition references a logical model key:

```yaml
---
name: account-specialist
model: qwen2.5-0.5b-funccall
tools:
  - account_status
  - unlock_user
  - reset_password
  - enable_user
  - disable_user
---
```

Model profile configuration owns deployment-specific details such as:

```text
backend
model_path
quantization
compute_dtype
model_dtype
device_map
offload_folder
```

Current supported quantization profile values include:

```text
auto
none
bnb4
fp8
```

The public Phoenix/frontend contract does not depend on model brand or checkpoint path.

---

## GPU scheduling

All model inference is mediated by:

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

The current scheduler provides:

- serialized admission;
- priority ordering;
- FIFO within equal priority;
- non-preemptive execution;
- blocking inference outside the asyncio event loop;
- cancellation safety;
- GPU/runtime metrics.

Observed metrics include queue wait, execution time, total latency, loaded state, CUDA memory, generated-token count, and tokens/sec where supported.

Residency/concurrency policy should be changed from evidence rather than GPU-name-specific hardcoding.

---

## MCP and trusted providers

FastAPI owns one persistent `MCPRuntime`.

The MCP server registers trusted implementations for:

```text
directory / LDAP
access mutations
account lifecycle
developer workspace
Git
ticketing
assets
knowledge
Atlassian administration
Jira project administration
```

`tools/registry.py` and domain catalogs contain trusted capability metadata. They do not grant the model arbitrary provider access.

---

## Active Directory / LDAP

Directory provider selection is configuration driven:

```env
DIRECTORY_BACKEND=mock
```

or:

```env
DIRECTORY_BACKEND=ldap
```

Current LDAP-oriented support includes account status, lock state, account lifecycle operations, and group-backed logical access operations.

Separate read and mutation credentials are supported.

Local Samba AD may be used for integration testing.

Preflight:

```bash
python scripts/ldap_preflight.py
```

---

## Durable approvals

Default approval database:

```text
.runtime/approvals.sqlite3
```

Execution-safety state:

```text
pending
   |
   v
executing
  /     \
 v       v
approved failed
```

Important behavior:

- exact tool and execution arguments are persisted before side effects;
- `pending -> executing` is committed before provider execution;
- approved results can be replayed;
- failed approvals are terminal;
- concurrent approval attempts are serialized;
- pending approvals survive process restart;
- ambiguous executing mutations are not automatically replayed.

Phoenix remains responsible for the user-facing durable approval/job lifecycle. AI SQLite owns execution-safety state.

---

## Phoenix completion handshake

Phoenix dispatches a durable job to the AI service:

```text
Phoenix
   |
   | POST /v1/jobs/execute
   v
FastAPI
   |
   | 202 Accepted
   v
background execution
```

When execution completes:

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

Default outbox:

```text
.runtime/completion_outbox.sqlite3
```

The outbox prevents temporary Phoenix/network failure from silently losing completion delivery.

---

## Learning and runtime evidence

Runtime trajectories are evidence, not direct training data.

The controlled adaptation path is:

```text
runtime trajectory
    |
    v
provenance validation
    |
    v
filter / deduplicate
    |
    v
human corrections / preferences
    |
    v
curated batch
    |
    v
offline training
    |
    v
evaluation / regression
    |
    v
gated promotion
```

The live runtime does not silently retrain model weights from raw model output.

Current evidence capture includes approval execution lineage and bounded runtime context suitable for later curation.

---

## Configuration

Create a local environment file:

```bash
cp env.example .env
```

Do not commit `.env`.

Important configuration groups include:

```text
AI_HOST
AI_PORT

PHOENIX_BASE_URL
ITSM_INTERNAL_JOB_TOKEN
PHOENIX_HTTP_TIMEOUT_SECONDS
JOB_HEARTBEAT_INTERVAL_SECONDS

COMPLETION_OUTBOX_PATH
OUTBOX_RETRY_SECONDS
OUTBOX_BATCH_SIZE

APPROVAL_STORE_PATH

PROCESS_WORKSPACE_ROOT
GIT_DEFAULT_REPOSITORY
GIT_REPOSITORIES

AGENTS_DIR

HUB_MODEL_KEY
MODEL_PROFILES

DIRECTORY_BACKEND
TICKETING_BACKEND
ASSET_BACKEND
KNOWLEDGE_BACKEND

JIRA_BASE_URL
JIRA_EMAIL
JIRA_API_TOKEN
JIRA_HTTP_TIMEOUT_SECONDS

ATLASSIAN_ADMIN_API_KEY
ATLASSIAN_ORG_ID
ATLASSIAN_ADMIN_HTTP_TIMEOUT_SECONDS

AD_HOST
AD_PORT
AD_USE_SSL
AD_BASE_DN
AD_BIND_USER / AD_BIND_DN
AD_BIND_PASSWORD
AD_WRITE_BIND_USER / AD_WRITE_BIND_DN
AD_WRITE_BIND_PASSWORD
AD_ACCESS_GROUPS

LEARNING_CAPTURE_ENABLED
LEARNING_TRAJECTORY_PATH
LEARNING_CORRECTION_PATH
```

Deployment-specific values belong in `Settings` / model profiles rather than scattered environment reads.

---

## Installation

Create or activate the Python environment used for the project, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

For the current local setup:

```bash
conda activate qwen-infra
```

Some model profiles may require additional backend-specific dependencies such as bitsandbytes depending on the selected quantization/runtime.

---

## Start the AI service

Development launch:

```bash
python -m uvicorn api.app:app \
  --host 127.0.0.1 \
  --port 8000
```

Readiness:

```bash
curl -s http://127.0.0.1:8000/ready \
  | python -m json.tool
```

Phoenix should remain the public application boundary. Do not expose FastAPI directly as the browser-facing API.

---

## Provider preflights

Useful provider checks include:

```bash
python scripts/ldap_preflight.py
python scripts/atlassian_preflight.py
python scripts/jira_projects_preflight.py
python scripts/jira_project_create_preflight.py
python scripts/jira_project_update_preflight.py
python scripts/jira_project_archive_preflight.py
```

Preflight scripts should remain read-only unless their filename and purpose explicitly describe an approval/execution proof.

---

## Tests

Run the full Python suite:

```bash
python -m pytest -q
```

Run focused Jira tests:

```bash
python -m pytest -q \
  tests/unit/services/jira \
  tests/unit/tools/jira \
  tests/unit/subagents/core/definitions/test_jira_specialist.py
```

Run Git-focused unit tests:

```bash
python -m pytest -q tests/unit/tools/git
```

Check whitespace before committing:

```bash
git diff --check
```

Real-model and live-provider proofs are intentionally more expensive and should be run as controlled checkpoints rather than casually mixed into every fast unit-test cycle.

---

## Important source layout

```text
api/
    app.py
    runtime.py

agent/
    approval_store.py
    approvals.py
    completion_outbox.py
    mcp_client.py

config/
    settings.py

learning/
    evidence/
    integrations/

services/
    atlassian/
    directory/
    jira/
    knowledge/
    ticketing/
    git_repositories.py
    process_runner.py

subagents/
    hub.py
    agents/
    prompts/
    core/
        orchestration/
        tooling/
    llm/
        runtime/

tools/
    account/
    access/
    assets/
    atlassian/
    developer/
    git/
    jira/
    knowledge/
    ticketing/
    workspace/
    registry.py

scripts/
tests/

mcp_server.py
requirements.txt
env.example
```

---

## Cross-repository development

The three repositories are intentionally separate:

```text
Xenosians/agenticFrontend
Xenosians/agenticBackend
Xenosians/agenticAI
```

When changing a cross-service contract, verify all three boundaries:

```text
AI result shape
    ↓
Phoenix persistence / public contract
    ↓
Frontend decoding / rendering
```

A tool can be correct inside the AI repository and still be broken end-to-end if a structured field is dropped by MCP serialization, Phoenix persistence, JSON Schema, or frontend decoding.

Treat each serialization boundary as part of the contract.

---

## Current deliberate limitations

The current system is still an evolving MVP.

Known architectural/product gaps include:

- backend durable lifecycle does not yet expose first-class `denied` and `outcome_unknown` public states;
- Jira project deletion is not yet exposed;
- Jira/JSM provider coverage is still being expanded;
- broad enterprise RBAC/ABAC is not yet implemented;
- frontend capability discovery is not yet fully dynamic;
- polling remains in the browser-facing flow;
- some ITSM domains still use mock providers;
- GPU scheduling is process-local;
- no automatic live self-training;
- deployment/bootstrap automation remains separate from runtime authorization.

These are capability gaps, not reasons to weaken the execution boundary.

---

## Development rules

When extending the platform:

1. Prefer direct conversational response when no tool is required.
2. Prefer a typed read capability over a mutation.
3. Prefer a domain-specific trusted mutation over generic process execution.
4. Keep model-visible arguments minimal and user-grounded.
5. Keep provider URLs, methods, credentials, IDs, and translation logic trusted-side.
6. Bind mutable provider state before approval when stale execution would be dangerous.
7. Revalidate trusted state immediately before the side effect.
8. Verify provider state afterward.
9. Never blindly retry an ambiguous remote mutation.
10. Keep specialist free-form prompts capability-name-agnostic; exact capability IDs belong in the runtime/YAML tool list.
11. Add tests at every serialization and authority boundary.
12. Do not turn a successful HTTP response into `completed` unless the requested operation actually succeeded.

---

## Design summary

```text
Models understand intent.
Trusted code owns authority.

Hub intent is not authorization.
Specialist output is not authorization.
SemanticGuard is not authorization.
Approval is meaningful only when bound to exact trusted execution state.
Remote success is not final until provider truth is verified.
```

That boundary is the foundation of this repository.
