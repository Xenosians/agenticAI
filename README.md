# Agentic AI

Local-first AI runtime for the **Agentic Developer Hub / ITSM Platform**.

This repository contains the Python/FastAPI intelligence and governed-execution layer that sits behind the Phoenix backend. It provides conversational orchestration, specialist routing, local model lifecycle management, typed Jira/Git/ITSM capabilities, deterministic execution guardrails, durable approvals, runtime evidence capture, and a gated learning/evaluation stack.

> **Core rule:** model output is a proposal, never authorization.

The runtime follows one design principle:

```text
RUSH CAPABILITIES.
DO NOT RUSH AUTHORITY.
```

---

## Repository role

The browser does **not** call this service directly.

```text
Nim / Karax frontend
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
Trusted provider / MCP
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

Cross-repository ownership:

```text
agenticFrontend
    Browser UX, authentication UI, chats, job polling,
    approval controls, result rendering.

agenticBackend
    Public API, durable jobs, chat/session ownership,
    SurrealDB persistence, leases, callback validation,
    browser-facing approval lifecycle, outbound mail.

agenticAI
    Model runtime, orchestration, specialist semantics,
    SemanticGuard, ToolGateway, approval execution safety,
    providers, completion outbox, evidence and learning.
```

---

## Authority model

The AI layer deliberately separates **understanding** from **authority**.

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
    | typed capability proposal
    v
SemanticGuard
    |
    | semantic consistency
    v
ToolGateway
    |
    | deterministic authorization
    v
Trusted provider
```

`SemanticIntent` may define:

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

It is **not** an authorization token.

`ToolGateway` and trusted application code own:

- tool existence and specialist allowlists;
- typed argument validation;
- exact user grounding;
- risk classification;
- approval requirements;
- trusted preconditions;
- hidden execution snapshots;
- provider dispatch;
- post-execution verification.

Unknown, malformed, unsupported, or unsafe requests fail closed.

---

## Exact approval execution

Mutation approval persists the exact proposed action before side effects occur.

```text
model proposes exact capability + arguments
        |
        v
trusted policy validates current state
        |
        v
trusted policy binds immutable execution facts
        |
        v
durable approval
        |
        | explicit user approval
        v
execute exact persisted action
        |
        v
provider read-back / reconciliation
```

The model is not rerun after approval to regenerate mutation arguments.

Trusted policy may append facts such as:

```text
provider IDs
current project/resource identity
repository HEAD
current branch/upstream/remote
pre-mutation account state
```

Trusted policy may not silently rewrite model/user-controlled arguments.

Ambiguous remote mutations are never blindly retried.

---

## Runtime architecture

```text
FastAPI lifespan
    |
    +-- Settings
    +-- CompletionOutbox
    +-- Phoenix HTTP client
    +-- MCPRuntime
    +-- ApprovalStore / ApprovalManager
    +-- ModelManager
    +-- GpuScheduler
    +-- InferenceCoordinator
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

Model execution is serialized through the GPU scheduler. Logical specialists do not require dedicated physical model weights.

---

## Current specialists

Current agent definitions include:

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

```text
account_status
unlock_user
reset_password
enable_user
disable_user
```

### Access

```text
check_access
grant_access
revoke_access
```

### Ticketing

Read operations:

```text
ticket_get
ticket_search
ticket_history
ticket_comments
```

Governed mutations:

```text
ticket_add_comment
ticket_create
ticket_assign
ticket_transition
```

Ticket mutations are approval-gated and bind provider-side state before execution where required.

### Developer workspace

Representative capabilities include:

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

The generic process runner remains intentionally narrow. Domain-specific trusted capabilities are preferred over arbitrary shell access.

### Knowledge

```text
knowledge_search
knowledge_get
runbook_get
```

Knowledge retrieval never grants authority to execute actions described by a runbook.

---

## Governed Git

Git tools operate on trusted logical repository identifiers such as:

```text
ai
backend
frontend
```

Physical repository paths are resolved by application configuration.

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

Important boundaries include:

- no model-selected arbitrary repository root;
- no unrestricted shell Git mutation;
- no force push;
- no arbitrary remote/refspec selection;
- exact path/repository grounding;
- pre-approval repository-state binding;
- post-execution verification;
- fail-closed behavior when push outcome is ambiguous.

`workspace_git_push` is treated as a high-risk mutation.

---

## Jira and Atlassian

The Jira layer exposes typed semantic capabilities rather than arbitrary HTTP access.

The model does **not** receive:

```text
arbitrary HTTP method
arbitrary Atlassian URL
arbitrary REST path
raw Authorization header
provider credentials
unrestricted JQL
```

### Ticket operations

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

### Jira project operations

Current project capabilities are:

```text
jira_project_list
jira_project_get
jira_project_create
jira_project_update
jira_project_archive
jira_project_delete
```

`jira_project_create`, `jira_project_update`, `jira_project_archive`, and `jira_project_delete` are governed mutations.

Archive and delete are high-risk administrative operations.

`jira_project_delete`:

- requires explicit approval;
- binds immutable project identity before approval;
- rejects archived projects;
- enables Jira provider undo;
- does not permit model-requested permanent deletion;
- verifies deleted state after execution.

### Atlassian administration reads

```text
atlassian_credential_status
atlassian_org_list
atlassian_org_get
atlassian_workspace_list
atlassian_api_token_metadata
```

Secret token values remain trusted-side.

---

## Directory / LDAP

Directory provider selection is configuration driven:

```env
DIRECTORY_BACKEND=mock
```

or:

```env
DIRECTORY_BACKEND=ldap
```

Current support includes account status, account lifecycle, lock handling, and logical group-backed access operations.

Separate read/write bind identities are supported.

Local Samba AD is available for integration testing.

```bash
python scripts/ldap_preflight.py
```

---

## Durable approvals

Default execution-safety store:

```text
.runtime/approvals.sqlite3
```

Approval state protects execution identity independently of the browser-facing Phoenix job state.

Important properties:

- exact capability and arguments persist before mutation;
- execution state is persisted before provider side effects;
- approved results can be replayed;
- failed approvals are terminal;
- concurrent approval attempts are serialized;
- pending approvals survive restart;
- ambiguous in-progress mutations are not automatically replayed.

Phoenix remains the durable user-facing lifecycle owner.

---

## Phoenix completion handshake

Phoenix dispatch:

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

Completion:

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
Phoenix acknowledgement
   |
   v
outbox deletion
```

Default outbox:

```text
.runtime/completion_outbox.sqlite3
```

Temporary Phoenix/network failures therefore do not silently discard completed AI results.

---

## Model runtime

Logical model profiles select backend/runtime behavior:

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
local model backend
```

Profiles own deployment-specific details such as:

```text
backend
model_path
quantization
compute_dtype
model_dtype
device_map
offload_folder
```

Supported quantization profile values currently include:

```text
auto
none
bnb4
fp8
```

Runtime inference passes through:

```text
InferenceCoordinator
        |
        v
GpuScheduler
        |
        v
ModelManager
```

Current scheduling behavior includes serialized accelerator admission, priority ordering, FIFO inside equal priority, cancellation safety, and runtime/GPU metrics.

---

## Learning architecture

The learning stack is deliberately separated from live execution authority.

```text
runtime evidence
    |
    v
provenance / sanitation
    |
    v
filtering / deduplication
    |
    v
curation / review
    |
    v
versioned dataset/materialization
    |
    v
candidate training
    |
    v
fixed evaluation / regression
    |
    v
promotion gate
    |
    v
adapter/model activation
```

Current repository support includes:

```text
runtime evidence and rewards
structured Jira/Git/shell/task context
corpus registry and materialization
decontamination and replay
reviewed datasets and preference data
curriculum chapters/mastery/planning
training membership
Jira specialist SFT
DPO / QLoRA infrastructure
PPO infrastructure
Phase 5 hybrid QLoRA
continual-learning cycles and automation
promotion gates and rollback-oriented model lifecycle
developer holdout materialization
candidate patch generation
sandboxed developer behavioral evaluation
training guard diagnostics
adaptive underfit detection
```

Raw model output is evidence, not automatically trusted training truth.

### Current learning focus

The developer learning/corpus bridge, grounded candidate generation, holdout materialization, sandboxed behavioral evaluation, and adaptive underfit diagnostics are implemented.

The current next item is **LEARN-005**: train and gate the Jira specialist on contextual, paraphrased, and typo-tolerant language.

Current bounded path:

```text
Jira hybrid SFT corpus
    |
    | 300 records
    | 236 train / 64 validation
    | all 14 Jira tools represented
    v
training preflight
    |
    | environment must match datasets==5.0.1
    v
bounded specialist training
    |
    v
held-out behavioral gate
    |
    +-- correct tool selection
    +-- argument extraction
    +-- paraphrase robustness
    +-- typo robustness
    +-- read/mutation semantic distinction
    +-- no authority leakage into model behavior
```

The model is being trained to understand Jira semantics. Authorization remains deterministic in `SemanticGuard`, `ToolGateway`, approval state, and trusted providers.

A candidate is not promoted because training loss improves; held-out behavior must pass.

---

## Configuration

Create a local environment file:

```bash
cp env.example .env
```

Do not commit real credentials.

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

Deployment-specific values belong in `Settings` and model profiles rather than scattered environment reads.

---

## Install and run

```bash
python -m pip install -r requirements.txt
```

Training dependencies are separated where possible:

```bash
python -m pip install -r requirements-training.txt
```

Current local environment:

```bash
conda activate qwen-infra
```

Start the AI service:

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

Phoenix remains the browser-facing application boundary.

---

## Useful preflights

```bash
python scripts/ldap_preflight.py
python scripts/atlassian_preflight.py
python scripts/jira_projects_preflight.py
python scripts/jira_project_create_preflight.py
python scripts/jira_project_update_preflight.py
python scripts/jira_project_archive_preflight.py
```

The repository also includes isolated approval/execution proof scripts for governed Jira mutations.

---

## Tests

Full fast suite:

```bash
python -m pytest -q
```

Jira-focused:

```bash
python -m pytest -q \
  tests/unit/services/jira \
  tests/unit/tools/jira \
  tests/unit/tools/ticketing
```

Git-focused:

```bash
python -m pytest -q tests/unit/tools/git
```

Whitespace:

```bash
git diff --check
```

Real-model, GPU, external-provider, and Docker behavioral evaluations are intentionally treated as controlled checkpoints rather than ordinary fast unit tests.

---

## Important source layout

```text
api/
agent/
config/

learning/
    cli/
    code_corpus/
    context/
    continual/
    curation/
    curriculum/
    datasets/
    evaluation/
    evidence/
    integrations/
    training/

services/
    atlassian/
    directory/
    jira/
    knowledge/
    ticketing/
    process_runner.py
    workspace_repositories.py
    git_repositories.py

subagents/
    agents/
    core/
    llm/
    prompts/

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

scripts/
tests/

mcp_server.py
```

---

## Cross-repository SDLC visibility

Cross-repository implementation status is tracked outside the AI runtime by the local `projectOps` workspace.

```text
Git history from AI / backend / frontend
        +
explicit config/status_catalog.json
        |
        v
projectOps board + weekly digest
        |
        +--> Notion SDLC board
        |
        +--> Gmail stakeholder report
```

Important boundary:

- Git activity is evidence, not an automatic declaration of completion;
- status transitions remain explicit in `projectOps/config/status_catalog.json`;
- the reporting workflow does not grant or execute Jira/Git/shell authority;
- the reporting system does not require an LLM.

The weekly reporting/scheduler infrastructure has been validated end-to-end and is tracked as implemented. The current engineering focus remains the Jira specialist learning gate described above.

---

## Current deliberate limitations

The system is still an evolving MVP.

Current gaps include:

- Hub/specialist semantic robustness is not yet strong enough across all natural-language Jira/Git/shell requests;
- developer corpus/evaluation alignment is still being hardened;
- developer behavioral parser coverage is narrower than the upstream SWE corpus;
- Phoenix public job lifecycle does not yet expose first-class `denied` and `outcome_unknown` states;
- Jira/JSM provider coverage is still being expanded beyond the currently implemented ticket/project surface;
- broad enterprise RBAC/ABAC is not complete;
- some ITSM providers remain mock-backed;
- frontend lifecycle updates currently use polling;
- frontend capability discovery is not fully dynamic;
- GPU scheduling is process-local and optimized for one local workstation;
- continual learning is gated/offline-oriented rather than unrestricted live self-training;
- deployment/bootstrap automation remains separate from runtime authorization.

These are capability/product gaps, not reasons to weaken execution safety.

---

## Development rules

1. Model output is semantic intent or a proposal, never authorization.
2. Prefer domain-specific typed capabilities over arbitrary shell/HTTP access.
3. Keep model-visible arguments minimal and user-grounded.
4. Keep provider URLs, credentials, IDs, translation logic, and irreversible policy trusted-side.
5. Bind mutable state before approving dangerous actions.
6. Revalidate trusted state immediately before the side effect.
7. Verify provider state afterward.
8. Never blindly retry an ambiguous remote mutation.
9. Add tests at serialization and authority boundaries.
10. Keep learning-data provenance and evaluation isolated from production authority.
11. Do not promote a candidate because training loss improved; require behavioral evaluation.
12. Preserve the separation between semantic understanding and deterministic execution authority.

---

## Design summary

```text
Models understand intent.
Trusted code owns authority.
Phoenix owns durable product state.
Learning improves behavior only through governed evidence, evaluation and promotion.
```
