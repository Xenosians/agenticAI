# Architecture

Janus separates presentation, durable coordination, inference and governed ITSM
execution across three repositories. Models, provider endpoints and credentials
are deployment/account selections; the application does not require one named
model or commercial provider. This describes source boundaries, not a declaration
that every provider or deployment passed live acceptance.

## Runtime and authority

```text
Nim/Karax browser
    -> Phoenix API: session, owner, CSRF, chats, durable jobs and request intents
        -> Python AI: configured Hub, specialists, inference and reference context
            -> Lua adapter: selected protocol/data mapping, without credentials
            -> strict proposal parser / SemanticGuard / ToolGateway
                -> read: registered MCP capability
                -> mutation: exact durable approval, then registered MCP capability
                    -> configured provider adapter -> receipt and readback
        <- durable completion callback / saved job state
    <- authenticated WebSocket progress and authoritative state reconciliation
```

| Component | Responsibility | Cannot establish |
| --- | --- | --- |
| Browser | Render actual state; select owned connections/resources; request decisions | Ownership, execution permission or completed effects |
| Phoenix | Authentication, owner/session scope, request identities, durable jobs, workflow versions and approval coordination | Provider success from an HTTP acknowledgement |
| AI orchestration | Route to registered specialists; generate and validate proposals; compose evidence-backed answers | Authority from model output, prior chat, RAG or a workflow |
| Lua adapters | Map an admitted protocol using real `.lua` artifacts and pinned bindings | Credentials, additional grants, a replacement approved target or invented receipts |
| SemanticGuard / ToolGateway | Validate registered capabilities, arguments, grounding and effect/approval policy | Permission from an inference provider or plugin |
| MCP / provider adapters | Execute the admitted operation and preserve actual provider observations | Verified completion from malformed or missing evidence |

The public browser talks to Phoenix. Internal AI requests use service credentials.
WebSockets carry progress/reconciliation hints and lease heartbeats; they do not
execute user tools or turn an uncertain operation into success.

## Durable requests and outcomes

Phoenix reserves an owner-scoped submission intent before job, workflow or
candidate-training submission. It pins the exact request and existing operation
identity. Recovery reads server state and retries only through the same explicit
identity; it does not depend on browser local/session storage or automatically
dispatch work when the page reloads.

Mutations require the saved exact action and arguments to be approved. Repeated
decisions must read the durable outcome instead of performing the action again.
An MCP/transport failure can happen after the provider acted. Missing,
contradictory or unverified receipts remain unresolved and require reconciliation.
Provider acceptance, successful readback and final mail delivery are distinct.

See [provider outcomes](docs/provider-outcomes.md), the canonical Phoenix
contracts and [cross-repository acceptance](docs/cross-repo-acceptance.md).

## Models, harnesses and Lua plugins

Specialist definitions under `subagents/agents/` declare their tool allowlists.
The capability registry and gateway independently enforce those limits. A worker
proposal currently contains one tool call per task; saved workflows compose
governed steps rather than granting unrestricted model tool access.

`services/extensions/context.py` resolves an immutable owner/role selection.
`subagents/llm/runtime/inference.py` applies stage policy, disclosure/budget checks,
the scheduler and local/outbound generation. Pi/OpenCode and hosted model
connections supply inference through the same governed application path.
Networking retains its explicit routing/private-context restrictions.

Machine `.lua` profiles declare deployment links and model references. Named
`.lua` codec files implement protocol behavior. The isolated worker sees bounded
data; host code resolves protected credential references, validates the output and
performs network/model/process I/O. A missing/invalid required adapter refuses
dispatch; there is no silent Python-only substitute.

Normal model/API/dataset uploads and connection settings use the frontend.
Integrity digests and credential encryption are host responsibilities. A digest
is an identity/integrity check, not encryption or permission to install a plugin.
Custom machine-profile/plugin installation is currently operator-managed; a
complete owner-scoped frontend Lua plugin lifecycle is still separate work.

See [Lua adapters](docs/lua-adapters.md), [protocol plugins](docs/lua-plugins.md),
[machine profiles](docs/lua-machine-profiles.md) and
[user extensions](docs/user-extensions.md).

## Existing feature boundaries

| Area | Source boundary | Scope and limits |
| --- | --- | --- |
| ITSM tools | `tools/registry.py`, `tools/`, `services/` | Registered directory/access, ticketing, Git/workspace, operations, knowledge, mailbox and network adapters; actual setup and provider permissions still matter |
| Ticketing | `services/ticketing/`, `tools/ticketing/` | Jira reads and selected governed writes; no automatic migration of a failed Jira action to another provider |
| Workflows | Phoenix `Workflows`, `api/workflows.py`, frontend workflow canvas | Saved/versioned capability request contracts and durable runs; not a general arbitrary-code graph runtime |
| Engineering | `services/engineering/`, `api/engineering.py` | Read-only topology drift and KiCad netlist inspection; labelled advisory/simulated, with no infrastructure or physical actuation |
| RAG | `services/extensions/{store,context,service}.py` | Owned dataset versions, local retrieval, bounded untrusted context and exact citation resolution; external connector integration remains incomplete |
| Learning | `services/learning_control/`, `learning/` | Readiness/admission, candidate SFT controls and durable artifacts; DPO/PPO web dispatch and automatic serving promotion are not implied |
| Deployment | `Dockerfile`, `docker/`, `deploy/` | Compose, explicit database/schema bootstrap, GPU/profile/private-CA overlays and persistent state; readiness and provider E2E need independent evidence |

Use configured real providers explicitly. Mock/demo adapters are test selections,
not fallback evidence of a real provider's effect. A `.test` mailbox lab does not
provide public Internet verification mail for external accounts.

## Incremental architecture roadmap

The 30 October architecture review is an advisory source snapshot. SRS 1.8
remains the MVP trust/release baseline; SRS 1.9 work extends it without weakening
the existing gate. These work packages are not claims of implementation:

1. **Inference failover.** Persist an explicit ordered owner/specialist policy.
   Resolve eligibility before disclosure: consent, privacy/residency, stage,
   context support and shared total budgets. Record bounded attempts and health
   observations/circuit state. A connection probe is not a circuit breaker.
   Put inference-only switching at the coordinator; never wrap tool execution,
   approvals or workflow continuation in a retry loop. Security/policy refusals,
   denied approvals and unknown tool effects are not inference-availability
   failures and must never trigger failover. Existing format repair is not general
   provider failover.
2. **Consistent UI typography.** Share semantic size/font tokens, include native
   select/option controls and keep code/log/identifier monospace intentional.
   Verify computed styles, focus, responsive layout and zoom on actual views.
3. **Additional capabilities and a second ticket provider.** Extend the existing
   registry/adapters. Pin provider instance, target, preconditions and approval;
   execute once and read back. Evaluate GLPI/API/license compatibility separately
   before adoption. Provider identity must survive results and recovery; a failed
   Jira write must never silently become a GLPI write.
4. **Separate host operations agent.** Keep machine inspection, package/service
   management and any per-operation privileged broker outside the unprivileged AI
   container. Use authenticated host-scoped typed requests, protected paths,
   reviewed repositories/templates, deadlines, audit and recovery. Do not turn
   the existing restricted process runner into an arbitrary root shell.
5. **Executable workflow IR.** Start with one typed/versioned intermediate
   representation and one emitter, stable node/edge IDs, explicit secret
   references, bounded control flow and durable per-node state. Generated code
   uses the same capability/approval path. Saved canvas/request contracts alone
   are not a compiler, resumable graph executor or source round-trip.
6. **Deployment and further learning/RAG.** Keep images model/credential-free,
   declare all providers and verify bootstrap, TLS, sockets, persistence and
   interrupted outcomes. Kubernetes remains conditional. Preserve reviewed data
   admission and SFT -> DPO -> PPO ordering, independent evaluation and separately
   authorized promotion; a candidate run must not modify serving weights.

## Verification

Run appropriate isolated source/contract tests, frontend build/browser checks and
backend `mix precommit` for affected changes. The repo acceptance command is
documented in [cross-repository acceptance](docs/cross-repo-acceptance.md); inspect
its runtime/database configuration before running it. Do not run tests against
an application database or mail real recipients as an incidental test setup.

Keep source checks, model interpretation, provider restrictions, tool correctness,
container readiness and real local/outbound E2E results separate. Skipped,
unconfigured, denied-by-plan and outcome-unknown cases are incomplete acceptance.
Use the [release gate](docs/mvp-release-gate.md) for the actual MVP decision.
