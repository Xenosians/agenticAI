# Compact routing and ITSM execution

Implementation status: 2026-10-16.

The MVP is an agent that can use the supported ITSM tools to complete a user's
request. Model and harness switching supports that objective. The next acceptance
target is the complete durable request and approval flow across all three services.

## What changed

The broad Hub prompt previously carried roughly 21,000 estimated input tokens,
even for a short request. That exceeded external request budgets before a provider
could answer. Increasing those budgets would preserve the avoidable prompt cost.

Selected model connections now receive a compact routing protocol and a locally
retrieved subset of the trusted capability catalog. No second model is needed for
candidate selection. The existing full catalog remains available for local control
routing and authoritative validation.

```text
User request
  -> local catalog search and candidate validation
  -> compact semantic routing on the selected connection
  -> full semantic-contract validation
  -> specialist proposal
  -> SemanticGuard, policy and any required approval
  -> ToolGateway and configured tool/MCP provider
  -> result-backed synthesis
```

With no selected connection, the existing local routing prompt remains in use.
The networking specialist retains its dedicated local model and receives no
additional private reference dataset context.

## Candidate selection is not authority

`DeterministicCandidateSelector` uses full-text search and coverage of trusted
capability names, descriptions and metadata. It contains no per-request keyword-to-
tool routing table. It preserves registered owners of shared tools and widens the
candidate set when the initial boundary is ambiguous.

Weak evidence, uncovered request clauses, unresolved ties or an oversized packet
cause broad local routing. Unknown specialists, unknown capabilities and invalid
ownership in a selector result are rejected. Compact descriptions are rebuilt from
the registry, rather than accepted from a selector or a user dataset.

A model proposal must use a disclosed candidate and also pass the full registry's
semantic contract. Candidate retrieval grants no permission, approval, argument
authority or execution capability. SemanticGuard and ToolGateway remain in place.
The router permits one fresh repair after a strict contract rejection; an invalid
plan is not executed.

## Explicit inference stages

Trusted callers supply a stage; a model name does not determine policy.

| Stage | Connection | Private reference evidence | External total/output ceiling |
| --- | --- | --- | --- |
| `CONTROL_ROUTING` | Configured local Hub | No | Not applicable |
| `CANDIDATE_NARROWING` | Local; current selector is deterministic | No | Not applicable |
| `SEMANTIC_ROUTING` | Selected connection, or local default | No | 5,000 / 768 |
| `SPECIALIST_REASONING` | Selected connection, or local default | Yes, if selected | 6,000 / 1,024 |
| `PRIMARY_RESPONSE` | Selected connection, or local default | Yes, if selected | 6,000 / 1,024 |
| `SYNTHESIS` | Selected connection, or local default | Yes, if selected | 6,000 / 1,024 |
| `NETWORKING` | Dedicated local profile | No | Not applicable |
| `REPAIR` | Inherits the explicitly named original stage | Inherits original stage | Inherits original stage |

Total ceilings include estimated input and reserved output. Lower provider/profile
budgets and caller output requests still apply. The application can drop older
history to fit, but preserves mandatory instructions and the current request.
Mandatory content that cannot fit is rejected before external dispatch. These are
conservative application estimates, not exact provider token counts or guarantees
about CLI-added overhead. Repairs cannot recursively repair another repair.

Every inference attempt records its stage, adapter, outcome, elapsed time and
available budget counts, including failures and cancellations. The dispatch event
does not include prompts, responses, credentials or exception text. GPU timing and
memory measurements remain local-only; remote calls report no invented GPU values.
Observation failures do not change inference results or errors.

## Verification evidence

The following bounded checks ran on 2026-10-14 using the configured Groq model.
They establish the stated paths only; they are not a latency benchmark or a claim
that every supported capability works against a live provider.

| Check | Result | Scope |
| --- | --- | --- |
| HTTP compact routing | Passed; 1.420 s | Proposed `jira_project_list`; no tool execution |
| Pi compact routing | Passed; 2.414 s | Same routing contract; no tool execution |
| OpenCode compact routing | Passed; 28.656 s | Same routing contract; no tool execution |
| HTTP Hub, specialist and Jira read | Passed; 5.634 s | Real `jira_project_list` through the guarded tool/MCP path, followed by synthesis |

The three routing checks each recorded 2,179 estimated input tokens and a 512-token
output reservation within the 5,000-token routing ceiling. They sent no private
dataset. The real read recorded successful semantic routing, specialist reasoning
and synthesis, plus a structured tool result. It did not exercise Phoenix or the
browser. Local development evidence is stored outside this repository in
`projectOps/.runtime/2026-10-14-compact-routing-acceptance.json` and
`projectOps/.runtime/2026-10-14-itsm-live-read.json`; those files are not required at
runtime and are not bundled as public test fixtures.

The initial ITSM task check on 2026-10-15 exposed provider failures:

| Adapter | Real Jira read and synthesis | Workspace creation proposal |
| --- | --- | --- |
| Groq HTTP | Passed; 7.339 s | Failed before execution: truncated provider output |
| Pi / Groq | Failed at specialist inference | Failed at routing inference |
| OpenCode / Groq | Passed; 118.487 s | Passed; 118.484 s; pending exact proposal survived reopening the store |

The OpenCode approval case made no MCP calls and executed no mutation. It used a
temporary approval database and did not leave a live approval for the user. A
diagnostic recheck classified Pi's failures as the account's observed Groq
8,000-token-per-minute quota being exhausted across requests. The HTTP approval
failure was `provider_output_truncated`. Neither limit was raised and no partial
proposal was executed. These single-run timings are observations, not benchmarks.
Reports live in `projectOps/.runtime/2026-10-15-itsm-task-acceptance.json` and
`projectOps/.runtime/2026-10-15-itsm-diagnostics.json`.

Owner-configurable request/token admission and supported reasoning controls now
address those observed causes. A shared process-local queue reserves estimated
input plus output before dispatch across HTTP, Pi and OpenCode using the same
provider/model/credential identity. Quota waits have a deadline; cancellation
removes queued requests. Failed requests retain their reservation, and reported
usage can increase it. There is no automatic retry of truncated proposals.

With the account's explicit 8,000 TPM setting and supported low reasoning effort,
HTTP and Pi both subsequently passed the real Jira read and pending workspace
proposal checks. HTTP took 5.022 s for the proposal and 60.315 s for the read;
Pi took 7.610 s and 58.899 s respectively. Those read timings include deliberate
quota waiting, not GPU inference. A fresh Pi proposal check after the final CLI
configuration change passed in 7.288 s. Evidence is recorded in
`projectOps/.runtime/2026-10-15-itsm-admission-acceptance.json` and
`projectOps/.runtime/2026-10-15-current-pi-approval.json`.

Pi's verified retry controls are disabled. OpenCode's internal retry policy has
no verified disable control in this integration: its invocation is reserved,
but this is not a claim that the application observes every internal attempt.
The admission queue is not shared across multiple server processes.

The authenticated Phoenix v2 check passed a real workspace read, a persisted
pending directory proposal, and clarification for a missing account identifier.
The original acceptance client chose a database record ID ahead of the public
chat ID, however, so those jobs did not appear in the browser's canonical chat
history. The client now uses the public chat ID, and the backend binds new jobs
to the canonical ID returned by the authorized chat lookup. The original HTTP
report alone does not establish browser success or approved execution.

Cold local checkpoint fingerprints are now computed outside the async event
loop. Routing evidence is activated in the caller's task so parse/repair traces
remain intact. Remote-selected routing and specialists do not fingerprint local
weights. This removes a responsiveness defect; it does not make local inference
or cold disk reads instantaneous.

The new runner checks real tool use and approval persistence without approving a
mutation. Use an existing admitted connection and its owning internal account ID:

```bash
ITSM_ACCEPTANCE_OWNER='<account-id>' python scripts/itsm_task_acceptance.py --profile '<connection-id>' --case all
```

It records partial progress after each case and returns a nonzero exit status if
any case fails. It never switches the account's saved model selection. The smaller
`scripts/compact_routing_acceptance.py` checks routing only.

Run focused checks from this repository:

```bash
python -m pytest -q tests/unit/subagents/core/orchestration/test_candidate_selector.py tests/unit/subagents/core/orchestration/test_compact_routing.py tests/unit/subagents/llm/runtime/test_inference_stages.py tests/unit/subagents/llm/runtime/test_stage_dispatch.py tests/unit/subagents/llm/runtime/test_inference_observability.py
```

Run the broader regression suite with `python -m pytest -q tests/unit`. Live model
and provider checks need separately configured local models, services and private
credentials. A connection probe alone is not an ITSM execution test.

## Current boundaries and next work

- Execution is bounded: one tool call per validated specialist intent, normally
  one unconditional delegation per specialist. An explicit validated result
  condition can add one conditional mutation; condition chains and loops are
  rejected. This is not an unrestricted recursive agent loop.
- The remaining durable acceptance work is a real authenticated request through
  Phoenix, persisted approval and decision handling, tool execution, completion
  delivery and the browser's final state. Denial and ambiguous mutation outcomes
  also need evidence at that boundary; unknown outcomes must not be blindly retried.
- Existing `denied` outcomes represent policy denial. Human rejection/cancellation
  of a saved pending approval still needs an explicit durable lifecycle; this
  implementation does not claim that feature. The cross-repository runner now
  validates canonical v2 responses by default; its pending proposal check does
  not prove approved execution, human rejection, or unknown-outcome recovery.
- Hosted inference remains subject to cumulative account quotas and output
  truncation. Structured failure categories expose these causes; they do not
  automatically retry requests or reinterpret incomplete output as success.
- Candidate retrieval can miss unfamiliar phrasing and use the local fallback.
  Routing and argument accuracy remain model-dependent; compact prompts do not
  eliminate hallucinations.
- An empty plan from compact semantic routing is an explicit routing failure,
  with at most one repair. It cannot fall through to a successful conversational
  completion of an ITSM task. Ordinary local conversational routing is unchanged.
- External RAG adapters and n8n execution remain future work. Review incoming
  source/dependencies and preserve owner isolation, evidence provenance and the
  existing guard/approval/gateway path. Current dataset retrieval is local FTS,
  and the workflow surface validates manifests without running them.

The supplied SRS 1.8 forward roadmap guided sequencing. Its companion solutions
document was treated as design suggestions: the implemented candidate stage uses
local retrieval, and harness-native tools do not replace application authority.
