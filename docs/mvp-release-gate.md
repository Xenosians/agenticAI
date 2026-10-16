# SRS 1.8 MVP release gate

Status: **MVP candidate**.

This is a non-normative engineering gate beneath SRS 1.8. It does not replace,
amend, weaken, or reinterpret SRS 1.8. If this document conflicts with SRS 1.8,
SRS 1.8 wins.

## Release conditions

MVP v1.0 may be declared only when current evidence proves:

1. The repository gate passes across AI, backend and frontend.
2. Authenticated live reads return durable, history-visible structured tool evidence.
3. A pending mutation proposal survives reload with the exact saved action.
4. Human rejection reaches durable `denied`, executes no tool, survives reload,
   and terminal decision replays are rejected.
5. One separately reviewed disposable mutation fixture proves exact approval
   binding, execution, verified result, durable completion and replay rejection.
6. A controlled ambiguous-mutation fixture proves `outcome_unknown` and no
   automatic mutation retry.
7. Browser smoke validation covers login, chat/history restoration, live status,
   approve, reject, interrupted-decision recovery and terminal rendering.
8. Authored-unreviewed tool acceptance cases are reviewed or explicitly excluded.
9. README and architecture documents describe the same capability/limitation set.

A connection probe, model JSON proposal, approval record, tool result, fixture, or
backend job by itself is not sufficient MVP evidence.

Until this gate is complete use **MVP candidate / MVP alpha**.
After it is complete use **MVP v1.0 — SRS 1.8 baseline**.

## Deferred third-party RAG and n8n source

RAG and n8n will arrive as third-party codebases and are deliberately outside this
MVP gate. After the baseline is frozen, review licenses/dependencies, filesystem,
network, subprocess and credential behavior; preserve ownership/provenance; keep
retrieval as evidence rather than authority; and route workflow mutations through
SemanticGuard, approval and ToolGateway. Imported code must adapt to the SRS
boundary, not silently redefine it.

**RUSH CAPABILITIES. DO NOT RUSH AUTHORITY.**

## Ambiguous mutation fail-closed gate

A deliberately ambiguous mutation may terminate as `failed` when the strict
routing contract cannot construct a trustworthy governed plan. It passes the
security acceptance gate only when durable evidence proves:

- no specialist authority was delegated;
- no tool proposal or approval was created;
- no application execution result exists; and
- the public durable error identifies governed-plan refusal.

Generic runtime failures do not satisfy this gate.
