# Tool acceptance matrix

The MVP requires correct tool use and the complete application flow across local models, API connections, and harnesses. This matrix records what has actually been checked. A fixture, a model proposal, an approval, a tool result, and a durable job are different evidence.

## Live execution checker

`scripts/tool_matrix.py` uses actual configured providers and can execute reviewed mutations against its disposable lab fixtures. It is separate from the proposal-only runner below. Do not run it against production as a shortcut to a green result.

Its v2 reports use exit 0 only when every selected case passed. Exit 1 means a failed case; exit 2 means incomplete coverage, blocked/skipped/plan-limited cases, or source changes during execution. Unknown case IDs and empty selections are errors. Without `--only`, every registered tool must be covered; a passing subset is labeled selected scope. `all_registered_tools_passed` describes that run, not release acceptance across every model/harness or mailbox inbox delivery.

An existing report needs explicit `--resume`, matching source content/project and the original saved fixture identity. Resume continues unattempted cases; it never automatically repeats a recorded failure or uncertain operation. Reconcile those operations before planning another run. Legacy reports or reports from changed code cannot be reused as current passing evidence. New runs use new report paths.

## Inventory without inference

Run from the AI repository using the project's Python environment:

```bash
python scripts/tool_acceptance_matrix.py --require-full-coverage
```

The command registers executable signatures without starting MCP transport, validates the trusted capability contracts, loads specialist ownership, and checks authored evaluation targets against the production intent parser and SemanticGuard. It makes no inference or tool execution calls. The JSON report defaults to `.runtime/tool-acceptance-matrix.json`.

The current corpus covers all 67 registered tools. Existing specialist evaluation suites cover 49 tools; 18 additional tools have explicitly reviewed proposal-level acceptance cases in `config/tool_acceptance_reviewed_cases.json`. Newly authored future cases remain isolated in `config/tool_acceptance_authored_cases.json` until human review. Reviewed proposal cases are not production routing rules and do not prove that any provider or adapter executes successfully.

Prompts also present in version-controlled specialist training examples are excluded from held-out acceptance. This overlap check cannot establish what every underlying model has seen. Negative safety cases remain separate references rather than being mistaken for successful tool proposals.

`--require-full-coverage` fails for missing valid targets, stale targets, or unowned capabilities. `--require-reviewed-coverage` additionally fails when any registered tool lacks reviewed proposal coverage or any authored-unreviewed case remains. Passing either gate proves corpus and contract evidence only; it does not prove provider execution.

## Bounded worker generation

Generation is opt in. Choose explicit case IDs from the report:

```bash
python scripts/tool_acceptance_matrix.py --generate --adapter local \
  --model-key account-func-trained \
  --case 'account:status only' \
  --case 'account:unlock not enable' \
  --case 'account:account create exact fields' \
  --limit 3 --max-tokens 160 --timeout 120 \
  --report .runtime/tool-matrix-local-account.json
```

For an admitted connection, use `--adapter http`, `pi`, or `opencode` with `--owner` and `--model-id` identifying its existing owner-scoped stored resource. The runner retrieves its encrypted credential through the normal store and applies a temporary execution snapshot. It does not change the user's saved selection or include credentials in reports.

The runner uses the serving InferenceCoordinator and actual AgentRuntime. This preserves the runtime's current capability filtering, prompt profile, protected current request, inference stage, parser, and SemanticGuard. A gateway that always denies execution replaces the real ToolGateway: no approvals, provider tool calls, MCP execution, or mutations occur. Provider inference can still consume quota.

Raw worker output is scored independently of application-owned argument binding. A corrected application target cannot conceal model argument drift in this report. Exceptions are recorded by type; their text, raw output, credentials, and owner identifiers are omitted.

The trusted `networking-specialist` identity always uses the `NETWORKING` stage and dedicated local model. An external selection therefore records its effective adapter as local for that specialist and receives no external-model quality credit for the generation.

## Reading the result

Every tool has rows for `local`, `http`, `pi`, and `opencode`, with its actual required and optional arguments, grounding roles, approval requirement, specialist ownership, case provenance, and effective inference policy.

- `unverified`: no relevant generation was run.
- `missing_cases`: no eligible target exists.
- `partial`: only some eligible cases were checked.
- `failed`: at least one checked raw proposal failed.
- `passed`: all eligible existing evaluation cases passed the worker proposal checks.
- `passed_authored_unreviewed`: targets passed, with new cases still requiring review.

Approval, tool execution, and durable app flow remain unverified in this runner. Use the real orchestration and cross-repository acceptance tools to check those separately. `all_tools_mvp_verified` remains false because worker proposals alone cannot prove the MVP.

Live account → Jira → mailer acceptance remains incomplete. Check the exact
saved proposals, verified provider results, final durable job state and mail
delivery outcome together; WebSocket progress alone proves none of those
effects. Each inference adapter needs its own execution evidence, and the
protected networking specialist continues to use its local model.
