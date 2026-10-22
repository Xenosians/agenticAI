# Tool browser E2E

`scripts/tool_browser_e2e.py` drives the real frontend in Chromium (Playwright) and
types one natural request per registered tool, the way a user would. Mutating tools
raise an approval card and the harness clicks **Approve action** in the UI. Each case
passes only when the expected tool ran end to end (Hub -> specialist -> SemanticGuard ->
ToolGateway/MCP -> provider), the provider reported success, the chat history round-trips,
and at least one live-update WebSocket frame arrived during the case. An already
open socket is supported; frames from earlier cases are not reused as evidence.

The cases (prompt, expected tool, approval) are shared with the API-level live matrix in
`scripts/tool_matrix.py`, so both suites exercise the same requests.

## Run

From the AI repository root, with the stack running (frontend :8080, Phoenix :4000,
AI service :8000) and the acceptance login in the backend's
`.runtime/acceptance/login.env`:

```bash
python scripts/tool_browser_e2e.py                 # every case
python scripts/tool_browser_e2e.py --only ticket-create ticket-get
python scripts/tool_browser_e2e.py --resume        # continue only cases never recorded
python scripts/tool_browser_e2e.py --headed        # watch the browser
python scripts/tool_browser_e2e.py --report .runtime/acceptance/new-browser-run.json
```

Optional environment: `ACCEPTANCE_JIRA_PROJECT_KEY`, `ACCEPTANCE_JIRA_ASSIGNEE`,
`ATLASSIAN_ACCOUNT_ID`, `ATLASSIAN_ORG_ID`, `ACCEPTANCE_GIT_FIXTURE`.

The default report is `.runtime/acceptance/tool-browser-e2e.json`; failure screenshots
go to a sibling directory named after the report without its extension. An existing
report is never silently overwritten. `--resume` requires the same source hashes,
project, frontend/backend URLs, login configuration location and Git fixture location.
It retains the original fixture run identity and binds further execution to the same
acceptance account. Legacy reports cannot supply passing evidence for this version.

Results merge only within this validated report. `acceptance` judges the explicitly
recorded selection and scope, separate from merged historical counts. Source changes
during a run invalidate its passing acceptance result. Intermediate checkpoints are
incomplete until the final source check; no zero-pass run can exit successfully.

A case is recorded before any UI submission, and observed chat/submission/job
identities are saved as soon as available. A lost response keeps those identities
and an unresolved outcome. Resume skips **every** recorded case, including failed,
blocked, skipped and interrupted cases: it never creates a replacement job merely
because the first attempt did not pass. Reconcile existing jobs/provider state before
starting a separate evidence run for a previously attempted action. Do not delete a
row to force a retry. An interrupted Git fixture preparation is also held for
reconciliation; fixture mutations are not repeated automatically.

Archived disposable Jira projects are listed under `cleanup_required` for explicit
operator cleanup. The browser harness does not restore/delete them automatically
after a run or on resume.

Exit status: 0 only when every selected case passed (and, without `--only`, every
registered tool was covered); 1 when a case failed; 2 when the run is incomplete
(blocked, skipped or plan-limited cases, or uncovered tools). An unknown `--only`
case id is rejected before the browser starts.

## Verdicts

| Verdict | Meaning |
|---|---|
| `pass` | The expected tool ran and the provider reported success. |
| `plan_limited` | Routed, approved and executed, but the provider account's plan does not include the API (tool result `error_code: plan_required`). Example: Atlassian user invitations need a paid subscription, so they are plan-limited on the free plan. Not a product failure. |
| `blocked` | A prerequisite is missing or an in-flight/interrupted case needs reconciliation. |
| `skipped` | The case needs context from an earlier case (for example a ticket key) that is missing. |
| `fail` | Anything else. A screenshot is saved. |

`tool_matrix.py` uses the same verdicts.
