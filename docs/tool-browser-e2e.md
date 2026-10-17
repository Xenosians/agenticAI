# Tool browser E2E

`scripts/tool_browser_e2e.py` drives the real frontend in Chromium (Playwright) and
types one natural request per registered tool, the way a user would. Mutating tools
raise an approval card and the harness clicks **Approve action** in the UI. Each case
passes only when the expected tool ran end to end (Hub -> specialist -> SemanticGuard ->
ToolGateway/MCP -> provider), the provider reported success, the chat history round-trips,
and the live-update WebSocket opened.

The cases (prompt, expected tool, approval) are shared with the API-level live matrix in
`scripts/tool_matrix.py`, so both suites exercise the same requests.

## Run

From the AI repository root, with the stack running (frontend :8080, Phoenix :4000,
AI service :8000) and the acceptance login in the backend's
`.runtime/acceptance/login.env`:

```bash
python scripts/tool_browser_e2e.py                 # every case
python scripts/tool_browser_e2e.py --only ticket-create ticket-get
python scripts/tool_browser_e2e.py --resume        # repeat only cases without a final verdict
python scripts/tool_browser_e2e.py --headed        # watch the browser
```

Optional environment: `ACCEPTANCE_JIRA_PROJECT_KEY`, `ACCEPTANCE_JIRA_ASSIGNEE`,
`ATLASSIAN_ACCOUNT_ID`, `ATLASSIAN_ORG_ID`, `ACCEPTANCE_GIT_FIXTURE`.

The report is `.runtime/acceptance/tool-browser-e2e.json`; failure screenshots go to
`.runtime/acceptance/tool-browser-e2e/`. Results merge by case id, so `--only` never
discards other cases.

## Verdicts

| Verdict | Meaning |
|---|---|
| `pass` | The expected tool ran and the provider reported success. |
| `plan_limited` | Routed, approved and executed, but the provider account's plan does not include the API (tool result `error_code: plan_required`). Example: Atlassian user invitations need a paid subscription, so they are plan-limited on the free plan. Not a product failure. |
| `blocked` | A required credential is not configured (for example `ATLASSIAN_ADMIN_API_KEY`). |
| `skipped` | The case needs context from an earlier case (for example a ticket key) that is missing. |
| `fail` | Anything else. A screenshot is saved. |

`tool_matrix.py` uses the same verdicts.
