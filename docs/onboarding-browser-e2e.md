# Browser Onboarding End-to-End

`scripts/onboarding_browser_e2e.py` drives the real frontend in Chromium
(Playwright) through a full onboarding flow, with mail delivered to and read
from the local mailbox lab:

1. Register a new application user whose email is a fresh lab mailbox.
2. Open the verification link from the email in that mailbox, then sign in.
3. On the Models page, assign the Jira and Ticket specialists to the Groq
   connection. The selected model (Claude) runs the hub.
4. In chat, request a corporate account followed by a Jira ticket about it.
   Approve the account card, then the ticket card the workflow proposes.
5. Confirm that the completion email names both the account and the ticket.

Each step is recorded in
`.runtime/acceptance/onboarding-e2e/onboarding-<run>.json`, with screenshots
alongside.

## Trusted setup (no UI exists for these)

- The lab mailbox is created with Maddy's admin CLI. The product's mailbox tool
  requires an existing app user, and this user does not exist yet.
- Model connections are copied, encrypted, from the account named by
  `ACCEPTANCE_EMAIL` in the backend acceptance `login.env`. No API key appears
  in the output. The copies are deleted when the run ends.

## Prerequisites

- Phoenix, the AI service, the frontend and the `itsm-mailbox-lab` container
  are running.
- The backend `.env` sets `LAB_MAIL_SENDER` (an address at `MAILBOX_LAB_DOMAIN`)
  and `LAB_MAIL_PASSWORD`. Phoenix then delivers mail for lab addresses through
  the lab's SMTP submission port (`ItsmBackend.Mail.LabRelay`), and other
  recipients keep the configured provider. Lab delivery is never enabled in
  prod.
- The source account has a ready Claude connection (Anthropic Messages) and a
  ready Groq connection.

## Run

From the AI repository root, using the AI service's Python:

```bash
python scripts/onboarding_browser_e2e.py --project SUP          # headless
python scripts/onboarding_browser_e2e.py --project SUP --headed # watch it
```

`ACCEPTANCE_BACKEND_REPO` overrides the backend checkout location.

## Model mix

Each specialist can run on its own connection (Models page, "Specialist
models"):

| Role | Typical assignment |
| --- | --- |
| Hub: routing, synthesis | Selected model (Claude) |
| Jira, Ticket, Atlassian, Asset, Knowledge, Networking | Groq |
| Account, Access, Developer | Their own local fine-tuned models |

The `[INFERENCE_METRIC]` log line records `stage`, `role` and `adapter`
(`http` for an outbound API, `local` for an on-box model), so the log shows
which tier served each call. Groq's free tier allows about 8,000 tokens per
minute. To queue requests instead of failing on a 429, set the connection's
tokens-per-minute and admission wait limits.
