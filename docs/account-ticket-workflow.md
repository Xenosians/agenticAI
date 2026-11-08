# Account → Jira ticket → mailer

The product's durable job path supports one bounded onboarding workflow. A user
requests a corporate directory account, then a Jira ticket about that created
account in an explicitly named project. Phoenix sends its existing completion
notification after the final verified ticket result. Brevo remains the outbound
mailer; directory creation does not provision a mailbox or grant Jira access.

For example, supply the actual intended identity and configured project:

> Create a corporate account for given name Alex, family name Example, department
> IT and role Engineer. After the account is successfully created, create a Jira
> ticket in project MYPROJECT for that created account, then notify me by email.

Those are illustrative request values, not application defaults. The server
never supplies a project, identity, recipient, password, or successful result
from this example.

## Execution

For explicit completeness checking, open **Required operations** in the chat
composer, select the account and ticket capabilities, and choose the ticket's
supported success dependency. Review the steps before sending. Optional target
fields pin their exact value lists; fields without pins still require grounding
from the original message. Include the intended identity/project in that message.
This review declares requested work; it does not approve execution.

Phoenix persists the reviewed contract and its content hash with the owner,
chat and submission identity. Retry/reload recovery uses that same message and
contract. A changed contract conflicts with the original submission. Both local
and outbound routing receive the selected scope and must return every requested
operation in order with matching dependencies and pinned targets. A mismatch
shares the existing single repair attempt, then stops with clarification and
no specialist dispatch. The system never fills in a missing operation itself.
The original free-text path remains available; without a reviewed contract it
does not certify whole-plan completeness.

1. The selected local model or outbound inference adapter proposes the account
   and ticket delegations from the current request. The ticket delegation uses
   `after_success_of` to refer to the preceding account specialist. The account
   capability's trusted `success_continuations` metadata discloses the supported
   follow-up. Independent operations are not silently converted into a chain.
2. Normal specialist validation, SemanticGuard and ToolGateway prepare the exact
   account approval. The validated remaining delegation, original request,
   conversation context, owner, job/attempt and selected resource IDs are saved
   in the approval database before Phoenix exposes the approval.
3. Approval executes the saved account action once. The continuation requires a
   successful directory result, provider read-back, encrypted credential
   persistence, and a username/email matching the approved policy snapshot.
4. The saved Jira specialist runs once using the original project scope and
   selected inference connection. It sees only the verified username/email;
   passwords and credential values are excluded. Its ticket summary must link
   the verified account. The Hub does not reconstruct the already approved
   account action.
5. The next exact Jira proposal replaces the active proposal on the same durable
   job. It requires a separate approval. No Jira action runs during continuation
   preparation. Phoenix retains the preceding account approval evidence.
6. Jira executes its saved proposal through the existing provider, including
   approved-state revalidation and result read-back. Completion mail follows
   the terminal durable job result through the existing mail outbox.

## Handing over the temporary password

The new person's own mailbox does not exist yet, so the temporary password is
never mailed to it, and nobody has to sign in as the new person to receive it.
The person who asked for the account hands it over:

1. After approval, the answer offers **Reveal temporary password** (also after a
   reload, and after the onboarding ticket step whatever its outcome).
2. The requester confirms with their own Janus password. Wrong passwords count
   against the normal sign-in limit.
3. Janus shows the password once, on one line, with **Copy password**. It is not
   stored in the browser and cannot be revealed again; Phoenix erases the
   ciphertext in the same step that marks it revealed.
4. The person signs in with it and must change it at first sign-in. If it was
   lost or has expired, request an approved password reset; that reset's
   password is handed over the same way.

Only the owner of the request can reveal it, and only for a credential that an
approval Phoenix recorded as approved and succeeded names. A new local mailbox
password and a password reset use the same hand-over. The API and its refusals
are described in the backend's `docs/account-controls.md`.

## Recovery and limits

- SQLite commits a generation claim before preparing the follow-up proposal.
  Concurrent/replayed decisions reuse one saved proposal ID. An interrupted
  claim is not automatically regenerated or executed; it requires trusted
  reconciliation. No automatic mutation retries are introduced.
- The original owner-scoped model and dataset IDs are pinned. Secrets remain
  in the existing encrypted extension store. Deleting a pinned resource blocks
  continuation instead of silently selecting a different model.
- A reviewed contract/hash also remains bound to the saved continuation. Its
  ticket step is rechecked against current capability metadata before follow-up
  inference. An inconsistent saved scope stops for reconciliation.
- Rejection, failed account verification, an identity mismatch, invalid ticket
  scope or an unrelated ticket proposal stops the workflow. A partially created
  account is reported truthfully; the server does not claim rollback. When the
  specialist's proposal is refused before any approval exists (for example
  `semantic_required_argument_missing` for an omitted summary), the job error
  names that stable runtime code; model output never reaches the job.
- This is a durable-job feature. The transitional synchronous AI endpoint does
  not provide workflow persistence. Arbitrary graphs, public mailbox creation,
  Jira product invitations and automatic account cleanup are outside this
  bounded workflow.
- Reviewed requests currently permit at most eight independent operations whose
  possible specialist owners do not overlap, or one declared two-step success
  continuation. A capability with several registered owners (for example the
  provider-specific Jira specialist and the provider-neutral ticket specialist)
  may be carried by any of them; a dependency must name the specialist that
  carried its source step. Unsupported repeated-specialist or dependent graphs
  fail admission. Selecting a registered capability does not establish provider
  configuration or readiness.
- Provider credentials and permissions must be configured. Unit tests verify
  orchestration and recovery contracts; they are not evidence that a live LDAP
  account, Jira issue, or external email was created or delivered.

Implementation: `agent/onboarding.py`, `agent/workflow_store.py`,
`api/workflows.py`, the existing orchestrator and the Phoenix approval controller.
Focused checks: `tests/unit/api/test_onboarding_continuation.py`.
