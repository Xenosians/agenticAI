# Directory and Atlassian outcome verification

## LDAP security state

Account state reads require readable, single-valued nonnegative decimal security
attributes. The account's enabled state comes from `userAccountControl`. Current
lockout state comes first from the domain controller's constructed
`msDS-User-Account-Control-Computed` lockout bit, which also handles an account that
has never had `lockoutTime` written. If computed state is unavailable, only an
explicit readable `lockoutTime` can supply the legacy lockout check. Malformed
computed state is not hidden by falling back to a healthy-looking timestamp.
Absent, access-restricted, multivalued or malformed required evidence is unknown;
it never becomes an observed zero. A directory search failure or unreadable state
returns `ok: false` rather than enabled/unlocked success. Optional primary-group
metadata can still use an explicit fallback without granting that behavior to
security attributes.

A failed prerequisite read performs no write. If an attempted LDAP mutation
loses its response, or its acknowledged change cannot be verified, the result is
`outcome_unknown`, `changed: null`, `verification_ok: false` and `retry_safe: false`.
Inspect the directory through a trusted read/reconciliation path before another
approved action. Enable/disable MCP results preserve these fields. A successful
no-op requires readable provider state matching the requested state.

Unlock verification after a write requires an explicitly observed zero
`lockoutTime` and a cleared lockout state. Enable/disable verification requires
the requested `userAccountControl` state. Tests supply explicit evidence for
healthy fixtures and separately cover absent attributes, never-locked accounts,
computed lockout flags and missing values after an acknowledged modification.

## Atlassian product invitations

The implementation follows the official [Organizations REST API invitation
endpoint](https://developer.atlassian.com/cloud/admin/organization/rest/api-group-users/#api-v2-orgs-orgid-users-invite-post)
and its [published OpenAPI schemas](https://dac-static.atlassian.com/cloud/admin/organization/swagger.v3.json?_v=1.234.175).
HTTP 206 describes an invitation failure, so it cannot be treated as ordinary
2xx success. `AdminInviteResponse.id` identifies the invited **account**, not an
invitation. Results now expose `account_id`; the previous `invitation_id` field
remains empty for compatibility.

Success requires exactly the requested recipient, a bounded account identity,
and a successful `INVITED` outcome for the configured Jira resource and user
role. Each reported role/group outcome must be successful; missing, malformed,
partial, wrong-target or `ERROR` receipts remain uncertain. `verification_scope`
is `provider_invitation_receipt`: this confirms the provider's invitation result,
not the user's acceptance, sign-in or email delivery.

`sendNotification: true` requests provider email. The response schema contains
no sending or delivery receipt. Therefore `notification_requested` describes the
request, `notification_sent_by_provider` remains null, and
`notification_delivery_verified` remains false. The tool's answer explicitly
says that sending/delivery are unconfirmed.

Known pre-dispatch connection failures remain errors with no known mutation.
Ambiguous writes/reads, malformed success responses, partial responses, redirects,
server failures and unrecognized response statuses become `outcome_unknown` with
`retry_safe: false`. There is one POST and no automatic resend. For a lost
transport response, `request_dispatched` and `notification_requested` are null;
known pre-dispatch failures set both false. Provider refusal responses preserve
their clear denial/error classification without exposing raw response bodies.

Focused tests use production adapters with local LDAP doubles and injected
HTTP transports. They do not call real directory or Atlassian services or read
deployment credentials. Live provider compatibility and credential permissions
still require a separately configured acceptance run.
