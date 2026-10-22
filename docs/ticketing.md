# Ticketing (Jira and mock providers)

Ticket tools are owned by both `jira-specialist` and `ticket-specialist`. They
talk to the provider selected by `TICKETING_BACKEND` (`jira` or `mock`) through
`services/ticketing`: `TicketService` for reads and `TicketMutationService` for
changes. The model only proposes one call; SemanticGuard checks that every
grounded value appears in the user's request, the trusted policy prepares
mutations read-only, the user approves, and only then does the gateway execute.

Credentials (`JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`,
`JIRA_HTTP_TIMEOUT_SECONDS`) come from settings and never appear in tool
arguments or results.

## Reads

| Tool | Provider call | Notes |
|---|---|---|
| `ticket_get` | `GET /issue/{key}` | One ticket's metadata; not comments or history. |
| `ticket_search` | `POST /search/jql` | Project, status, priority and text filters; 1–25 results. |
| `ticket_history` | `GET /issue/{key}/changelog` | Field changes with authors and times. |
| `ticket_comments` | `GET /issue/{key}/comment` | Existing comments only. |
| `ticket_links` | `GET /issue/{key}?fields=issuelinks` | See below. |
| `ticket_assigned` | user lookup + `POST /search/jql` | See below. |

### `ticket_links`

Lists every link of one issue as a sentence from that issue's perspective:
"SUP-9 blocks SUP-8", "SUP-8 is blocked by SUP-9", with the other issue's
summary and status. Jira reports the phrase per side (`outwardIssue` uses the
type's outward phrase, `inwardIssue` the inward one), so no direction logic is
needed when reading.

### `ticket_assigned`

Lists tickets assigned to one explicitly identified user: a Jira account ID, or
an exact (case-insensitive) email address or display name that
`/rest/api/3/user/search` resolves to exactly one account. Ambiguous or unknown
users are refused with a clear message; nothing is guessed. Self-references
("me", "myself", "current user") are refused because the provider
authenticates as the integration's service account, which is not the person
asking. Optional exact `status` filter and `limit` (1–25). The query reuses the
JQL escaping and search decoding of `ticket_search`.

Email addresses are only matched when Jira returns them; with restrictive
profile visibility, use the account ID or display name.

## Mutations

All mutations are `risk: medium`, need approval, and follow the trusted
prepare → approve → revalidate → execute → read back pattern:

1. The policy reads a provider snapshot **before** approval and adds it as
   `expected_*` trusted arguments, which the model cannot propose.
2. At execution the snapshot is read again; any difference refuses the call
   (`status: denied`, nothing sent).
3. The request is sent exactly once. Provider refusals (400/401/403/404/409/422)
   are `denied` with Jira's message, HTTP 402 is `denied` with
   `error_code: plan_required`, 429 is `rate_limited`, transport failures and
   5xx are `outcome_unknown` (no automatic retry).
4. The result is read back; anything other than exactly the approved state is
   `outcome_unknown`.

| Tool | Provider call | Approval text |
|---|---|---|
| `ticket_add_comment` | `POST /issue/{key}/comment` | Adding a comment to KEY |
| `ticket_create` | `POST /issue` | Creating a ticket in project P |
| `ticket_assign` | `PUT /issue/{key}/assignee` | Assigning KEY |
| `ticket_transition` | `POST /issue/{key}/transitions` | Changing the status of KEY |
| `ticket_update` | `PUT /issue/{key}` | each change as old → new |
| `ticket_link` | `POST /issueLink` | "SUP-12 blocks SUP-9" |
| `ticket_worklog_add` | `POST /issue/{key}/worklog` | Logging 15m of work on KEY |

### `ticket_update`

Changes priority, labels and/or due date of one issue. Arguments: `ticket_key`
and at least one of `priority` (exact name), `add_labels`, `remove_labels`
(exact labels, no spaces) and `due_date` (exact `YYYY-MM-DD`; relative phrases
such as "next Friday" are refused). An empty update is refused.

- Prepare: issue snapshot with current priority, labels and due date;
  `GET /issue/{key}/editmeta` must list each requested field with the needed
  operation (`set`, `add`, `remove`); the priority name is resolved against
  `GET /priority` (and the field's allowed values), unknown names are refused
  with the valid options.
- Execute: one `PUT` with `fields` for priority/due date and `update.labels`
  add/remove operations, never a full label replace. Values that already match
  are skipped; when nothing would change the result is success with
  `changed: false` and no request.
- Result: `changes` lists each field as old → new.

Limit: fields hidden from the issue's edit screen cannot be changed. In the lab
(SUP project), `duedate` is not on the screen, so due-date updates are refused
there with a clear message.

### `ticket_link`

Links two existing issues. Arguments: `ticket_key`, `other_ticket_key` and the
user's `relationship` words such that "`ticket_key` `relationship`
`other_ticket_key`" states the request ("blocks", "is blocked by", "duplicates",
"relates to", or a type name for symmetric types such as "relates").

- Prepare: the words are matched exactly (case and spacing aside) against the
  inward/outward phrases of `GET /issueLinkType`; unknown or ambiguous words are
  refused with the list of valid phrases. Both issues are snapshotted, the
  `LINK_ISSUES` permission is checked, and an existing identical link is
  detected.
- Execute: Jira states a link as "<inwardIssue> <outward phrase>
  <outwardIssue>", so "A blocks B" is posted with `inwardIssue: A` and
  `outwardIssue: B`. This was verified live by reading the links back from
  both issues. An identical existing link returns success with
  `changed: false` and no duplicate; for symmetric types a link created from
  either side counts.
- Result: `link_id` and the sentence in `new_value`.

Limit: links cannot be removed through this tool.

### `ticket_worklog_add`

Logs time on one issue. Arguments: `ticket_key`, `time_spent` (exact Jira
duration: `w`, `d`, `h`, `m` units in that order, e.g. `15m`, `2h`, `1h 30m`),
optional exact `comment`.

- Prepare: issue snapshot, time tracking must be enabled, the duration is
  converted to seconds with the site's working hours per day and days per week,
  and the `WORK_ON_ISSUES` permission is checked.
- Execute: one `POST` with the comment in ADF (the same document builder as
  `ticket_add_comment`) and `adjustEstimate=leave`, so the remaining estimate is
  not touched. The new worklog is read back by id and must match the issue,
  seconds and exact comment.
- Result: `worklog_id`.

Limit: worklogs cannot be edited or deleted through this tool, and the start
time is the provider's default (now).

## Mock provider

`MockTicketService`/`MockTicketMutationService` keep all state in memory and
implement every tool with the same validation rules
(`services/ticketing/validation.py`), so the read tools observe mock mutations
immediately. Jira snapshot arguments are refused by the mock provider.

## Lab notes

- Site `fiosalis.atlassian.net` (free plan), projects `SUP` and `ITSM`. The
  link types are Blocks, Cloners, Duplicate, Problem/Incident and Relates; the
  priorities are Highest, High, Medium, Low and Lowest.
- The live matrix (`scripts/tool_matrix.py`) runs these tools after the ticket
  cases, on the ticket it created. `discover_fixtures` adds `other_ticket` (the
  newest issue in `ACCEPTANCE_JIRA_PROJECT_KEY`) and `ticket_priority` (an
  allowed priority different from that issue's) read-only. `ticket-assigned`
  needs `ACCEPTANCE_JIRA_ASSIGNEE` (an account ID, exact email or display name).
