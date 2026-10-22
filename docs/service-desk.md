# Service desk (Jira Service Management)

The service desk specialist (`subagents/agents/service-desk-specialist.md`) owns three
capabilities on the Jira Service Management (JSM) service desks of the configured Jira site.

| Tool | Risk | What it does |
|---|---|---|
| `service_request_create` | medium, approval | Raises one customer request of one named request type in one named service desk |
| `sla_status` | read | SLA metrics of one request: ongoing, paused, completed or not started, breached, goal, elapsed, remaining |
| `queue_list` | read | Queues of one service desk with the number of issues in each |

## Configuration

JSM uses the Jira ticketing settings; there is nothing extra to configure:
`TICKETING_BACKEND=jira`, `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`
(optional `JIRA_HTTP_TIMEOUT_SECONDS`). Without them every tool returns
`error_code: "not_configured"`.

The account needs agent access to the service desk. JSM is available on the free plan;
a plan refusal (HTTP 402) is reported as `error_code: "plan_required"`.

## Arguments

- `service_desk`: an exact project key (`SUP`) or service desk name (`Support`), compared
  without case. An unknown desk lists the desks that exist; a name shared by two desks is refused.
- `request_type`: an exact request type name such as `Get IT help`. An unknown name lists the
  desk's request types.
- `summary`: exact when the user gives one, otherwise composed from the requested outcome.
- `description`: optional, exact.
- `request_key`: a request key such as `SUP-12`.

## Raising a request

1. Before approval (read only), trusted policy resolves the desk and request type, checks that
   the account can raise that request type, and reads the request type's fields. A request type
   that requires any field other than summary and description is refused, because this tool
   never fills custom fields. A required description must be supplied.
2. The desk ID and request type ID are frozen into the approval as
   `expected_service_desk_id` / `expected_request_type_id`; the model never sees or sets them.
3. After approval the same checks run again. If the desk or request type changed, nothing is raised.
4. One `POST /rest/servicedeskapi/request`. A transport failure or server error after the request
   was sent is reported as `outcome_unknown`; it is never retried automatically.
5. The request is read back (`GET /rest/servicedeskapi/request/{key}`). Success requires the same
   desk, request type, summary and description; otherwise the result is `outcome_unknown` with the
   created key, so a person can check it.

## Reads

- `sla_status` reads `/rest/servicedeskapi/request/{key}/sla`. JSM computes SLA cycles shortly
  after a request is created, so a brand-new request can briefly report no metrics. A key that
  exists but is not a service desk request reports `not_found`.
- `queue_list` reads `/rest/servicedeskapi/servicedesk/{id}/queue?includeCount=true`
  (at most 50 queues; `truncated` says when there are more). It lists queue counts, not the issues.

## Results

Results carry `ok`, `status` (`success`, `denied`, `not_found`, `rate_limited`, `error`,
`outcome_unknown`), `error` and a stable `error_code`: `invalid_argument`, `not_found`,
`ambiguous`, `not_allowed`, `unsupported_fields`, `permission_denied`, `plan_required`,
`rate_limited`, `provider_error`, `not_configured`.

## Limits

- Customer-portal only fields (attachments, custom fields, participants, raise-on-behalf-of) are
  not supported.
- Desk and request type lookups scan at most 10 pages of 50.
- Requests are raised as the configured Jira account, not as the chat user.
