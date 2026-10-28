# User models, harnesses and datasets

Local models remain the default. The signed-in account can select another model
or harness in **Models, harnesses and datasets** (Ctrl/Cmd+,). Changes apply to
future requests. Networking uses its existing dedicated local model by default;
each connection can explicitly opt in to handling networking requests too.
The application database is not replaced by user datasets.

## Model connections

- **Model API:** base URL ending at the API root, model identifier and optional
  API key. Uses the common `POST /chat/completions` JSON protocol; no OpenAI
  account or SDK is required. Works with compatible local servers and providers.
- **Installed local model:** choose a enabled profile discovered from the runtime.
- **Pi / OpenCode:** installed stable CLI adapters, with explicit provider/model
  identifiers. The host currently has OpenCode 1.18.35 and Pi 1.1.0. Native tools,
  extensions and project configuration discovery are disabled. ITSM's existing
  parser, SemanticGuard, approval and ToolGateway remain the execution boundary.
  CLI calls use an isolated temporary home, bounded output and 120-second timeout.
  Their token limits are provider-specific; the adapter does not promise the same
  exact token cap as direct HTTP generation.
- **Uploaded model:** choose a single Transformers model folder containing
  safetensors weights, config and tokenizer files. Only data files are uploaded;
  custom Python code and pickle weights are not accepted. The server validates
  file references, tokenizer loading, supported causal architecture and estimated
  GPU capacity before registering a lazy 4-bit model. Format admission is not
  evidence of inference quality or a guarantee against runtime GPU exhaustion.

API keys are encrypted in the owner-scoped extension store and excluded from
catalog responses. Pi keyed profiles currently support explicit `openrouter/`,
`anthropic/`, `google/`, `groq/`, `mistral/`, `openai/`, `xai/`, and `cerebras/`
prefixes. OpenCode receives the selected provider key through its private process
environment and provider options. Account credentials from the host are not inherited.

### Which stages use the selected connection

Selecting a connection applies to compact semantic routing, specialist reasoning,
direct answers and synthesis. A deterministic local search first narrows the
trusted tool catalog. A weak, ambiguous or oversized candidate packet falls back
to broad routing on the configured local Hub model. Candidate selection does not
authorize actions: the full registry, semantic contract and ToolGateway still
validate the proposal. Keep the local Hub profile configured even when using an API.

External semantic routing has a total request ceiling of 5,000 estimated tokens
and an output ceiling of 768. External specialist reasoning, direct answers and
synthesis use ceilings of 6,000 and 1,024 respectively. Lower provider, profile or
caller limits still apply. These are application-side estimates; provider
tokenization and CLI overhead may differ. Mandatory instructions and the current
request are never silently truncated to fit. An oversized mandatory request is
rejected before dispatch, and routing permits at most one contract repair.

Broad control routing stays local, independent of model names. Networking stays
on its dedicated local profile unless the signed-in owner explicitly chooses
**Use this connection for networking** in that model's settings. This option works
with HTTP, Pi, OpenCode, installed local profiles, and admitted uploaded models.
An outbound connection receives the networking request and disclosed capability
context; private reference datasets remain excluded. Selecting another local
model does not modify or retrain the dedicated networking model.
Private reference evidence is available only to specialist reasoning, direct
answers and synthesis; it is not injected into routing or networking. A repair
inherits the original stage's connection, retrieval and budget policy.

See [compact routing and verification](compact-routing.md) for the trust
boundary, live evidence and remaining acceptance work.

The HTTP client permits known provider origins and the local endpoints
`http://127.0.0.1:11434` and `http://127.0.0.1:1234`. An administrator can add exact
origins through `EXTENSION_ALLOWED_ORIGINS` (comma separated). Remote URLs require
HTTPS. Embedded URL credentials, redirects, query strings and fragments are
rejected. Endpoints use bearer authentication when a key is supplied.

## Provider usage settings

Personal connections expose **Usage settings**. Set the token/request-per-minute
limits from your provider account and the maximum time a request may wait for
capacity. Empty limits make no assumption about your account tier. Saved settings
can be changed without re-entering the API key; the owner, resource ID, encrypted
credential and selected connection are preserved.

HTTP, Pi and OpenCode connections using the same provider, model and credential
share a process-local 60-second admission window. Requests wait in FIFO order,
oversized requests fail before dispatch, and cancellation removes a waiting item.
Conflicting settings use the lower limit until the bucket expires after inactivity.
Reservations include failures and never refund estimated tokens. Verified HTTP/Pi
usage totals can increase reservations. This does not coordinate other processes,
other applications, or separate keys sharing a provider account. OpenCode accounting
is estimated per invocation; its internal retries cannot currently be disabled
through a verified installed-interface setting. Pi's automatic retries are disabled.

The UI offers reasoning controls from server capability metadata. Supported Groq
GPT-OSS connections via HTTP/Pi can select low, medium or high effort without
raising output limits. Unsupported combinations are rejected. Provider quotas and
truncation remain explicit errors; partial output is not an executable proposal.

The owner-scoped `model_options` command accepts `id`,
`provider_tokens_per_minute`, `provider_requests_per_minute`,
`admission_max_wait_seconds`, `reasoning_effort`, and `networking_inference`.
The last field accepts only `dedicated_local` or `selected_model`; absence defaults
to `dedicated_local`. Null clears an option, including resetting networking to
its dedicated default. Networking mode can be changed for local/uploaded
connections without adding external usage settings or re-entering credentials.
The catalog publishes detached `networking_inference_options` choices for the UI.

This setting changes inference dispatch only. Strict tool-call parsing,
capability schemas, SemanticGuard, provider grounding, approvals, and execution
remain unchanged. A selected model can still fail to produce a valid call; the
runtime reports that failure and never discards invented arguments to force a
successful result. No model weights or existing users' preferences are changed
when deploying this option.

## Datasets and retrieval

Upload UTF-8 text/Markdown for reference retrieval or JSONL for training, up to
2 MB. Alternatively, provide an allowed dataset API URL that returns
`{"text":"..."}` and an optional bearer API key. Supply source and usage permission
metadata and confirm review. The shared sanitizer and additional conservative
secret/identifier checks apply to contents and metadata; manual review remains
necessary.

Reference datasets are owner-private. Search returns chunks with source, digest
and trust metadata. Evidence enters model context separately from user-authored
history, never the semantic authority or permission inputs. Networking inference
does not receive this additional dataset. Retrieval currently uses local FTS;
embedding/vector-provider connectors remain future work.

### Cited answers

A completed answer records `result.references`: one citation for each reference
passage its model calls were actually given, in the order given, at most ten and
without passage text. A citation has exactly `locator`, `dataset_id`, `chunk`,
`sha256`, `content_sha256`, `source` and `license`. Phoenix keeps only well-formed
citations whose locator matches their own dataset, chunk and version, and stores
them with the chat turn so they survive a reload. Each request starts with an
empty list, so a continuation never inherits another request's citations.

`resolve` with payload `{"id", "chunk", "sha256"}` reopens one cited passage for
its owner only, at that exact version. A malformed citation is refused before the
store is read. The reopened passage is still `untrusted_retrieval`: the client
shows it quoted under the answer, never as an instruction.

Training JSONL rows must contain only a `messages` array of role/content pairs,
ending with an assistant target. At least 10 distinct inputs are required.
Conflicting targets are rejected and inputs are split deterministically into
training and validation groups. **Download reviewed training dataset** exports
records with provenance and split metadata for the existing adaptation recipes.
Import does not start training or promote a model. Jobs using a private model or
dataset are excluded from automatic shared trajectory training capture.

### Dataset versions

A changed dataset is uploaded as a new version: choose **Upload as → New version
of …** (payload field `replaces`, the id of the version being replaced). Each
version has its own id and content digest and records `version`, `lineage_id`
(the first version's id), `previous_id` and `previous_sha256`. Earlier versions
are kept unchanged, so a citation made against one still resolves; the catalog
marks a replaced version with `superseded_by`. Only the latest version of a
history can be replaced, the purpose cannot change, an identical upload is
refused, and histories are owner-private.

Training splits are fixed across versions: an input keeps the split an earlier
version gave it, even if a version dropped it and a later one restores it. New
inputs fill validation up to a fifth of the version, so evaluation examples
never become training examples. The export carries the version identity.

## Future workflow connectors

`workflow_validate` accepts a version-1 manifest with name and ordered nodes:

```json
{"version":1,"name":"Read account","nodes":[{"id":"read","capability":"account_status","arguments":{"user_id":"operator"}}]}
```

Validation uses the executable capability catalog, checks arguments/dependencies,
and returns `governed-workflow.v1`. It does **not** execute workflows or register
an n8n webhook. A future runner must supply an authenticated service identity,
pass the existing guard/approval/gateway boundary and reconcile unknown outcomes
without automatically repeating mutations.

## Deployment and API

Phoenix exposes authenticated GET/POST `/api/v1/extensions`. Browser mutations
require CSRF. Phoenix derives the owner from the session and calls the AI's
`/internal/v1/extensions` with the internal token. The synchronous and durable AI
execution endpoints also require that token. Keep both services' tokens aligned.

Commands: `list`, `model`, `select`, `delete`, `probe`, `dataset`, `dataset_url`,
`search`, `resolve`, `export_dataset`, `workflow_validate`, `upload_begin`, `upload_chunk`,
`upload_finish`. Each command has `action` and `payload`; only the trusted internal
envelope contains `owner`.

Configuration:

- `EXTENSION_STORAGE_ROOT`: defaults to `~/.local/share/agentic-extensions` on the
  AI host. Preserve `credentials.key` with the database when backing up.
- `USER_MODEL_ROOT`: defaults to `agenticaiPersonal/models/user-imports`, outside
  `itsm-agent`. Uploads are grouped by hashed owner and upload ID; maximum bundle
  size is 20 GB. Removing a connection does not delete its stored model files.
  Operators should monitor disk usage and remove abandoned uploads when safe.
- `OPENCODE_BINARY`, `PI_BINARY`: explicit installed CLI locations if different
  from `~/.local/share/agentic-harnesses/node_modules/.bin`.

Restart the backend and AI after deploying changes; serve the rebuilt frontend.
Run one AI application process per extension/model storage root. Upload filesystem
work is serialized off the request loop so large-file inspection cannot block job
heartbeats; coordinating shared upload storage across multiple workers is not supported.
Ctrl/Cmd+K opens the command palette; Ctrl/Cmd+, opens connections. The palette
also exposes new chat, search, inspector, refresh and composer focus.

## Free-provider verification

The live unauthenticated OpenCode Zen probe and supported OpenCode CLI probe both
returned HTTP 403. They are not counted as successful external inference tests.
OpenRouter offers free model routes, including `openrouter/free`, but requires an
account/API key and imposes usage limits. No working strong, unlimited, no-key
cloud API is assumed. Local inference remains available without a cloud key.

Official references: [OpenCode providers](https://opencode.ai/docs/providers/),
[OpenCode Zen](https://opencode.ai/docs/zen/),
[OpenRouter free route](https://openrouter.ai/openrouter/free/apps).
