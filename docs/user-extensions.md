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
`{"text":"..."}` and an optional bearer API key, or crawl a documentation site
the machine's profile allows (below). Supply source and usage permission
metadata and confirm review. The shared sanitizer and additional conservative
secret/identifier checks apply to contents and metadata; manual review remains
necessary.

### Crawling a site

`dataset_crawl` reads one site into a reference dataset: payload `url` (the page
to start from), `name`, `license`, `reviewed` and optional `replaces`. Only
origins the machine profile lists under `sources` can be crawled, so what is
reachable is the operator's decision, per machine ([Lua machine
profiles](lua-machine-profiles.md#knowledge-sources)). The crawl stays on that
origin beneath the starting path, honours `robots.txt` (with `Crawl-delay`) and
robots meta tags, follows a redirect only when its target passes the same checks
as a link (at most three hops), reads only HTML, Markdown and plain text, and
stops at 25 pages, depth 3, 1.5 MB of text or 110 seconds, keeping the pages read
so far. Each request crosses the Lua connection boundary.

Pages are reduced to visible text: scripts, styles, navigation, headers, footers
and forms are dropped. The admitted text is one reference dataset in which each
page begins with its title and address, and the dataset records each page's
address and SHA-256 together with skip counts (`robots`, `noindex`, `duplicate`,
`redirect`, `unsupported_content`, `too_large`, `unavailable`, `sensitive`). A
page that admission would refuse (a possible credential or personal identifier,
including in its title) is withheld and counted as `sensitive`, so one page
cannot block the rest of a site; when every page is withheld nothing is saved
(`crawl_all_pages_withheld`). Crawled third-party text is reference knowledge
only and is never admitted for training. A refused crawl returns a `crawl_*`
code and saves nothing. Crawling again with `replaces` stores a new version, so
a changed site never silently rewrites earlier citations.

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

### Lua row transforms

An optional Lua script can reshape your own JSON Lines rows before they are
admitted, for example to turn ticket fields into chat messages. On the Data tab,
open **Transform rows with Lua**, choose the file, edit the script and press
**Preview**. **Apply and upload** is offered only after a preview of the
current script. Command `dataset_transform` takes `script` and `text` (the JSON
Lines rows). Without `apply`, it returns the counts, at most five transformed
rows and the isolation layers used, and admits nothing. With `apply: true` and
the usual dataset fields, the output is admitted exactly like a `dataset`
upload. The same secret/PII scan, license, review and training-row checks apply
to what the script produced. The dataset records `transform` provenance: the
script and input digests and the row counts.

```lua
return function(row)            -- row is one JSON object
  if row.internal then return nil end   -- nil drops the row
  return {messages = {
    {role = "user", content = row.question},
    {role = "assistant", content = row.answer},
  }}
end
```

The script never runs in the AI service or the browser. Each transform starts a
separate worker process (`services/extensions/transform_worker.py`). The worker
inherits no environment beyond `PATH` and the locale, so it has no credentials
or tokens. Before the script runs it locks itself down:

- no new privileges and no child processes;
- no file writes and no new descriptors (no file or socket can be opened);
- CPU-time and address-space limits;
- where the host allows it, a private network namespace and an empty Landlock
  rule set (no file access, no TCP).

The script sees only `assert`, `error`, `ipairs`, `next`, `pairs`, `rawequal`,
`rawlen`, `select`, `tonumber`, `tostring` and `type`, and selected pure
functions of `string`, `table`, `math` and `utf8`. It cannot use `io`, `os`,
`require`, `load`, `debug`, `coroutine`, `pcall`, metatables, `print` or random
numbers. Only text chunks load; precompiled bytecode is refused.

Limits per transform: script 20,000 bytes, input 2,000,000 bytes and 10,000 rows,
output 2,000,000 bytes, 20 million Lua instructions, 64 MiB Lua memory, 10 s CPU
and 20 s wall time.

Only JSON data leaves the worker: tables with text keys, arrays with keys 1..n,
text, finite numbers and booleans, nested at most 16 deep. A refusal crosses
the API as `{code, message}`; the codes are `transform_syntax_error`,
`transform_contract_invalid`, `transform_failed` (with the row number),
`transform_instruction_limit`, `transform_memory_limit`, `transform_time_limit`,
`transform_output_invalid`, `transform_output_empty`,
`transform_output_too_large`, `transform_input_invalid`,
`transform_input_too_large` and `transform_script_*`. The catalog's
`transforms` entry says whether this host can run them; transforms need Linux
and the `lupa` package (MIT, bundles Lua 5.4).

Lua is not a security boundary: the isolation comes from the separate
locked-down process, and admission still decides what is stored. Row
transforms are one use of Lua; every model, connection, link and tool also
passes through the mandatory Lua adapter layer (see
[Lua protocol adapters](lua-adapters.md)).

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
`dataset_crawl`, `dataset_transform`, `search`, `resolve`, `export_dataset`, `workflow_validate`, `upload_begin`,
`upload_chunk`, `upload_finish`. Each command has `action` and `payload`; only the trusted internal
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
