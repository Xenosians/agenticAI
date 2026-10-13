# User models, harnesses and datasets

Local models remain the default. The signed-in account can select another model
or harness in **Models, harnesses and datasets** (Ctrl/Cmd+,). Changes apply to
future requests. The networking specialist continues using its existing model.
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

The HTTP client permits known provider origins and the local endpoints
`http://127.0.0.1:11434` and `http://127.0.0.1:1234`. An administrator can add exact
origins through `EXTENSION_ALLOWED_ORIGINS` (comma separated). Remote URLs require
HTTPS. Embedded URL credentials, redirects, query strings and fragments are
rejected. Endpoints use bearer authentication when a key is supplied.

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

Training JSONL rows must contain only a `messages` array of role/content pairs,
ending with an assistant target. At least 10 distinct inputs are required.
Conflicting targets are rejected and inputs are split deterministically into
training and validation groups. **Download reviewed training dataset** exports
records with provenance and split metadata for the existing adaptation recipes.
Import does not start training or promote a model. Jobs using a private model or
dataset are excluded from automatic shared trajectory training capture.

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
`search`, `export_dataset`, `workflow_validate`, `upload_begin`, `upload_chunk`,
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
