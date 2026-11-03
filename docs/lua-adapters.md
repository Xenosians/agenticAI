# Lua protocol adapters

The application uses Lua as its required protocol adapter layer for local
models, external connections, tools and data sources. Lua describes a mapping;
the host performs the admitted operation. The existing dataset row transform is
a separate use of the same isolated Lua runtime facilities.

## What crosses Lua first

| Boundary | Event (`kind` / `phase`) | Where |
| --- | --- | --- |
| Every model request and its response: outbound HTTP, Pi/OpenCode harnesses, installed and uploaded local models | `model` / `request`, `response`; `warm` before a local load | `InferenceCoordinator` |
| Testing a saved connection (provider or harness probe) | `model` / `request` (harness: also `response`) | `remote_probe`, `probe` action |
| Every local tokenizer, base-weight and adapter load (serving, uploads, training, trainer process) | `model` / `warm` | `lua_adapters.model_loading.from_pretrained` |
| Building a local model backend | `model` / `admission` | `build_model_backend` |
| Every extension command (connections, datasets, uploads, transforms) | `connection` or `source` / `admission` | `extensions.command` |
| Every HTTP request the services send (providers, Jira, Atlassian, OpenWrt, Docker, Phoenix) | `connection` / `request` | `lua_adapters.http_clients` |
| Directory binds (LDAP), mailbox SMTP/IMAP, git network commands, the Phoenix progress socket | `connection` / `request` | `inspect_link`, `job_progress` |
| Every registered MCP tool call and its result | `tool` / `request`, `response` | `ContractAwareMCPServer.tool` |

A refusal at any of these stops the operation before it is sent. Through the
extension API it returns `422 {"code": "lua_adapter_*", "message": ...}` with a
generic message; the script's own error text never leaves the host.

## Choosing the script

Without configuration the packaged `default.lua` runs at every boundary and
keeps each event unchanged. An operator replaces it with one reviewed script
for the whole deployment:

- `ITSM_LUA_ADAPTER_FILE`: absolute path of the script (not a symlink);
- `ITSM_LUA_ADAPTER_SHA256`: its SHA-256. A changed or unreadable file
  refuses every boundary (`lua_adapter_configuration_invalid`) instead of
  falling back to Python.

The trainer process inherits these two settings, so candidate training loads
weights through the same script.

## Cost

Each event runs in its own locked-down worker process, which serves that one
event and exits. Two workers are started ahead of time and wait, with their
imports done and no request data, so an event does not wait for a process to
start. On the development host a hop takes about 11 ms (70 ms when a worker
has to be started on demand, for example during a burst); a waiting worker
uses about 15 MB. A worker locks itself down only after it reads its request,
exactly as before, and a worker that has ended while waiting is never used.
A chat job pays one hop on each model request and response, each tool request
and response, and each outbound call.

Model-provider calls (chat completions, the Anthropic Messages and Models APIs)
keep their HTTPS connections open between calls
(`http_clients.shared_async_client`), so a call does not pay a new TCP and TLS
handshake first (33-52 ms to the providers from the development host). Every
request on a kept connection still crosses the Lua connection adapter. Clients
are kept per event loop and per option set, including the CA bundle, and are
closed when the service stops.

`services.lua_adapters` runs one short Lua script in a separate worker process.
The packaged default script executes Lua and preserves the event payload. A
connection-specific script can reshape a model request, normalize a provider's
response, or translate a source protocol. There is no Python fallback if Lua
fails or is unavailable.

## Host interface

```python
from services.lua_adapters import run_adapter_async

mapped = await run_adapter_async(
    kind="model",
    phase="request",
    binding={"resource_id": admitted_model_id, "stage": trusted_stage},
    payload={"messages": messages},
    source=admitted_source,  # None uses the packaged default.lua.
)
```

Kinds are `model`, `tool`, `connection` and `source`. Phases are `admission`,
`warm`, `request` and `response`. The host supplies a nonempty JSON binding for
the exact owner, resource, stage and request. The script cannot change that
binding or the event's kind, phase or schema. The returned payload remains
untrusted and must pass the host's normal protocol validation.

`AdapterResult.payload` contains the mapped JSON. Its receipt identifies the
runtime version, actual source digest, binding/input/output digests, full adapter
call time (`elapsed_ms`), Lua mapping time (`worker_elapsed_ms`) and the isolation
layers actually applied. This receipt proves an
adapter mapping only; it does not certify provider success, tool execution,
email delivery, training or promotion.

## Lua script contract

Return a function taking one event and returning the same event envelope:

```lua
return function(event)
  if event.kind == "model" and event.phase == "request" then
    local prompts = adapter.array()
    for _, message in ipairs(event.payload.messages) do
      table.insert(prompts, message.content)
    end
    event.payload = adapter.object({prompts = prompts, context = adapter.null})
  end
  return event
end
```

The host must validate that the selected provider accepts this mapped shape.
The example itself does not add a provider or authorize a request.

Existing JSON values retain nulls, empty arrays/objects, booleans and finite
numbers through the default mapping. Use `adapter.array()`, `adapter.object()`
and `adapter.null` when constructing new values whose JSON type would otherwise
be ambiguous in Lua. Untagged nonempty tables use contiguous integer keys for
arrays and string keys for objects. Sparse/mixed tables, functions, nonfinite
numbers, cycles and excessive nesting are rejected.

## Isolation and limits

The worker reuses the dataset transform worker's restricted Lua environment
and operating-system lockdown. It has no Python objects, files, network APIs,
provider credentials, process controls, clock or randomness. It closes every
descriptor beyond its three pipes and cannot acquire another descriptor;
resource limits prohibit new children and file growth. No-new-privileges is
required. Network namespaces and Landlock are used when supported; receipts
do not invent those layers on hosts that refuse them. The required descriptor,
process and resource controls remain in effect independently.

Default limits are 2 million Lua instructions, 32 MiB of Lua heap, 2 CPU seconds,
5 seconds of wall time and 4 MB each for input/output payloads (a whole
2,000,000-byte dataset with its metadata fits). Bindings are
limited to 16 KB, scripts to 20 KB and JSON nesting to 16 levels. The host may
choose stricter limits; the runtime rejects limits above the established hard
ceilings. At most 8 workers serve events at once (`ACTIVE_WORKERS`): a burst
waits for a free worker within each event's own wall-time limit and is refused
with `lua_adapter_busy` if none frees in time; it never starts more processes.

## Authority, secrets and uncertain effects

The host keeps credentials outside Lua. Obvious credential field names are
refused recursively. Omit them before adapter dispatch, using opaque protected
field descriptors if needed, and reattach only the trusted originals at the
transport boundary. Replacing a password's value while leaving a `password`
field in the adapter payload is still refused. `is_secret_field` exposes the
same field-name policy for host redaction. This check cannot discover every
possible secret embedded in free text; the caller remains responsible for
excluding runtime credentials.

Lua cannot grant permissions, select a different owner, bypass stage privacy,
raise request/resource budgets, change an exact tool approval, or turn a
provider failure into success. The host must retain its original identity,
target and outcome checks before dispatch and when processing results.

A request-side mapping error prevents dispatch. A response-side mapping error
after an actual mutation must preserve the original provider receipt and its
known or unknown effect state. It is never permission to resend that operation.
`LuaAdapterError` exposes a stable `lua_adapter_*` code and a generic message;
raw script errors and payload text do not cross that diagnostic boundary.

## Streamed answer previews

While a job is watched, an answer can be shown as the provider writes it
(SRS 1.9 FR-080). The preview skips the `model` / `response` phase, so it is
offered only when that phase cannot change the text: the packaged model codecs
and no `adapt` function in the machine profile
(`services.lua_adapters.profile.model_text_unchanged`). A profile that maps
responses therefore turns previews off; its answers still arrive whole and
pass through Lua. The `model.anthropic` request codec receives the host's
`stream` flag and must return it unchanged. The answer that is saved is
always the complete response after the response phase, never the preview.
