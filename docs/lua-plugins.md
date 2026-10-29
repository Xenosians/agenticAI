# File-based Lua plugins

Lua files own the portable protocol behavior. Python loads reviewed artifacts,
validates identities and resource bounds, and performs the admitted native
operation. A plugin mapping receipt does not certify that a provider completed
work. The [adapter runtime](lua-adapters.md) describes process isolation and the
host's authority boundaries.

## Packaged protocol files

| Plugin ID | Executed source | Behavior |
| --- | --- | --- |
| `model.local` | `services/lua_adapters/plugins/local_model.lua` | Canonical local request/response shapes and pinned loader descriptors. |
| `model.openai` | `services/lua_adapters/plugins/openai_chat.lua` | OpenAI-compatible chat body, selected format/effort options and visible response text. |
| `model.anthropic` | `services/lua_adapters/plugins/anthropic_messages.lua` | System extraction, adjacent-turn merging, Unicode-aware cache framing, output/thinking/effort fields and visible text-block concatenation. |
| `model.harness` | `services/lua_adapters/plugins/harness_prompt.lua` | Inference-only Pi/OpenCode prompt framing around the canonical conversation JSON. |
| `connection.default` | `services/lua_adapters/plugins/connection.lua` | Credential-free connection descriptors with unchanged admitted targets. |
| `source.default` | `services/lua_adapters/plugins/source.lua` | Data-source/reference protocol; no training or tool authority. |
| `tool.default` | `services/lua_adapters/plugins/tool.lua` | Registered argument/result protocol preserving actual denied, failed and unknown outcomes. |

No packaged file contains a selected account, endpoint, local-model path or
provider key. Machine link and model selections belong in the reviewed machine
profile; credentials remain in its separate protected secret store.

## Select another reviewed file

A machine profile can select a file for one protocol without rebuilding Python:

```lua
return {
  plugins = {
    ["model.harness"] = {
      file = "plugins/my_harness.lua",
      sha256 = "REPLACE_WITH_THE_FILE_SHA256"
    }
  }
}
```

Replace the digest with the actual lowercase SHA256 of the exact UTF-8 file
bytes; the placeholder is deliberately rejected. Relative files resolve beneath
the machine profile's directory. Unknown plugin IDs, extra artifact fields,
directory escapes, symbolic links, missing files, oversized scripts and changed
digests are refused. An explicitly selected file never falls back to a packaged
file when invalid. The loader rechecks its source bytes at dispatch.

Executable examples live beside this guide:

- [JSON harness framing](examples/lua/json_harness.lua).
- [A lower OpenAI output cap](examples/lua/bounded_openai.lua).

Copy an example into the deployment's reviewed plugin directory, review changes,
compute its digest and select it in the machine profile. Every script runs in the
existing isolated Lua worker; arbitrary `require`, files, network, processes and
credentials are unavailable inside the interpreter.

## Host interfaces

`services.lua_adapters.plugins.run_plugin` and its async counterpart return the
mapped payload plus an actual source/input/output/isolation receipt. The host
binding pins the plugin ID and selected source digest. The higher-level helpers
are `openai_request`, `anthropic_request`, `harness_prompt`, `openai_response` and
`anthropic_response`, each with an `_async` version.

The OpenAI request guard preserves the admitted model and conversation, caps
output, keeps streaming disabled and excludes native tools/functions. The
Messages guard verifies ordered role/content fidelity, exact system text,
output limits and selected thinking/effort options; Lua constructs the wire
body. Harness JSON serialization is host I/O, while prompt framing is Lua.
Custom harness framing must retain the original conversation and stay within
the actual admitted request budget and byte limit before dispatch.

Response helpers receive only visible text projections. Hidden reasoning is
removed by the host before Lua sees the response. Empty or malformed visible
data cannot be turned into a successful completion by a custom script. HTTP
status, provider finish/stop reason, unexpected native-tool blocks, truncation,
usage accounting and actual raw receipt provenance remain separate host checks.
After a real mutation, a mapping failure cannot erase its provider receipt or
authorize resending it.

The machine profile's general `adapt` hook is distinct from a named protocol
codec. It does not replace `model.openai` merely by returning the original
event. The host applies the reviewed profile hook and selected codec at their
appropriate boundaries, then validates both outputs before I/O.
