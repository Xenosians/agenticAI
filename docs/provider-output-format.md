# Provider output formatting

Verified Groq HTTP connections using `openai/gpt-oss-20b` or
`openai/gpt-oss-120b` request `response_format: {"type":"json_object"}` for
semantic Hub routing and an explicitly identified repair of that stage. The
trusted inference stage selects the format. Model and endpoint metadata only
establish compatibility; a saved connection cannot choose an execution stage
or inject a response-format option.

The routing contract contains sparse argument maps and an optional conditional
`when` field. Groq's strict schema mode requires closed objects and every
property to be required. Applying a fixed strict schema to this contract would
force absent arguments/conditions into the response or reject valid maps.
JSON-object syntax mode preserves the existing protocol. It does not prove
correct tool selection, grounding, schema compliance, or permission.

Worker proposals remain arrays, while primary and synthesis responses retain
their existing text protocol. Local models, Pi, OpenCode, unverified models,
and other endpoints receive no inferred JSON-object option. Existing token
limits, quota admission, truncation rejection, one routing repair, and all
application validation remain unchanged. Responses are never patched to add
missing braces or silently retried without the format.

Primary provider references, checked 2026-10-16:
[Groq structured outputs and JSON-object mode](https://console.groq.com/docs/structured-outputs),
[Groq Chat API response format](https://console.groq.com/docs/api-reference#chat-create).
