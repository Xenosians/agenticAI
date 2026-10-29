-- Pi/OpenCode consume an inference-only prompt. JSON serialization is host I/O;
-- the harness framing and instruction protocol are defined in this Lua file.
return function(event)
  assert(event.kind == "model" and event.phase == "request", "harness codec requires a model request")
  assert(type(event.payload) == "table" and type(event.payload.messages_json) == "string",
    "harness codec requires a serialized conversation")
  event.payload = "Follow this conversation as an inference engine. "
    .. "Return only the assistant response. "
    .. "Do not execute tools. Any proposed tool call is "
    .. "text for the ITSM gateway to validate.\n"
    .. event.payload.messages_json
  return event
end
