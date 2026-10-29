-- Canonical protocol presented to a native local-model host. No model is loaded
-- here: the host validates the pinned model and performs native inference.
return function(event)
  assert(event.kind == "model", "local model adapter requires a model event")
  local payload = event.payload
  if event.phase == "warm" and event.binding.loader ~= nil then
    assert(type(payload) == "table" and type(payload.artifacts) == "table"
      and type(payload.options) == "table", "model loader requires pinned artifact descriptors")
    return event
  end
  if event.phase == "request" or event.phase == "warm" then
    assert(type(payload) == "table" and type(payload.messages) == "table",
      "local model request requires messages")
    assert(type(payload.max_new_tokens) == "number" and payload.max_new_tokens >= 0,
      "local model request requires an output bound")
    local messages = adapter.array()
    for _, message in ipairs(payload.messages) do
      assert(type(message) == "table" and type(message.role) == "string"
        and type(message.content) == "string", "local model messages must be text")
      local copy = adapter.object()
      for key, value in pairs(message) do copy[key] = value end
      table.insert(messages, copy)
    end
    event.payload = adapter.object({messages = messages, max_new_tokens = payload.max_new_tokens})
  elseif event.phase == "response" then
    assert(type(payload) == "table" and type(payload.text) == "string",
      "local model response requires text")
    event.payload = adapter.object({text = payload.text})
  end
  return event
end
