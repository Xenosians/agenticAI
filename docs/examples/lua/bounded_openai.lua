-- Example replacement for model.openai requests. This adapter may lower the
-- admitted output cap; it cannot raise it or introduce provider-native tools.
return function(event)
  assert(event.kind == "model", "expected a model event")
  if event.phase == "response" then
    event.payload = event.payload.content
    return event
  end
  local input = event.payload
  local body = adapter.object({model = input.model, messages = input.messages,
    max_tokens = math.min(input.max_tokens, 128), stream = false})
  local options = input.options
  if options.reasoning_effort ~= nil and options.reasoning_effort ~= adapter.null then
    body.reasoning_effort = options.reasoning_effort
  end
  if options.response_format ~= nil and options.response_format ~= adapter.null then
    body.response_format = adapter.object({type = options.response_format})
  end
  event.payload = body
  return event
end
