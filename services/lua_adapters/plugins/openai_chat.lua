-- OpenAI-compatible chat request codec. Provider-native tools stay absent:
-- all ITSM effects continue through the host's registered capabilities.
return function(event)
  assert(event.kind == "model", "chat codec requires a model event")
  if event.phase == "response" then
    assert(type(event.payload) == "table" and type(event.payload.content) == "string",
      "chat response requires visible text")
    event.payload = event.payload.content
    return event
  end
  assert(event.phase == "request", "chat codec requires a request or response")
  local input = event.payload
  assert(type(input) == "table" and type(input.model) == "string"
    and type(input.messages) == "table" and type(input.max_tokens) == "number",
    "chat codec requires a bounded model request")
  local body = adapter.object({
    model = input.model,
    messages = input.messages,
    max_tokens = input.max_tokens,
    stream = false
  })
  local options = input.options
  if options ~= nil and options ~= adapter.null then
    if options.reasoning_effort ~= nil and options.reasoning_effort ~= adapter.null then
      body.reasoning_effort = options.reasoning_effort
    end
    if options.response_format ~= nil and options.response_format ~= adapter.null then
      body.response_format = adapter.object({type = options.response_format})
    end
  end
  event.payload = body
  return event
end
