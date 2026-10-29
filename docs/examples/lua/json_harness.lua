-- Example replacement for model.harness. The complete conversation remains
-- present; the host still checks size/budget and disables harness-native tools.
return function(event)
  assert(event.kind == "model" and event.phase == "request", "expected a model request")
  event.payload = "Conversation JSON follows. Produce only the assistant answer; "
    .. "tool proposals are text for the ITSM gateway, never executable actions.\n"
    .. event.payload.messages_json
  return event
end
