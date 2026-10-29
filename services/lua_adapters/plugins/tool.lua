-- Preserve the actual registered tool protocol. In particular, false, denied
-- and unknown provider outcomes are data and must not be turned into success.
return function(event)
  assert(event.kind == "tool", "tool adapter requires a tool event")
  if event.phase == "request" then
    assert(type(event.payload) == "table", "tool arguments must be an object")
  end
  return event
end
