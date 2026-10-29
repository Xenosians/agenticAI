-- A connection protocol descriptor carries no credentials. Existing targets
-- and method identities remain immutable host bindings, never Lua authority.
return function(event)
  assert(event.kind == "connection", "connection adapter requires a connection event")
  if event.phase == "request" or event.phase == "admission" then
    assert(type(event.payload) == "table", "connection descriptor must be an object")
  end
  return event
end
