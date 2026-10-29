-- The default adapter preserves the protocol. A connection may supply a
-- reviewed script to reshape payloads, while the host keeps the binding fixed.
return function(event)
  if event.schema ~= "lua-adapter-event.v1" then
    error("unsupported adapter event")
  end
  return event
end
