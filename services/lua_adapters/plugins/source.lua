-- Dataset and retrieval data remain reference content. This plugin never
-- creates training admission, tool permissions or provider success receipts.
return function(event)
  assert(event.kind == "source", "source adapter requires a source event")
  return event
end
