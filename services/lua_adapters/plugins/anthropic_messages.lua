-- Anthropic Messages codec. Capability discovery, resource admission, HTTP
-- status and stop-reason checks stay with the host; wire construction lives here.
local function has_text(text)
  -- Match Unicode whitespace rather than treating UTF-8 byte length as content.
  for _, point in utf8.codes(text) do
    local space = (point >= 9 and point <= 13) or (point >= 28 and point <= 32)
      or point == 0x85 or point == 0xa0 or point == 0x1680
      or (point >= 0x2000 and point <= 0x200a) or point == 0x2028 or point == 0x2029
      or point == 0x202f or point == 0x205f or point == 0x3000
    if not space then return true end
  end
  return false
end

return function(event)
  assert(event.kind == "model", "Messages codec requires a model event")
  local input = event.payload
  if event.phase == "response" then
    -- The host projected visible text blocks; hidden reasoning never enters Lua.
    assert(type(input) == "table" and type(input.content) == "table",
      "Messages response requires visible blocks")
    local text = adapter.array()
    for _, block in ipairs(input.content) do
      assert(block.type == "text" and type(block.text) == "string", "unsupported response block")
      table.insert(text, block.text)
    end
    event.payload = table.concat(text)
    return event
  end
  assert(event.phase == "request", "Messages codec requires a request or response")
  assert(type(input) == "table" and type(input.model) == "string"
    and type(input.messages) == "table" and type(input.max_tokens) == "number",
    "Messages codec requires a bounded model request")
  local systems, turns = adapter.array(), adapter.array()
  for _, message in ipairs(input.messages) do
    local role, content = message.role, message.content
    assert(type(content) == "string", "Messages content must be text")
    if role == "system" then
      if has_text(content) then table.insert(systems, content) end
    else
      assert(role == "user" or role == "assistant", "unsupported Messages role")
      if has_text(content) then
        if #turns > 0 and turns[#turns].role == role then
          turns[#turns].content = turns[#turns].content .. "\n\n" .. content
        else
          table.insert(turns, adapter.object({role = role, content = content}))
        end
      end
    end
  end
  if #turns == 0 or turns[1].role ~= "user" then
    table.insert(turns, 1, adapter.object({role = "user",
      content = "Continue with the task described in the system instructions."}))
  end
  assert(turns[#turns].role == "user", "newest Messages turn must be user")
  local tokens = input.max_tokens
  if input.max_tokens_limit ~= nil and input.max_tokens_limit ~= adapter.null then
    tokens = math.min(tokens, input.max_tokens_limit)
  end
  local body = adapter.object({model = input.model, max_tokens = tokens, messages = turns, stream = false})
  local system = table.concat(systems, "\n\n")
  if has_text(system) then
    if utf8.len(system) >= 4096 then
      body.system = adapter.array({adapter.object({type = "text", text = system,
        cache_control = adapter.object({type = "ephemeral"})})})
    else
      body.system = system
    end
  end
  if input.thinking ~= nil and input.thinking ~= adapter.null then
    body.thinking = adapter.object({type = input.thinking})
  end
  if input.effort ~= nil and input.effort ~= adapter.null then
    body.output_config = adapter.object({effort = input.effort})
  end
  event.payload = body
  return event
end
