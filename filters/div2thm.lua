function Div(el)
  local title = el.attributes.title or ""

  if el.classes:includes("assumption") then
    local blocks = {pandoc.RawBlock('latex', '\\begin{assumption}')}
    if title ~= "" then
      table.insert(blocks, pandoc.RawBlock('latex', '\\textbf{' .. title .. '}\\\\')) -- title as bold text
    end
    for _, b in ipairs(el.content) do
      table.insert(blocks, b)  -- keep math blocks intact
    end
    table.insert(blocks, pandoc.RawBlock('latex', '\\end{assumption}'))
    return blocks

  elseif el.classes:includes("problem") then
    local blocks = {pandoc.RawBlock('latex', '\\begin{problem}')}
    if title ~= "" then
      table.insert(blocks, pandoc.RawBlock('latex', '\\textbf{' .. title .. '}\\\\')) -- title as bold text
    end
    for _, b in ipairs(el.content) do
      table.insert(blocks, b)
    end
    table.insert(blocks, pandoc.RawBlock('latex', '\\end{problem}'))
    return blocks

  elseif el.classes:includes("definition") then
    local blocks = {pandoc.RawBlock('latex', '\\begin{bookdef}')}
    if title ~= "" then
      table.insert(blocks, pandoc.RawBlock('latex', '\\textbf{' .. title .. '}\\\\'))
    end
    for _, b in ipairs(el.content) do
      table.insert(blocks, b)
    end
    table.insert(blocks, pandoc.RawBlock('latex', '\\end{bookdef}'))
    return blocks
  end
end
