-- these classes, when placed on a span will be replaced
-- with an identical LaTeX command for PDF output
local texMappings = {
  "proglang",
  "pkg",
  "fct",
  "class"
}

-- layout and style
local kTypes = pandoc.List({'article', 'codesnippet', 'bookreview', 'softwarereview'})
local kSuppress = pandoc.List({'title', 'headings', 'footer'})

local function setClassOption(meta, option)
  if meta['classoption'] == nil then
    meta['classoption'] = pandoc.List({})
  end
  
  meta['classoption']:insert({ pandoc.Str(option)})
end

local function printList(list) 
  local result = ''
  local sep = ''
  for i,v in ipairs(list) do
    result = result .. sep .. v
    sep = ', '
  end
  return result
end

return {
  {
    Span = function(el) 
      -- read the span contents and emit correct output
      local contentStr = pandoc.utils.stringify(el.content)
      
      for i, mapping in ipairs(texMappings) do
        if #el.attr.classes == 1 and el.attr.classes:includes(mapping) then
          if quarto.doc.is_format("pdf") then
            return pandoc.RawInline("tex", "\\" .. mapping .. "{" .. contentStr .. "}" )
          else 
            return pandoc.Code(contentStr);
          end
        end
      end
    end,  
    Meta = function(meta)            
      if quarto.doc.is_format("pdf") then
        -- Authors output in the template uses a special separator
        -- to join authors (including wrapping to a new line)
        -- this computes the proper prefix and places it in the author metadata
        -- for use by the template
        local byAuthor = meta['by-author']
        if byAuthor ~= nil then
          for i, author in ipairs(byAuthor) do
            local prefix = {pandoc.RawInline("tex ","")};
            if i > 1 and i % 2 == 1 then
              prefix = {pandoc.RawInline("tex", "\\AND")}
            elseif i > 1 then
              prefix = {pandoc.RawInline("tex", "\\And")}
            end
            author['metadata']['latex-prefix'] = prefix
          end
        end
        
        -- read the journal settings
        local journal = meta['journal']
        local type = nil
        local shortnames = nil
        local suppress = nil
        local jss = nil
        
        if journal ~= nil then         
          type = journal['type']
          shortnames = journal['cite-shortnames']
          suppress = journal['suppress']
          jss = journal['include-jss-layout']
        end
        
        -- process the type
        if type ~= nil then
          type = pandoc.utils.stringify(type)
          if kTypes:includes(type) then
            setClassOption(meta, type)
          else
            error("Unknown type " .. type .. "\nPlease use one of " .. printList(kTypes))
          end
        else
          setClassOption(meta, 'article')
        end
        
        -- process the citation variant
        if shortnames == true then
          setClassOption(meta, 'shortnames')
        end
        
        -- process the suppressed parts
        if suppress ~= nil then
          for i,v in ipairs(suppress) do
            s = pandoc.utils.stringify(v)
            if kSuppress:includes(s) then
              setClassOption(meta, 'no' .. s)
            else
              error("Unknown suppressed part " .. s .. "\nPlease use one of " .. printList(kSuppress))
            end
          end
        end
        
        -- process switch for JSS layout
        if jss == false then
          setClassOption(meta, 'nojss')
        end

        -- make sure there is a plain title
        if meta['title'] ~= nil and meta['title-plain'] == nil then
          meta['title-plain'] = pandoc.utils.stringify(meta['title'])
        end
      end
      
      return meta
    end
  }
}