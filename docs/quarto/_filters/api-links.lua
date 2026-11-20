-- Link documentation prose against the current quartodoc inventory.
local roles = {class = true, func = true, meth = true, obj = true, exc = true, attr = true, data = true}

local function reference(text)
  local label, target = text:match("^(.-)%s*<([^>]+)>$")
  target = target or text:gsub("^~", "")
  return label or (text:sub(1, 1) == "~" and target:match("[^.]+$")) or text, target
end

-- True when the page file name names the anchored object, e.g. api/Combiner.html#pkg.Combiner.
local function dedicated(uri)
  local stem, fragment = uri:match("([^/]+)%.html#(.+)$")
  return stem ~= nil and (fragment == stem or fragment:sub(-#stem - 1) == "." .. stem)
end

local function index(items, project)
  local names, short, explicit = {}, {}, {}
  local function add(table, key, uri)
    local previous = table[key]
    if previous == nil then
      table[key] = uri
    elseif previous and previous ~= uri then
      local anchor = uri:match("#(.+)$")
      if anchor and anchor == previous:match("#(.+)$") then
        -- One object rendered on two pages: prefer its own page.
        if dedicated(uri) and not dedicated(previous) then table[key] = uri end
      else
        table[key] = false -- Ambiguous, even if a third entry matches one of these.
      end
    end
  end
  for _, item in ipairs(items) do
    local callable = item.role == "class" or item.role == "exception"
      or item.role == "function" or item.role == "method"
    if item.domain == "py" and (callable or item.role == "attribute") then
      local uri = item.uri:gsub("%$$", item.name)
      -- quartodoc lists attribute anchors that its pages do not render: link the page.
      if not callable then uri = uri:gsub("#.*$", "") end
      add(names, item.name, uri)
      local relative = item.name:sub(#project + 2)
      if item.name:sub(1, #project + 1) == project .. "." then
        if relative:find(".", 1, true) then
          add(names, relative, uri)
          local member = relative:match("^([A-Z][%w_]*%.[%w_]+)$")
            or relative:match("%.([A-Z][%w_]*%.[%w_]+)$")
          if member then add(names, member, uri) end
        end
        local leaf = relative:match("[^.]+$")
        add(explicit, leaf, uri)
        if callable then add(short, leaf, uri) end
      end
    end
  end
  return names, short, explicit
end

-- Other packages: the Sphinx inventories that `quartodoc interlinks` saves as
-- _inv/<source>_objects.json for each `interlinks.sources` entry. A dotted name links
-- on an exact full-name match in the source named by its first part, else in the
-- `python` source (standard library). A bare name links only to a Python exception,
-- so words such as `array` or `min` never link. Without the config or the files,
-- nothing links. Each inventory loads on first use.
local function external(meta, offset)
  local sources = meta.interlinks and meta.interlinks.sources
  local loaded = {}
  local function load(source)
    if loaded[source] ~= nil then return loaded[source] end
    loaded[source] = false
    local spec = sources[source]
    local file = spec and spec.url and io.open(offset .. "/_inv/" .. source .. "_objects.json", "r")
    if not file then return false end
    local url = pandoc.utils.stringify(spec.url)
    local inventory = quarto.json.decode(file:read("*a"))
    file:close()
    local entries = {}
    for _, item in ipairs(inventory.items) do
      if item.domain == "py" and entries[item.name] == nil then
        entries[item.name] = {url = url .. (item.uri:gsub("%$$", item.name)), role = item.role}
      end
    end
    loaded[source] = entries
    return entries
  end
  return function(name)
    if not sources then return nil end
    local entry
    if name:find(".", 1, true) then
      local top = name:match("^[^.]+")
      local entries = sources[top] and load(top)
      entry = entries and entries[name]
      if not entry and top ~= "python" and sources.python then
        entries = load("python")
        entry = entries and entries[name]
      end
    else
      local entries = sources.python and load("python")
      entry = entries and entries[name]
      if entry and entry.role ~= "exception" then entry = nil end
    end
    return entry and entry.url
  end
end

local function Pandoc(doc)
  local settings = doc.meta["api-links"] or {}
  local offset = quarto.project.offset
  local file = assert(io.open(offset .. "/objects.json", "r"),
    "Missing quartodoc inventory: generate the API reference before rendering")
  local inventory = quarto.json.decode(file:read("*a"))
  file:close()
  local names, short, explicit_short = index(inventory.items, inventory.project)
  local outside = external(doc.meta, offset)
  -- Site pages are relative to the project root; other packages' pages are absolute.
  local function href(uri)
    return uri:match("^%a[%w+.-]*:") and uri or offset .. "/" .. uri
  end
  local page = pandoc.path.make_relative(quarto.doc.input_file, quarto.project.directory)
  page = page:gsub("%.[^./]+$", ".html")
  -- On an API page a parameter name hides a package object of the same name.
  local parameters = {}
  local ids = {}
  local function collect(element)
    if element.identifier and element.identifier ~= "" then ids[element.identifier] = true end
  end
  doc:walk({
    Span = function(span)
      collect(span)
      if span.classes:includes("parameter-name") then
        parameters[pandoc.utils.stringify(span)] = true
      end
    end,
    Header = collect, Div = collect, Code = collect, Link = collect, Image = collect,
    CodeBlock = collect, Figure = collect, Table = collect,
  })

  local function resolve(text, explicit)
    local name = text:match("^([%a_][%w_.]*)%b()$") or text:match("^([%a_][%w_.]*)$")
    if not name then return nil end
    local uri
    local prefix, rest = name:match("^([^.]+)%.(.+)$")
    if prefix then
      local alias = settings.aliases and settings.aliases[prefix]
      if alias then name = pandoc.utils.stringify(alias) .. "." .. rest end
      local reexport = settings.reexports and settings.reexports[name]
      if reexport then name = pandoc.utils.stringify(reexport) end
      uri = names[name]
    elseif explicit then
      uri = explicit_short[name]
    -- Short common words can describe an algorithm, option, or parameter.
    -- Require a call, a conventional class/underscore name, or an explicit reference.
    elseif (text ~= name or name:match("^[A-Z]") or name:find("_", 1, true))
        and not (text == name and parameters[name]) then
      uri = short[name]
    end
    if uri == nil then uri = outside(name) end
    -- A page does not link to itself or its own object; links to its member anchors stay.
    if uri and uri:match("^[^#]*") == page and (not uri:find("#", 1, true) or dedicated(uri)) then
      return nil, true
    end
    return uri
  end

  return doc:walk({
    traverse = "topdown",
    Link = function(link)
      -- Griffe's NumPy See Also section can emit an empty Markdown link with
      -- a Sphinx role as the target. Resolve it as a reference, not as a URL.
      local role, text = link.target:gsub("%%60", "`"):match("^:([%a:]+):`(.-)`$")
      if role and roles[role:gsub("^py:", "")] then
        local label, target = reference(text)
        local uri = resolve(target, true)
        if not uri then
          return #link.content > 0 and link.content or pandoc.Code(label), false
        end
        if #link.content == 0 then link.content = {pandoc.Code(label)} end
        link.target = href(uri)
      end
      -- quartodoc summary tables link attributes to sections it does not render.
      -- Make such a row the missing anchor so the link and inventory entry land on it.
      local anchor = link.target:match("^#(.+)$")
      if anchor and not ids[anchor] then
        ids[anchor] = true
        return pandoc.Span(link.content, pandoc.Attr(anchor)), false
      end
      return link, false
    end,
    Image = function(img) return img, false end,
    Header = function(header) return header, false end,
    Inlines = function(inlines)
      local out = pandoc.List()
      for _, el in ipairs(inlines) do
        if el.t == "Code" and not el.classes:includes("no-api-link") then
          local previous = out[#out]
          local role
          if previous and previous.t == "Str" then
            role = previous.text:match(":py:(%a+):$") or previous.text:match(":(%a+):$")
          end
          local target = el.attributes["api-target"]
          local label = el.text
          if roles[role] then
            label, target = reference(label)
            -- Remove the role marker whether or not the target resolves.
            previous.text = previous.text:gsub(":py:" .. role .. ":$", ""):gsub(":" .. role .. ":$", "")
            if previous.text == "" then out:remove(#out) end
          end
          local uri, own_page = resolve(target or el.text, target ~= nil)
          if el.attributes["api-target"] and not uri and not own_page then
            error("Unknown or ambiguous API target: " .. target)
          end
          el.attributes["api-target"] = nil
          el.text = label
          if uri then el = pandoc.Link(el, href(uri)) end
        end
        out:insert(el)
      end
      return out
    end,
  })
end

return {{ Pandoc = Pandoc }}
