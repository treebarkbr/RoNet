--!strict
-- Weight serialization. Pure string round-trip (no filesystem in the Luau CLI),
-- so checkpoints are portable strings — feed them to real Roblox storage LATER.
-- Deterministic, compact, full precision (tostring(number)).
--   snapshot(params)                       -> {shape, data} states (deep copy)
--   dump(params)                           -> one string for the whole param list
--   load(params, str)                      -> writes parsed values into params in order
--   restore(params, snap)                  -> restores from snapshot
--   parse(str)                             -> {shape, data} list for inspection
-- Format per param:  "shape=<d>,<d>; <n space-separated values>", joined by "\n".
-- Parse is a single forward pass over the string (no per-value table builds).
return function(deps: any): any
	local Serialize = {}

	-- Snapshot current param values into an independent table of {shape, data}.
	function Serialize.snapshot(params: { any }): { any }
		local out: { any } = {}
		for i = 1, #params do
			local d = params[i].data
			out[i] = { shape = table.clone(params[i].shape), data = table.clone(d) }
		end
		return out
	end

	function Serialize.restore(params: { any }, snap: { any })
		for i = 1, #params do
			local target = params[i].data
			local src = snap[i].data
			for j = 1, #target do target[j] = src[j] end
		end
	end

	-- One comma-separated line for a shape + all values.
	local function lineFor(snap: any): string
		local parts: { string } = table.create(2 + #snap.shape + #snap.data)
		table.insert(parts, "shape=")
		if #snap.shape > 0 then
			table.insert(parts, table.concat(snap.shape, ","))
		end
		table.insert(parts, "; ")
		table.insert(parts, table.concat(snap.data, " "))
		return table.concat(parts)
	end

	function Serialize.dump(params: { any }): string
		local lines: { string } = table.create(#params)
		for i = 1, #params do
			lines[i] = lineFor({ shape = params[i].shape, data = params[i].data })
		end
		return table.concat(lines, "\n")
	end

	-- Parse the dump format into a {shape, data} snapshot list. Single pass.
	function Serialize.parse(str: string): { any }
		local out: { any } = {}
		local idx = 1
		local len = #str
		while idx <= len do
			local semi = string.find(str, ";", idx, true)
			if not semi then break end
			local header = string.sub(str, idx, semi - 1)
			-- shape=... 
			local shapeStart = string.find(header, "shape=", 1, true)
			local shape: { number } = {}
			if shapeStart then
				local body = string.sub(header, shapeStart + 6)
				while #body > 0 do
					local comma = string.find(body, ",", 1, true)
					if not comma then
						if #body > 0 then table.insert(shape, assert(tonumber(body))) end
						break
					end
					table.insert(shape, assert(tonumber(string.sub(body, 1, comma - 1))))
					body = string.sub(body, comma + 1)
				end
			end
			-- values until end of line
			local nl = string.find(str, "\n", semi + 1, true)
			local eol = nl or len
			local body = string.sub(str, semi + 2, eol)
			local data: { number } = {}
			local pieces = string.gmatch(body, "[^ ]+")
			for piece in pieces do
				data[#data + 1] = assert(tonumber(piece))
			end
			table.insert(out, { shape = shape, data = data })
			idx = eol + 1
		end
		return out
	end

	-- Write parsed values back into an EXISTING param list (in order). Counts
	-- must match or it errors.
	function Serialize.load(params: { any }, str: string)
		local states = Serialize.parse(str)
		if #states ~= #params then
			error(string.format("Serialize.load: %d states vs %d params", #states, #params))
		end
		for i = 1, #params do
			local target = params[i].data
			local src = states[i].data
			if #target ~= #src then
				error(string.format("Serialize.load: param %d size %d vs state %d", i, #target, #src))
			end
			for j = 1, #target do target[j] = src[j] end
		end
	end

	return Serialize
end