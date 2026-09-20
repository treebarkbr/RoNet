--!strict
-- Byte-level BPE tokenizer (GPT-style sub-word units), pure Luau (no deps).
--   BPE.train(text: string | {string}, numMerges) -> tokenizer
--   tok:encode(text)  -> {number}   (token ids, ids 0..255 are raw bytes)
--   tok:decode(ids)   -> string
--   tok:vocabSize()   -> number     (256 + numMerges)
--   tok:merges()      -> { {number, number} } ordered (a, b) build pairs
--   tok:piece(id)     -> string     (byte string for a token id)
-- Deterministic: identical corpus + numMerges yields identical merges (ties are
-- broken by the lexicographically smallest pair, then earliest appearance).
-- Performance: byte sequences live as arrays of numbers; pair counting uses a
-- single composite-integer key (a * BASE + b) to avoid string allocation.
local BPE = {}
BPE.__index = BPE

local BASE = 5000000

-- Split a single string into an array of byte values (0..255).
local function bytesOf(s: string): { number }
	local n = #s
	local out = table.create(n)
	for i = 1, n do out[i] = string.byte(s, i) end
	return out
end

local function toBytes(s: any): { number }
	if type(s) == "string" then return bytesOf(s) end
	return s
end

-- Concatenate two token pieces (raw byte arrays) into one.
local function pieceConcat(p1: { number }, p2: { number }): { number }
	local out: { number } = table.create(#p1 + #p2)
	local n = #p1
	for i = 1, n do out[i] = p1[i] end
	for i = 1, #p2 do out[n + i] = p2[i] end
	return out
end

-- Replace all occurrences of the byte-pair (a, b) in every sequence with newId.
local function applyMerge(seqs: { { number } }, a: number, b: number, newId: number)
	for si = 1, #seqs do
		local s = seqs[si]
		local n = #s
		local out = table.create(n) -- upper bound; will trim with a pointer
		local m = 0
		local i = 1
		while i <= n do
			if i < n and s[i] == a and s[i + 1] == b then
				m += 1
				out[m] = newId
				i += 2
			else
				m += 1
				out[m] = s[i]
				i += 1
			end
		end
		local trimmed: { number } = table.create(m)
		for j = 1, m do trimmed[j] = out[j] end
		seqs[si] = trimmed
	end
end

-- Train: greedily merge the most frequent adjacent pair numMerges times.
-- Returns the tokenizer instance (chainable).
function BPE.train(corpus: any, numMerges: number): any
	local docs: { { number } } = {}
	if type(corpus) == "string" then
		table.insert(docs, bytesOf(corpus))
	else
		for i = 1, #corpus do table.insert(docs, toBytes(corpus[i])) end
	end
	local pieces: { { number } } = {}
	for b = 0, 255 do pieces[b + 1] = { b } end -- ids 1..256 map to byte b = id-1

	local counts: { [number]: number } = {}
	local firstSeen: { [number]: number } = {}
	local merges: { any } = {}

	for it = 1, numMerges do
		table.clear(counts)
		table.clear(firstSeen)
		for si = 1, #docs do
			local s = docs[si]
			local n = #s
			for i = 1, n - 1 do
				local key = s[i] * BASE + s[i + 1]
				local c = counts[key]
				if c == nil then
					counts[key] = 1
					firstSeen[key] = i * 1000 + si
				else
					counts[key] = c + 1
				end
			end
		end
		-- find the globally most frequent pair (count, then earliest first-seen)
		local bestKey = 0
		local bestCount = 0
		local bestFirst = math.huge
		for k, c in pairs(counts) do
			if c > bestCount or (c == bestCount and firstSeen[k] < bestFirst) then
				bestCount = c
				bestKey = k
				bestFirst = firstSeen[k]
			end
		end
		if bestKey == 0 or bestCount < 2 then break end
local a = math.floor(bestKey / BASE)
			local b = bestKey - a * BASE
			local newId = 256 + it - 1
			table.insert(merges, { a, b })
			pieces[newId + 1] = pieceConcat(pieces[a + 1], pieces[b + 1])
			applyMerge(docs, a, b, newId)
	end

	local self: any = setmetatable({
		mergeRules = merges,
		pieces = pieces,
		vocab = 256 + #merges,
	}, BPE)
	return self
end

function BPE.new(): any
	local self: any = setmetatable({
		mergeRules = {},
		pieces = {},
		vocab = 256,
	}, BPE)
	for b = 0, 255 do self.pieces[b + 1] = { b } end
	return self
end

-- Add a single merge rule (a, b) -> newId (manual extension; ids must be new).
function BPE.addMerge(self, a: number, b: number): number
	local newId = self.vocab
	table.insert(self.mergeRules, { a, b })
	self.pieces[newId + 1] = pieceConcat(self.pieces[a + 1], self.pieces[b + 1])
	self.vocab += 1
	return newId
end

function BPE.vocabSize(self): number
	return self.vocab
end

function BPE.merges(self): { any }
	return self.mergeRules
end

function BPE.piece(self, id: number): string
	local p = self.pieces[id + 1]
	if not p then error("BPE.piece: unknown id " .. tostring(id)) end
	local chars: { string } = table.create(#p)
	for i = 1, #p do chars[i] = string.char(p[i]) end
	return table.concat(chars)
end

-- Encode: bytes -> ids, then repeatedly apply merges in training order, each to
-- exhaustion. Matches the training path, so trained text round-trips exactly.
function BPE.encode(self, text: string): { number }
	local toks = bytesOf(text)
	local ms = self.mergeRules
	for r = 1, #ms do
		local m = ms[r]
		local a = m[1]
		local b = m[2]
		while true do
			local n = #toks
			local found = 0
			for i = 1, n - 1 do
				if toks[i] == a and toks[i + 1] == b then
					found = i
					break
				end
			end
			if found == 0 then break end
			local merged: { number } = table.create(n - 1)
			local m2 = 0
			for i = 1, n do
				if i == found + 1 then
					continue
				end
				m2 += 1
				if i == found then
					merged[m2] = 256 + r - 1
				else
					merged[m2] = toks[i]
				end
			end
			toks = merged
		end
	end
	return toks
end

function BPE.decode(self, ids: { number }): string
	local chars: { string } = table.create(#ids)
	for i = 1, #ids do chars[i] = self:piece(ids[i]) end
	return table.concat(chars)
end

-- Serializable tokenizer state: { vocab, merges }. Feeds BPE.fromState.
function BPE.state(self): any
	local ms: { any } = {}
	for r = 1, #self.mergeRules do ms[r] = { self.mergeRules[r][1], self.mergeRules[r][2] } end
	return { vocab = self.vocab, merges = ms }
end

-- Rebuild a tokenizer from a state produced by tok:state().
function BPE.fromState(st: any): any
	local self: any = setmetatable({
		mergeRules = {},
		pieces = {},
		vocab = 256,
	}, BPE)
	for b = 0, 255 do self.pieces[b + 1] = { b } end
	for r = 1, #st.merges do
		local a = st.merges[r][1]
		local b = st.merges[r][2]
		local newId = self.vocab
		table.insert(self.mergeRules, { a, b })
		self.pieces[newId + 1] = pieceConcat(self.pieces[a + 1], self.pieces[b + 1])
		self.vocab += 1
	end
	if st.vocab and st.vocab ~= self.vocab then
		error("BPE.fromState: vocab mismatch, state is corrupt")
	end
	return self
end

return BPE