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
-- Memory: token pieces are stored packed as byte strings (one allocation per
-- token, not one table per byte), and merge rules are a single flat array
-- {a1, b1, a2, b2, ...} instead of one {a, b} table per rule. Keeps the
-- per-vocab-entry table count to a bare minimum.
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

-- Concatenate two token pieces (byte strings) into one.
local function pieceConcat(p1: string, p2: string): string
	return p1 .. p2
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
	local pieces: { string } = {}
	for b = 0, 255 do pieces[b + 1] = string.char(b) end -- ids 1..256 map to byte b = id-1

	local counts: { [number]: number } = {}
	local firstSeen: { [number]: number } = {}
	local ms: { number } = {} -- flat {a1, b1, a2, b2, ...} merge rules

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
		table.insert(ms, a)
		table.insert(ms, b)
		pieces[newId + 1] = pieceConcat(pieces[a + 1], pieces[b + 1])
		applyMerge(docs, a, b, newId)
	end

	local self: any = setmetatable({
		ms = ms,
		pieces = pieces,
		vocab = 256 + #ms // 2,
	}, BPE)
	return self
end

-- An empty byte-level tokenizer (vocab 256, no merges). Useful for building a
-- tokenizer by hand with addMerge.
function BPE.new(): any
	local self: any = setmetatable({
		ms = {},
		pieces = {},
		vocab = 256,
	}, BPE)
	for b = 0, 255 do self.pieces[b + 1] = string.char(b) end
	return self
end

-- Add a single merge rule (a, b) -> newId (manual extension; ids must be new).
function BPE.addMerge(self, a: number, b: number): number
	local newId = self.vocab
	table.insert(self.ms, a)
	table.insert(self.ms, b)
	self.pieces[newId + 1] = pieceConcat(self.pieces[a + 1], self.pieces[b + 1])
	self.vocab += 1
	return newId
end

-- Total vocabulary size (256 bytes + number of merge rules).
function BPE.vocabSize(self): number
	return self.vocab
end

-- Ordered list of merge rules {a, b} as applied during training.
function BPE.merges(self): { any }
	local ms = self.ms
	local out: { any } = table.create(#ms // 2)
	for r = 1, #ms, 2 do out[(r + 1) // 2] = { ms[r], ms[r + 1] } end
	return out
end

-- The byte string a token id decodes to.
function BPE.piece(self, id: number): string
	local p = self.pieces[id + 1]
	if not p then error("BPE.piece: unknown id " .. tostring(id)) end
	return p
end

-- Encode: bytes -> ids, then apply merge rules in training order, each to
-- exhaustion, left-to-right. Matches the training path exactly.
-- Linear-time structure: tokens live in a doubly-linked list; a per-rule
-- candidate queue tracks only the positions where a rule could fire, and a
-- merge creates at most two new candidate positions (its new neighbours), so
-- a full scan per rule is never needed.
function BPE.encode(self, text: string): { number }
	if #text == 0 then return {} end
	local ms = self.ms
	local R = #ms // 2
	local TBASE = 256 + R
	-- (a, b) -> 1-based rule index. Rules emit unique ids so each pair maps to
	-- at most one rule; the composite key keeps lookups allocation-free.
	local ruleOf: { [number]: number } = {}
	for r = 1, R do
		ruleOf[ms[2 * r - 1] * TBASE + ms[2 * r]] = r
	end

	local n = #text
	local t: { number } = table.create(n) -- token ids; -1 marks a dead slot
	local nxt: { number } = table.create(n)
	local prv: { number } = table.create(n)
	for i = 1, n do
		t[i] = string.byte(text, i)
		nxt[i] = i < n and i + 1 or 0
		prv[i] = i > 1 and i - 1 or 0
	end

	-- Candidate queues: heads[r] = link-list head for rule r; each element is a
	-- (position, nextHead) event pair so a slot may be queued for several rules.
	local heads: { [number]: number } = {}
	local chain: { number } = {}
	local chainNext: { number } = {}
	local ev = 0
	local function push(p: number, r: number)
		ev += 1
		chain[ev] = p
		chainNext[ev] = heads[r]
		heads[r] = ev
	end

	for i = 1, n - 1 do
		local r = ruleOf[t[i] * TBASE + t[i + 1]]
		if r then push(i, r) end
	end

	local work: { number } = {}
	local less = function(a: number, b: number): boolean
		return a < b
	end
	for r = 1, R do
		local h = heads[r]
		if h ~= nil then
			local k = 0
			local cur = h
			while cur ~= nil do
				k += 1
				work[k] = chain[cur]
				cur = chainNext[cur]
			end
			table.sort(work, less)
			local a = ms[2 * r - 1]
			local b = ms[2 * r]
			local cid = 256 + r - 1
			for j = 1, k do
				local p = work[j]
				if t[p] ~= a then continue end
				local q = nxt[p]
				if q == 0 or t[q] ~= b then continue end
				local pp = prv[p]
				local nn = nxt[q]
				t[p] = cid
				t[q] = -1 -- slot consumed
				prv[p] = pp
				if pp ~= 0 then nxt[pp] = p end
				nxt[p] = nn
				if nn ~= 0 then prv[nn] = p end
				if pp ~= 0 then
					local r2 = ruleOf[t[pp] * TBASE + cid]
					if r2 then push(pp, r2) end
				end
				if nn ~= 0 then
					local r2 = ruleOf[cid * TBASE + t[nn]]
					if r2 then push(p, r2) end
				end
			end
			for j = 1, k do work[j] = nil end
		end
	end

	local out: { number } = {}
	local s = 1
	while s ~= 0 do
		out[#out + 1] = t[s]
		s = nxt[s]
	end
	return out
end

-- Decode token ids back into a string.
function BPE.decode(self, ids: { number }): string
	local chars: { string } = table.create(#ids)
	for i = 1, #ids do chars[i] = self:piece(ids[i]) end
	return table.concat(chars)
end

-- Serializable tokenizer state: { vocab, merges }. Feeds BPE.fromState.
function BPE.state(self): any
	local ms = self.ms
	local out: { any } = table.create(#ms // 2)
	for r = 1, #ms, 2 do out[(r + 1) // 2] = { ms[r], ms[r + 1] } end
	return { vocab = self.vocab, merges = out }
end

-- Rebuild a tokenizer from a state produced by tok:state().
function BPE.fromState(st: any): any
	local self: any = setmetatable({
		ms = {},
		pieces = {},
		vocab = 256,
	}, BPE)
	for b = 0, 255 do self.pieces[b + 1] = string.char(b) end
	for r = 1, #st.merges do
		local a = st.merges[r][1]
		local b = st.merges[r][2]
		local newId = self.vocab
		table.insert(self.ms, a)
		table.insert(self.ms, b)
		self.pieces[newId + 1] = pieceConcat(self.pieces[a + 1], self.pieces[b + 1])
		self.vocab += 1
	end
	if st.vocab and st.vocab ~= self.vocab then
		error("BPE.fromState: vocab mismatch, state is corrupt")
	end
	return self
end

return BPE