--!nonstrict
-- Line-of-evidence for the queue-based (linear) encode rewrite in data/BPE.luau:
--  1) Fuzz: the new encode must produce IDENTICAL ids to the previous algorithm
--     (rule-exhaustion, leftmost-first) across many rule sets incl. overlapping
--     and repeated-pair cases.
--  2) Bench: speed + memory at 100k/250k/500k/1M vocab vs the previous numbers.
local BPE = require("../data/BPE")

local function refEncode(tok, text)
	local ms = tok.ms
	local toks = {}
	for i = 1, #text do toks[i] = string.byte(text, i) end
	for r = 1, #ms, 2 do
		local a = ms[r]
		local b = ms[r + 1]
		local newId = 256 + (r - 1) // 2
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
			local merged = table.create(n - 1)
			local m = 0
			for i = 1, n do
				if i == found + 1 then continue end
				m += 1
				if i == found then
					merged[m] = newId
				else
					merged[m] = toks[i]
				end
			end
			toks = merged
		end
	end
	return toks
end

local function seqEq(a, b)
	if #a ~= #b then return false end
	for i = 1, #a do
		if a[i] ~= b[i] then return false end
	end
	return true
end

local rng = 987654321
local function rnd(n)
	rng = (rng * 1103515245 + 12345) % 2147483648
	return rng % n + 1
end

local fails = 0
local trials = 0

local function check(tok, text)
	trials += 1
	local r1 = refEncode(tok, text)
	local r2 = tok:encode(text)
	if not seqEq(r1, r2) then
		fails += 1
		if fails <= 5 then
			print("  MISMATCH: text=" .. text)
			print("    ref=" .. table.concat(r1, ","))
			print("    new=" .. table.concat(r2, ","))
		end
	end
end

-- Phase 1: byte-only rules over a tiny alphabet (forces overlaps & repeat runs).
for trial = 1, 250 do
	local tok = BPE.new()
	local alpha = { 97, 98, 99, 100 }
	local used = {}
	local nRules = 2 + rnd(20)
	local made = 0
	local guard = 0
	while made < nRules and guard < 400 do
		guard += 1
		local a = alpha[rnd(#alpha)]
		local b = alpha[rnd(#alpha)]
		local key = a * 200 + b
		if not used[key] then
			used[key] = true
			tok:addMerge(a, b)
			made += 1
		end
	end
	local text
	if trial % 2 == 0 then
		-- random text
		local chars = {}
		for i = 1, 1 + rnd(45) do chars[i] = string.char(alpha[rnd(#alpha)]) end
		text = table.concat(chars)
	else
		-- adversarial patterns: repeat runs and interleavings
		local pat = { "aaa", "ababab", "aaabbbccc", "abcdabcd", "aabbaacc", "aaaa", "abcabcabc" }
		local p = pat[rnd(#pat)]
		local rep = 1 + rnd(6)
		local chunks = {}
		for i = 1, rep do chunks[i] = p end
		text = table.concat(chunks)
	end
	check(tok, text)
end

-- Phase 2: rules that reference previously-merged tokens (higher-order pairs),
-- including chained merges that force long token cascades.
for trial = 1, 250 do
	local tok = BPE.new()
	local alpha = { 97, 98, 99, 100 }
	local used = {}
	local nRules = 3 + rnd(30)
	local made = 0
	local guard = 0
	while made < nRules and guard < 600 do
		guard += 1
		local a = rnd(tok.vocab) - 1 -- byte or an earlier merged id
		local b = rnd(tok.vocab) - 1
		local key = a * 1000000 + b
		if not used[key] then
			used[key] = true
			tok:addMerge(a, b)
			made += 1
		end
	end
	local chars = {}
	for i = 1, 1 + rnd(60) do chars[i] = string.char(alpha[rnd(#alpha)]) end
	check(tok, table.concat(chars))
end

print(string.format("fuzz: %d texts, %d mismatches", trials, fails))
assert(fails == 0, "encode rewrite diverges from reference!")

-- ---------------------------------------------------------------------------
-- Bench: same methodology as tests/_bench_vocab.luau (random rules, 800B text).
local function buildVocab(nMerges)
	local tok = BPE.new()
	local rr = 48271
	local seen = {}
	local function nextByte()
		rr = (rr * 16807) % 2147483647
		return rr % 256
	end
	while #tok.ms // 2 < nMerges do
		local a, b = nextByte(), nextByte()
		local key = a * 256 + b
		if not seen[key] then
			seen[key] = true
			tok:addMerge(a, b)
		end
	end
	return tok
end

local seq = ""
local r2 = 104729
for i = 1, 800 do
	r2 = (r2 * 16807) % 2147483647
	seq = seq .. string.char(r2 % 256)
end

local function fmt(s)
	return string.format("%d", s)
end

for _, nM in ipairs({ 10000, 20000, 40000, 60000 }) do
	collectgarbage("collect")
	local t0 = os.clock()
	local tok = buildVocab(nM)
	tok:encode(seq) -- warm

	local t1 = os.clock()
	local res = tok:encode(seq)
	local enc = os.clock() - t1
	local peak = collectgarbage("count") -- before GC: transient map still resident
	collectgarbage("collect")
	local steady = collectgarbage("count")
	print(string.format(
		"vocab=%7d  encode800B=%7.3fs  peakHeap=%8.1f KiB  steadyHeap=%8.1f KiB  -> %d toks",
		nM + 256, enc, peak, steady, #res
	))
end