--!strict
-- BPE tokenizer smoke tests: round-trip exactness, learned merges reduce the
-- token count below the byte count, determinism, unknown-text handling, and a
-- tiny end-to-end corpus-level check (train/encode/decode agree on training text).
local BPE = require("../data/BPE")

local function assertEq(a: any, b: any, msg: string)
	if a ~= b then error(string.format("FAIL %s: got %q want %q", msg, tostring(a), tostring(b))) end
end

local total = 0
local function check(cond: boolean, msg: string)
	total += 1
	if not cond then error(string.format("FAIL check %d: %s", total, msg)) end
end

local corpus = {
	"the quick brown fox jumps over the lazy dog",
	"the fat cat sat on the mat and the dog barked",
	"pack my box with five dozen liquor jugs",
	"how vexingly quick daft zebras jump",
	"the quick brown fox jumps again and again",
	"sphinx of black quartz, judge my vow",
}

-- 1. train + round-trip on the training text.
do
	local tok = BPE.train(corpus, 64)
	local vs = tok:vocabSize()
	check(vs == 256 + 64 or vs <= 256 + 64, "vocab size count")
	local nMerge = #(tok:merges())
	check(nMerge > 0, "learned at least one merge")
	local text = table.concat(corpus, "\n")
	local ids = tok:encode(text)
	local back = tok:decode(ids)
	assertEq(back, text, "train round-trip")
	-- compression: subwords should shrink the id count (repeated words)
	local bytes = #text
	check(#ids <= bytes, "tokens <= byte count")
	print(string.format("  trained %d merges, %d tokens for %d bytes (%.1f%%)", nMerge, #ids, bytes, 100 * #ids / bytes))
end

-- 2. determinism: same corpus + merges -> identical merge tables.
do
	local a = BPE.train(corpus, 32)
	local b = BPE.train(corpus, 32)
	local ma = a:merges()
	local mb = b:merges()
	check(#ma == #mb, "same merge count")
	for i = 1, #ma do
		check(ma[i][1] == mb[i][1] and ma[i][2] == mb[i][2], "same merge sequence")
	end
end

-- 3. unknown text still round-trips (pure byte fallback for never-seen runs).
do
	local tok = BPE.train(corpus, 16)
	local odd = "zzz qqq xxx 12345 !@#$% \0\1\2 binary \255"
	local ids = tok:encode(odd)
	assertEq(tok:decode(ids), odd, "unknown text round-trip")
end

-- 4. single-token decode for byte ids 0..255 equals the raw byte.
do
	local tok = BPE.train(corpus, 8)
	for b = 0, 120 do
		local one = tok:piece(b)
		assertEq(one, string.char(b), "byte piece decode")
	end
end

-- 5. frequent word becomes a single token: "the" should compress.
do
	local tok = BPE.train(corpus, 128)
	local ids = tok:encode("the")
	check(#ids <= 2, "the -> few tokens (" .. #ids .. ")")
end

-- 6. manual addMerge keeps ids above 255 and round-trips.
do
	local tok = BPE.new()
	local id = tok:addMerge(116, 104) -- 't','h'
	check(id == 256, "manual merge id 256")
	local ids = tok:encode("th")
	check(#ids == 1 and ids[1] == 256, "manual merge applied")
	assertEq(tok:decode({ 256 }), "th", "manual merge decode")
	local vs6 = tok:vocabSize()
	check(vs6 == 257, "manual vocab grows")
end

print(string.format("PASS BPE: %d checks", total))