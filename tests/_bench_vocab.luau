--!nonstrict
local BPE = require("./../data/BPE")

local function buildVocab(nMerges: number)
	local tok = BPE.new()
	local rng = 48271
	local function nextByte()
		rng = (rng * 16807) % 2147483647
		return rng % 256
	end
	for i = 1, nMerges do
		tok:addMerge(nextByte(), nextByte())
	end
	return tok
end

local seq = ""
local rng2 = 104729
for i = 1, 800 do
	rng2 = (rng2 * 16807) % 2147483647
	seq = seq .. string.char(rng2 % 256)
end

for _, nM in ipairs({ 100000, 250000, 500000, 1000000 }) do
	collectgarbage("collect")
	local t0 = os.clock()
	local tok = buildVocab(nM)
	local buildTime = os.clock() - t0
	local kb = collectgarbage("count") -- in KiB, includes all heap so far
	collectgarbage("collect")
	local kbGc = collectgarbage("count")
	local t1 = os.clock()
	local toks = tok:encode(seq)
	local encTime = os.clock() - t1
	print(string.format(
		"vocab=%6d  build=%7.2fs  heap=%8.1f KiB (%.1f MiB)  encode800B=%7.3fs -> %d toks",
		nM + 256, buildTime, kbGc, kbGc / 1024, encTime, #toks
	))
end