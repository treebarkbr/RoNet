-- Predictable voc-head scaling: the cost of the token-prediction tail (one
-- [1, C] @ [V, C]^T lm-head matmul + last-dim softmax over V) as V grows from
-- small to massive, using the invested fused kernels (matmul auto-fast SIMD +
-- linear last-dim softmax). C = 288 (matches the TinyStories-15M import).
-- Honest timing: warmup + average over a fixed window, no best-of tricks.
local Tensor = require("../core/Tensor")
local function fill(n, fn)
	local t = table.create(n)
	for i = 1, n do t[i] = fn(i) end
	return t
end
local C = 288
local x = Tensor.fromData(fill(C, function(i) return ((i * 29) % 61) / 53 end), { 1, C }, false)
local function benchVocab(V, iters)
	local Wt = Tensor.fromData(fill(V * C, function(i) return ((i * 47) % 73) / 41 end), { C, V }, true)
	local Wt2 = Tensor.fromData(fill(V * C, function(i) return ((i * 47) % 73) / 41 end), { C, V }, false)
	local function step()
		local logits = Tensor.matmul(x, Wt2) -- auto-fast: [1, V]
		Tensor.softmax(logits, 2)
	end
	for _ = 1, 3 do step() end
	collectgarbage()
	local t0 = os.clock()
	for _ = 1, iters do step() end
	local per = (os.clock() - t0) / iters
	local function fmt(v)
		if v >= 1 then return string.format("%.0fms", v * 1e3) end
		if v * 1e3 >= 1 then return string.format("%.0fus", v * 1e6) end
		return string.format("%.0fns", v * 1e9)
	end
	print(string.format("vocab=%6d  lmHead(matmul+softmax) = %8s/step", V, fmt(per)))
	return per
end
print("--- interp ---")
for _, V in ipairs({ 256, 1024, 4096, 8192, 16384, 32000, 65536, 131072 }) do
	benchVocab(V, math.max(4, math.floor(8000000 / (V * C)) ))
	print("")
end
print("--- native ---")
for _, V in ipairs({ 256, 1024, 4096, 8192, 16384, 32000, 65536, 131072 }) do
	benchVocab(V, math.max(8, math.floor(80000000 / (V * C)) ))
	print("")
end