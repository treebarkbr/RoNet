--!strict
-- Smoke test for the core Tensor engine: forward values + backward vs
-- finite differences. Run from the RoNetV4.1 directory:
--   /tmp/opencode/luau core/_smoke.luau

local Tensor = require("./Tensor")

local function approx(a: number, b: number, eps: number?): boolean
	local e = eps or 1e-3
	return math.abs(a - b) < e
end

local failures = 0
local function check(name: string, cond: boolean)
	if not cond then
		failures += 1
		print("FAIL: " .. name)
	end
end

-- finite-diff scalar->scalar derivative helper for a function of one tensor.
local function numericGrad(f: (any) -> any, t, shape)
	local eps = 1e-4
	local data = t.data
	local n = #data
	local grad = table.create(n)
	local base = f(t)
	for i = 1, n do
		data[i] += eps
		local hi = f(t)
		data[i] -= 2 * eps
		local lo = f(t)
		data[i] += eps
		grad[i] = (Tensor.item(hi) - Tensor.item(lo)) / (2 * eps)
	end
	return grad
end

-- ---------------- construction ----------------
local a = Tensor.fromData({ 1, 2, 3, 4 }, { 2, 2 })
check("fromData shape", #a.shape == 2 and a.shape[1] == 2 and a.shape[2] == 2)
local z = Tensor.zeros({ 3, 2 })
check("zeros", z.data[6] == 0)
local o = Tensor.ones({ 2 })
check("ones", o.data[1] == 1 and o.data[2] == 1)
local s = Tensor.scalar(5)
check("scalar", s.data[1] == 5)

-- ---------------- pointwise ----------------
local x = Tensor.fromData({ 1, 2, 3 }, { 3 })
local y = Tensor.fromData({ 10, 20, 30 }, { 3 })
check("add values", Tensor.item(Tensor.sum(Tensor.add(x, y))) == 66)
check("mul values", Tensor.item(Tensor.sum(Tensor.mul(x, y))) == 10 + 40 + 90)
local sub = Tensor.sub(y, x)
check("sub values", sub.data[1] == 9 and sub.data[3] == 27)

-- ---------------- broadcasting ----------------
local bt = Tensor.fromData({ 1, 2, 3, 4, 5, 6 }, { 2, 3 })
local br = Tensor.fromData({ 10, 100, 1000 }, { 3 })
local badd = Tensor.add(bt, br)
check("broadcast add shape", badd.shape[1] == 2 and badd.shape[2] == 3)
check("broadcast add v", badd.data[1] == 11 and badd.data[6] == 1006)

-- ---------------- matmul ----------------
local A = Tensor.fromData({ 1, 2, 3, 4, 5, 6 }, { 2, 3 })
local B = Tensor.fromData({ 7, 8, 9, 10, 11, 12 }, { 3, 2 })
local C = Tensor.matmul(A, B)
check("matmul shape", C.shape[1] == 2 and C.shape[2] == 2)
check("matmul v[1,1]", C.data[1] == 1 * 7 + 2 * 9 + 3 * 11) -- 58
check("matmul v[2,2]", C.data[4] == 4 * 8 + 5 * 10 + 6 * 12) -- 134

-- ---------------- bmm ----------------
local A3 = Tensor.fromData({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 2, 2, 2 })
local B3 = Tensor.fromData({ 1, 1, 1, 1, 1, 0, 1, 0 }, { 2, 2, 2 })
local C3 = Tensor.bmm(A3, B3)
check("bmm v", C3.data[1] == 3 and C3.data[4] == 7 and C3.data[5] == 11 and C3.data[8] == 0)

-- ---------------- softmax ----------------
local logits = Tensor.fromData({ 1, 2, 3, 1, 2, 3 }, { 2, 3 })
local probs = Tensor.softmax(logits, 2)
local row1 = probs.data[1] + probs.data[2] + probs.data[3]
check("softmax row sums 1", approx(row1, 1, 1e-6))

-- ---------------- autograd: matmul scalar loss ----------------
-- loss = sum(C), dLoss/dA[i,k] should equal sum_j B[k,j]
local Am = Tensor.fromData({ 1, 2, 3, 4 }, { 2, 2 }, true)
local Bm = Tensor.fromData({ 5, 6, 7, 8 }, { 2, 2 }, true)
local Cm = Tensor.matmul(Am, Bm)
local loss = Tensor.sum(Cm)
Tensor.backward(loss)
check("dL/dA[1,1]", Am.grad ~= nil and approx(Am.grad[1], 5 + 6, 1e-6))
check("dL/dA[1,2]", Am.grad ~= nil and approx(Am.grad[2], 7 + 8, 1e-6))
check("dL/dB[1,1]", Bm.grad ~= nil and approx(Bm.grad[1], 1 + 3, 1e-6))
check("dL/dB[2,2]", Bm.grad ~= nil and approx(Bm.grad[4], 2 + 4, 1e-6))

-- ---------------- autograd: gelu + mul + broadcast ----------------
local pg = Tensor.fromData({ 0.5, -1.2 }, { 2 }, true)
local pw = Tensor.fromData({ 2, 0.5 }, { 2 }, true)
local pscale = Tensor.fromData({ 3 }, { 1 }, true)
local out = Tensor.mul(Tensor.gelu(pg), pw)
out = Tensor.add(out, pscale) -- broadcast scalar
local lg = Tensor.sum(out)
Tensor.backward(lg)

local ng1 = numericGrad(function(t)
	local o1 = Tensor.mul(Tensor.gelu(t), pw)
	return Tensor.sum(Tensor.add(o1, pscale))
end, pg, { 2 })
check("fwd diff gelu deriv match", approx(pg.grad[1] :: number, ng1[1], 5e-3))

local ng2 = numericGrad(function(t)
	local o2 = Tensor.mul(Tensor.gelu(pg), t)
	return Tensor.sum(Tensor.add(o2, pscale))
end, pw, { 2 })
check("fwd diff mul deriv match", approx(pw.grad[1] :: number, ng2[1], 5e-3))

local ng3 = numericGrad(function(t)
	local o3 = Tensor.mul(Tensor.gelu(pg), pw)
	return Tensor.sum(Tensor.add(o3, t))
end, pscale, { 1 })
check("fwd diff scalar bcast deriv match", approx(pscale.grad[1] :: number, ng3[1], 5e-3))

-- ---------------- autograd: matmul through a broadcastable bias ----------------
local pb = Tensor.full({ 2 }, 0.1, true)
local Xb = Tensor.fromData({ 1, 2 }, { 1, 2 }, true)
local Wb = Tensor.fromData({ 3, 4 }, { 1, 2 }, false)
local hb = Tensor.add(Tensor.matmul(Xb, Tensor.transpose2d(Wb)), pb)
local lb = Tensor.sum(hb)
Tensor.backward(lb)
check("bias grad", pb.grad ~= nil and approx(pb.grad[1], 1, 1e-6))

-- ---------------- sum / mean / max along dim + full ----------------
local arr = Tensor.fromData({ 1, 2, 3, 4 }, { 2, 2 }, true)
local rowSum = Tensor.sum(arr, 2)
check("row sum shape", rowSum.shape[1] == 2)
check("row sum v", rowSum.data[1] == 3 and rowSum.data[2] == 7)
local colMean = Tensor.mean(arr, 1)
check("col mean", approx(colMean.data[1], 2, 1e-9) and approx(colMean.data[2], 3, 1e-9))
local allMax = Tensor.max(arr)
check("all max", allMax.item() == 4)

local rr = Tensor.sum(arr, 2)
local lrr = rr.data[1] + rr.data[2]
local rrSum = Tensor.add(rr, Tensor.zeros({ 2 }))
local _ = rrSum -- no-op keep
Tensor.backward(Tensor.sum(rr))
check("sum-dim backprop", arr.grad ~= nil and arr.grad[2] == 1 and arr.grad[3] == 1)

-- ---------------- reshape / transpose ----------------
local rshape = Tensor.reshape(arr, { 4 })
check("reshape", rshape.data[3] == 3)
local tr = Tensor.transpose2d(arr)
check("transpose", tr.shape[1] == 2 and tr.data[2] == 3)

-- ---------------- SIMD fast matmul (forward parity with scalar) ----------------
if rawget(_G, "vector") then
	local nA, nB, nK = 9, 8, 7 -- deliberately not a multiple of 3 lanes / k
	local Amat = Tensor.rand({ nA, nK })
	local Bmat = Tensor.rand({ nK, nB })
	local ct = Tensor.matmul(Amat, Bmat)
	local cf = Tensor.matmulFast(Amat, Bmat)
	check("matmulFast shape", cf.shape[1] == nA and cf.shape[2] == nB)
	local worst = 0
	local ad, bd = Amat.data, Bmat.data
	for j = 1, nB do
		for i = 1, nA do
			local expect = 0
			for kk = 1, nK do expect += ad[(i - 1) * nK + kk] * bd[(kk - 1) * nB + j] end
			local rel = math.abs(cf.data[(i - 1) * nB + j] - expect)
			if rel > worst then worst = rel end
		end
	end
	check("matmulFast parity", worst < 1e-6)
end

if failures == 0 then
	print("ALL CORE SMOKE TESTS PASSED")
else
	print(string.format("%d FAILURES", failures))
	error("smoke test failures")
end