--!strict
-- Tensor: an N-dimensional tensor on a flat row-major array with reverse-mode
-- autograd (define-by-run). This is what makes RoNetV4.1 capable of any
-- network: write a forward pass with tensor ops and the recorded graph knows
-- how to differentiate everything.
--
-- Conventions:
--   * data is a flat {number}, row-major (last dim contiguous).
--   * shape {} denotes a scalar (size 1).
--   * Every op that produces a tensor with requiresGrad=true records a
--     _backward closure that flows .grad into its parents during backward().
--
-- Part 1: constructors, plumbing, pointwise ops, matmul.
--
-- IMPORTANT (CLI limitation): modules must not require() siblings; only the
-- entry script can. Tensor is therefore self-contained. PRNG is injected
-- through Tensor.setRNG() from the root loader so runs stay reproducible.

local Tensor = {}

export type RNG = {
	nextFloat: (self: any) -> number,
	nextGaussian: (self: any) -> number,
}

local function defaultGaussian(self: any): number
	local u1 = math.random()
	if u1 == 0 then u1 = 1e-12 end
	local u2 = math.random()
	return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
end

local defaultRng: RNG = {
	nextFloat = function(self) return math.random() end,
	nextGaussian = defaultGaussian,
}

function Tensor.setRNG(rng: RNG)
	defaultRng = rng
end


-- Forward-declared upvalues so the public reduce wrappers can close over them
-- even though they are defined later in the file (Lua scoping rule).
local UtilFillRepeat: (number, number) -> {number}
local reduceAlongDim: (Tensor, number, string) -> Tensor

local Meta: {__index: any} = {}
Meta.__index = Meta

local math_sqrt = math.sqrt
local math_exp = math.exp
local math_floor = math.floor
local math_max = math.max
local math_min = math.min

export type Tensor = {
	shape: {number},
	data: {number},
	requiresGrad: boolean,
	grad: {number}?,
	_backward: (() -> ())?,
	_prev: {Tensor},
	_op: string?,
	backward: (retainGraph: boolean?) -> (),
	numel: () -> number,
	item: () -> number,
}

local function sizeOf(shape: {number}): number
	local n = 1
	for i = 1, #shape do n *= shape[i] end
	return n
end

local function copyData(data: {number}): {number}
	local n = #data
	local out = table.create(n)
	table.move(data, 1, n, 1, out)
	return out
end

local function makeRawTensor(shape: {number}, data: {number}, requiresGrad: boolean, prev: {Tensor}?, op: string?): Tensor
	local t: any = {
		shape = shape,
		data = data,
		requiresGrad = requiresGrad,
		grad = nil,
		_backward = nil,
		_prev = prev or {},
		_op = op,
	}
	setmetatable(t, Meta)
	-- Attach methods directly (closure over t). This avoids relying on
	-- metatable __index for method calls, which the Luau CLI binds
	-- unreliably for `obj.method()` style calls.
	t.item = function() return t.data[1] end
	t.numel = function() return #t.data end
	t.backward = function(retainGraph: boolean?)
		Tensor.backward(t, retainGraph)
	end
	return (t :: Tensor)
end

--- Constructors -----------------------------------------------------------------

function Tensor.fromData(data: {number}, shape: {number}, requiresGrad: boolean?): Tensor
	local n = sizeOf(shape)
	if #data ~= n then
		error(string.format("fromData size mismatch: got %d elements for shape %s", #data, table.concat(shape, "x")), 2)
	end
	return makeRawTensor(shape, copyData(data), requiresGrad == true, nil, "data")
end

function Tensor.zeros(shape: {number}, requiresGrad: boolean?): Tensor
	local n = sizeOf(shape)
	return makeRawTensor(shape, table.create(n, 0), requiresGrad == true, nil, "zeros")
end

function Tensor.ones(shape: {number}, requiresGrad: boolean?): Tensor
	local n = sizeOf(shape)
	local data = table.create(n)
	for i = 1, n do data[i] = 1 end
	return makeRawTensor(shape, data, requiresGrad == true, nil, "ones")
end

function Tensor.full(shape: {number}, value: number, requiresGrad: boolean?): Tensor
	local n = sizeOf(shape)
	local data = table.create(n)
	for i = 1, n do data[i] = value end
	return makeRawTensor(shape, data, requiresGrad == true, nil, "full")
end

function Tensor.scalar(value: number, requiresGrad: boolean?): Tensor
	return makeRawTensor({}, { value }, requiresGrad == true, nil, "scalar")
end

function Tensor.randn(shape: {number}, rng: RNG?, requiresGrad: boolean?): Tensor
	local r = rng or defaultRng
	local n = sizeOf(shape)
	local data = table.create(n)
	for i = 1, n do data[i] = r.nextGaussian() end
	return makeRawTensor(shape, data, requiresGrad == true, nil, "randn")
end

function Tensor.rand(shape: {number}, rng: RNG?, requiresGrad: boolean?): Tensor
	local r = rng or defaultRng
	local n = sizeOf(shape)
	local data = table.create(n)
	for i = 1, n do data[i] = r.nextFloat() end
	return makeRawTensor(shape, data, requiresGrad == true, nil, "rand")
end

function Tensor.uniform(shape: {number}, lo: number, hi: number, rng: RNG?, requiresGrad: boolean?): Tensor
	local r = rng or defaultRng
	local span = hi - lo
	local n = sizeOf(shape)
	local data = table.create(n)
	for i = 1, n do data[i] = lo + r.nextFloat() * span end
	return makeRawTensor(shape, data, requiresGrad == true, nil, "uniform")
end

--- Methods ----------------------------------------------------------------------

function Tensor.numel(t: Tensor): number
	return #t.data
end

function Tensor.shapeOf(t: Tensor): {number}
	return t.shape
end

function Tensor.isScalar(t: Tensor): boolean
	return #t.shape == 0
end

function Tensor.item(t: Tensor): number
	return t.data[1]
end

-- Copies data, detaches from the graph.
function Tensor.clone(t: Tensor): Tensor
	return makeRawTensor(copyData(t.shape), copyData(t.data), false, nil, "clone")
end

-- A view sharing data with requiresGrad=false.
function Tensor.detach(t: Tensor): Tensor
	return makeRawTensor(t.shape, t.data, false, nil, "detach")
end

function Tensor.ensureGrad(t: Tensor): {number}
	local g = t.grad or table.create(#t.data, 0)
	t.grad = g
	return g
end

function Tensor.zeroGrad(t: Tensor)
	if t.grad then
		for i = 1, #t.grad do t.grad[i] = 0 end
	end
end

function Tensor.gradData(t: Tensor): {number}?
	return t.grad
end

--- Grad plumbing -----------------------------------------------------------

local function accumulateGrad(t: Tensor, g: {number})
	if not t.requiresGrad then return end
	local target = Tensor.ensureGrad(t)
	for i = 1, #g do target[i] += g[i] end
end

--- Broadcasting helpers --------------------------------------------------------

-- Right-aligned broadcast shape.
local function broadcastShape(a: {number}, b: {number}): {number}
	local nd = math_max(#a, #b)
	local out = table.create(nd)
	for d = 1, nd do
		local ad = a[#a - nd + d] or 1
		local bd = b[#b - nd + d] or 1
		if ad ~= bd and ad ~= 1 and bd ~= 1 then
			error(string.format("broadcastShape: incompatible dims (%d vs %d) at axis %d", ad, bd, d), 2)
		end
		out[d] = math_max(ad, bd)
	end
	return out
end

-- Map an output linear index (1-based) back to a source linear index (1-based)
-- under right-aligned broadcasting.
local function mapIndex(i0: number, inShape: {number}, outShape: {number}): number
	local nd = #outShape
	local rem = i0 - 1
	local offs = 0
	local place = 1
	for d = nd, 1, -1 do
		local outDim = outShape[d]
		local inDim = inShape[#inShape - nd + d] or 1
		local inPos = (rem % outDim) % inDim
		rem = math_floor(rem / outDim)
		offs += inPos * place
		place *= inDim
	end
	return offs + 1
end

local function sameShape(a: {number}, b: {number}): boolean
	if #a ~= #b then return false end
	for i = 1, #a do
		if a[i] ~= b[i] then return false end
	end
	return true
end

-- Reduce a per-output gradient array down to a per-source array by summing the
-- broadcast copies (partials already scaled by upstream grad by callers).
local function reduceGrad(partials: {number}, inShape: {number}, outShape: {number}): {number}
	if sameShape(inShape, outShape) then return copyData(partials) end
	local inSize = sizeOf(inShape)
	local outSize = sizeOf(outShape)
	local reduced = table.create(inSize, 0)
	for i = 1, outSize do
		local src = mapIndex(i, inShape, outShape)
		reduced[src] += partials[i]
	end
	return reduced
end

--- Pointwise binary ops ----------------------------------------------------------

type PwFn = (number, number) -> number

-- Core engine for broadcastable binary pointwise ops. f gives forward,
-- da/db give pointwise partials. Grads are reduced against broadcast dims.
local function pointwiseBinary(a: Tensor, b: Tensor, opName: string, da: PwFn, db: PwFn, f: PwFn): Tensor
	local outShape = broadcastShape(a.shape, b.shape)
	local outSize = sizeOf(outShape)
	local out = table.create(outSize)
	local aSize = #a.data
	local bSize = #b.data
	local needsGrad = a.requiresGrad or b.requiresGrad
	local singleA = aSize == 1
	local singleB = bSize == 1

	if singleA or singleB then
		if singleA and singleB then
			out[1] = f(a.data[1], b.data[1])
		elseif singleA then
			for i = 1, outSize do out[i] = f(a.data[1], b.data[i]) end
		else
			for i = 1, outSize do out[i] = f(a.data[i], b.data[1]) end
		end
	else
		local aShape, bShape = a.shape, b.shape
		for i = 1, outSize do
			local ai = mapIndex(i, aShape, outShape)
			local bi = mapIndex(i, bShape, outShape)
			out[i] = f(a.data[ai], b.data[bi])
		end
	end

	local res = makeRawTensor(outShape, out, needsGrad, { a, b }, opName)
	if needsGrad then
		local aShapeSaved = copyData(a.shape)
		local bShapeSaved = copyData(b.shape)
		res._backward = function()
			local grad = Tensor.ensureGrad(res)
			if a.requiresGrad then
				if singleA then
					local s = 0
					for i = 1, outSize do
						local bi = singleB and 1 or mapIndex(i, bShapeSaved, outShape)
						s += da(a.data[1], b.data[bi]) * grad[i]
					end
					accumulateGrad(a, { s })
				else
					local ga = table.create(outSize, 0)
					for i = 1, outSize do
						local ai = mapIndex(i, aShapeSaved, outShape)
						local bi = singleB and 1 or mapIndex(i, bShapeSaved, outShape)
						ga[i] = da(a.data[ai], b.data[bi]) * grad[i]
					end
					accumulateGrad(a, reduceGrad(ga, aShapeSaved, outShape))
				end
			end
			if b.requiresGrad then
				if singleB then
					local s = 0
					for i = 1, outSize do
						local ai = singleA and 1 or mapIndex(i, aShapeSaved, outShape)
						s += db(a.data[ai], b.data[1]) * grad[i]
					end
					accumulateGrad(b, { s })
				else
					local gb = table.create(outSize, 0)
					for i = 1, outSize do
						local ai = singleA and 1 or mapIndex(i, aShapeSaved, outShape)
						local bi = mapIndex(i, bShapeSaved, outShape)
						gb[i] = db(a.data[ai], b.data[bi]) * grad[i]
					end
					accumulateGrad(b, reduceGrad(gb, bShapeSaved, outShape))
				end
			end
		end
	end
	return res
end

function Tensor.add(a: Tensor, b: Tensor): Tensor
	return pointwiseBinary(a, b, "add",
		function() return 1 end,
		function() return 1 end,
		function(av, bv) return av + bv end)
end

function Tensor.sub(a: Tensor, b: Tensor): Tensor
	return pointwiseBinary(a, b, "sub",
		function() return 1 end,
		function() return -1 end,
		function(av, bv) return av - bv end)
end

function Tensor.mul(a: Tensor, b: Tensor): Tensor
	return pointwiseBinary(a, b, "mul",
		function(_, bv) return bv end,
		function(av) return av end,
		function(av, bv) return av * bv end)
end

function Tensor.div(a: Tensor, b: Tensor): Tensor
	return pointwiseBinary(a, b, "div",
		function(_, bv) return 1 / bv end,
		function(av, bv) return -av / (bv * bv) end,
		function(av, bv) return av / bv end)
end

function Tensor.mulScalar(a: Tensor, s: number): Tensor
	return Tensor.mul(a, Tensor.scalar(s))
end

function Tensor.addScalar(a: Tensor, s: number): Tensor
	return Tensor.add(a, Tensor.scalar(s))
end

function Tensor.neg(a: Tensor): Tensor
	return Tensor.mulScalar(a, -1)
end

function Tensor.pow(a: Tensor, p: number): Tensor
	return pointwiseBinary(a, Tensor.scalar(p), "pow",
		function(av) return p * (av ^ (p - 1)) end,
		function() return 0 end,
		function(av) return av ^ p end)
end

function Tensor.exp(a: Tensor): Tensor
	return pointwiseBinary(a, Tensor.scalar(1), "exp",
		function(av) return math_exp(av) end,
		function() return 0 end,
		function(av) return math_exp(av) end)
end

function Tensor.log(a: Tensor): Tensor
	return pointwiseBinary(a, Tensor.scalar(1), "log",
		function(av) return 1 / math_max(av, 1e-12) end,
		function() return 0 end,
		function(av) return math.log(math_max(av, 1e-12)) end)
end

function Tensor.sqrt(a: Tensor): Tensor
	return pointwiseBinary(a, Tensor.scalar(1), "sqrt",
		function(av) return 0.5 / math_sqrt(math_max(av, 1e-12)) end,
		function() return 0 end,
		function(av) return math_sqrt(math_max(av, 0)) end)
end

local function sigmoidValue(x: number): number
	if x >= 0 then
		local z = math_exp(-x)
		return 1 / (1 + z)
	else
		local z = math_exp(x)
		return z / (1 + z)
	end
end

function Tensor.tanh(a: Tensor): Tensor
	return pointwiseBinary(a, Tensor.scalar(1), "tanh",
		function(av)
			local t = math.tanh(av)
			return 1 - t * t
		end,
		function() return 0 end,
		function(av) return math.tanh(av) end)
end

function Tensor.sigmoid(a: Tensor): Tensor
	return pointwiseBinary(a, Tensor.scalar(1), "sigmoid",
		function(av)
			local s = sigmoidValue(av)
			return s * (1 - s)
		end,
		function() return 0 end,
		function(av) return sigmoidValue(av) end)
end

function Tensor.relu(a: Tensor): Tensor
	return pointwiseBinary(a, Tensor.scalar(1), "relu",
		function(av) return if av > 0 then 1 else 0 end,
		function() return 0 end,
		function(av) return math_max(av, 0) end)
end

-- Swish / SiLU: x * sigmoid(x)
function Tensor.silu(a: Tensor): Tensor
	return pointwiseBinary(a, Tensor.scalar(1), "silu",
		function(av)
			local s = sigmoidValue(av)
			return s + av * s * (1 - s)
		end,
		function() return 0 end,
		function(av) return av * sigmoidValue(av) end)
end

-- GELU (tanh approximation, as in GPT-2/J and most libs).
function Tensor.gelu(a: Tensor): Tensor
	return pointwiseBinary(a, Tensor.scalar(1), "gelu",
		function(av)
			local c = 0.7978845608028654
			local t = math.tanh(c * (av + 0.044715 * av * av * av))
			local dt = c * (1 + 3 * 0.044715 * av * av) * (1 - t * t)
			return 0.5 * (1 + t) + 0.5 * av * dt
		end,
		function() return 0 end,
		function(av)
			local c = 0.7978845608028654
			local t = math.tanh(c * (av + 0.044715 * av * av * av))
			return 0.5 * av * (1 + t)
		end)
end

function Tensor.clip(a: Tensor, lo: number, hi: number): Tensor
	return pointwiseBinary(a, Tensor.scalar(1), "clip",
		function(av) return if av > lo and av < hi then 1 else 0 end,
		function() return 0 end,
		function(av) return math_min(math_max(av, lo), hi) end)
end

--- Matmul -----------------------------------------------------------------------

-- C[i,j] = sum_k A[i,k] * B[k,j]  (2D only; use bmm for batches).
function Tensor.matmul(a: Tensor, b: Tensor): Tensor
	local ash, bsh = a.shape, b.shape
	if #ash ~= 2 or #bsh ~= 2 then
		error("matmul currently supports 2D tensors only", 2)
	end
	if ash[2] ~= bsh[1] then
		error(string.format("matmul dim mismatch: [%s] @ [%s]", table.concat(ash, "x"), table.concat(bsh, "x")), 2)
	end
	local m, k, n = ash[1], ash[2], bsh[2]
	local outShape = { m, n }
	local out = table.create(m * n, 0)
	local ad, bd = a.data, b.data

	for i = 1, m do
		local ioff = (i - 1) * k
		for j = 1, n do
			local sum = 0
			for kk = 1, k do
				sum += ad[ioff + kk] * bd[(kk - 1) * n + j]
			end
			out[(i - 1) * n + j] = sum
		end
	end

	local needsGrad = a.requiresGrad or b.requiresGrad
	local res = makeRawTensor(outShape, out, needsGrad, { a, b }, "matmul")
	if needsGrad then
		res._backward = function()
			local grad = Tensor.ensureGrad(res)
			if a.requiresGrad then
				local ga = table.create(m * k, 0)
				for i = 1, m do
					local aoff = (i - 1) * k
					for j = 1, n do
						local g = grad[(i - 1) * n + j]
						for kk = 1, k do
							ga[aoff + kk] += g * bd[(kk - 1) * n + j]
						end
					end
				end
				accumulateGrad(a, ga)
			end
			if b.requiresGrad then
				local gb = table.create(k * n, 0)
				for i = 1, m do
					local aoff = (i - 1) * k
					for j = 1, n do
						local g = grad[(i - 1) * n + j]
						for kk = 1, k do
							gb[(kk - 1) * n + j] += ad[aoff + kk] * g
						end
					end
				end
				accumulateGrad(b, gb)
			end
		end
	end
	return res
end

-- Batched matmul: [B, M, K] @ [B, K, N] -> [B, M, N].
function Tensor.bmm(a: Tensor, b: Tensor): Tensor
	local ash, bsh = a.shape, b.shape
	if #ash ~= 3 or #bsh ~= 3 or ash[1] ~= bsh[1] or ash[3] ~= bsh[2] then
		error(string.format("bmm shape mismatch: [%s] @ [%s]", table.concat(ash, "x"), table.concat(bsh, "x")), 2)
	end
	local B, M, K, N = ash[1], ash[2], ash[3], bsh[3]
	local outShape = { B, M, N }
	local out = table.create(B * M * N, 0)
	local ad, bd = a.data, b.data

	for bb = 1, B do
		local abase = (bb - 1) * M * K
		local bbase = (bb - 1) * K * N
		local obase = (bb - 1) * M * N
		for i = 1, M do
			local ioff = abase + (i - 1) * K
			for j = 1, N do
				local sum = 0
				for kk = 1, K do
					sum += ad[ioff + kk] * bd[bbase + (kk - 1) * N + j]
				end
				out[obase + (i - 1) * N + j] = sum
			end
		end
	end

	local needsGrad = a.requiresGrad or b.requiresGrad
	local res = makeRawTensor(outShape, out, needsGrad, { a, b }, "bmm")
	if needsGrad then
		res._backward = function()
			local grad = Tensor.ensureGrad(res)
			if a.requiresGrad then
				local ga = table.create(B * M * K, 0)
				for bb = 1, B do
					local abase = (bb - 1) * M * K
					local bbase = (bb - 1) * K * N
					local obase = (bb - 1) * M * N
					for i = 1, M do
						for j = 1, N do
							local g = grad[obase + (i - 1) * N + j]
							for kk = 1, K do
								ga[abase + (i - 1) * K + kk] += g * bd[bbase + (kk - 1) * N + j]
							end
						end
					end
				end
				accumulateGrad(a, ga)
			end
			if b.requiresGrad then
				local gb = table.create(B * K * N, 0)
				for bb = 1, B do
					local abase = (bb - 1) * M * K
					local bbase = (bb - 1) * K * N
					local obase = (bb - 1) * M * N
					for i = 1, M do
						for j = 1, N do
							local g = grad[obase + (i - 1) * N + j]
							for kk = 1, K do
								gb[bbase + (kk - 1) * N + j] += ad[abase + (i - 1) * K + kk] * g
							end
						end
					end
				end
				accumulateGrad(b, gb)
			end
		end
	end
	return res
end

--- Reshape / transpose ---------------------------------------------------------

function Tensor.reshape(t: Tensor, newShape: {number}): Tensor
	local newSize = sizeOf(newShape)
	if newSize ~= #t.data then
		error(string.format("reshape size mismatch: %d -> [%s]", #t.data, table.concat(newShape, "x")), 2)
	end
	local res = makeRawTensor(copyData(newShape), t.data, t.requiresGrad, { t }, "reshape")
	if t.requiresGrad then
		res._backward = function()
			accumulateGrad(t, Tensor.ensureGrad(res))
		end
	end
	return res
end

-- Flatten to [rows, cols]; either may be -1 (inferred).
function Tensor.flatten(t: Tensor, rows: number, cols: number): Tensor
	local r = rows
	local c = cols
	if r < 0 then r = #t.data / c end
	if c < 0 then c = #t.data / r end
	return Tensor.reshape(t, { r, c })
end

--- Coords helpers for reductions ------------------------------------------------

-- 1-based flat index from 0-based coords.
local function indexOf(shape: {number}, coords0: {number}): number
	local idx = 1
	for d = 1, #shape do
		idx = (idx - 1) * shape[d] + coords0[d] + 1
	end
	return idx
end

-- 0-based coords from a 0-based flat index.
local function coordsOf(shape: {number}, idx0: number): {number}
	local nd = #shape
	local c = table.create(nd)
	local rem = idx0
	for d = nd, 1, -1 do
		local dim = shape[d]
		c[d] = rem % dim
		rem = math_floor(rem / dim)
	end
	return c
end

-- Given output coords where dim `dim` is removed, rebuild full coords with the
-- reduced position `reducedPos` inserted at axis `dim`.
local function insertCoord(prefixCoords: {number}, dim: number, reducedPos: number, nd: number): {number}
	local c = table.create(nd)
	local pd = 1
	for d = 1, nd do
		if d == dim then
			c[d] = reducedPos
		else
			c[d] = prefixCoords[pd]
			pd += 1
		end
	end
	return c
end

-- Sum reduction: -1 dim means full reduction to a scalar. Backward routes the
-- scalar upstream grad to every element (for full reduce) or to the matching
-- column (for a specific dim).
function Tensor.sum(t: Tensor, dim: number?) : Tensor
	if dim == nil or dim == -1 then
		local s = 0
		for i = 1, #t.data do s += t.data[i] end
		local res = makeRawTensor({}, { s }, t.requiresGrad, { t }, "sumAll")
		if t.requiresGrad then
			res._backward = function()
				local g = Tensor.ensureGrad(res)[1]
				accumulateGrad(t, UtilFillRepeat(g, #t.data))
			end
		end
		return res
	end
	return reduceAlongDim(t, dim, "sum")
end

-- Mean reduction.
function Tensor.mean(t: Tensor, dim: number?) : Tensor
	if dim == nil or dim == -1 then
		local nEl = #t.data
		local s = 0
		for i = 1, nEl do s += t.data[i] end
		local res = makeRawTensor({}, { s / nEl }, t.requiresGrad, { t }, "meanAll")
		if t.requiresGrad then
			res._backward = function()
				local inv = 1 / nEl
				local g = Tensor.ensureGrad(res)[1] * inv
				accumulateGrad(t, UtilFillRepeat(g, nEl))
			end
		end
		return res
	end
	return reduceAlongDim(t, dim, "mean")
end

-- Max reduction along a single dim (or full max). Records the argmax so the
-- backward can route the upstream grad to exactly the winner positions.
function Tensor.max(t: Tensor, dim: number?) : Tensor
	if dim == nil or dim == -1 then
		local nEl = #t.data
		local best = -math.huge
		local bestIdx = 1
		for i = 1, nEl do
			if t.data[i] > best then
				best = t.data[i]
				bestIdx = i
			end
		end
		local res = makeRawTensor({}, { best }, t.requiresGrad, { t }, "maxAll")
		if t.requiresGrad then
			res._backward = function()
				local g = Tensor.ensureGrad(res)[1]
				local ga = table.create(nEl, 0)
				ga[bestIdx] = g
				accumulateGrad(t, ga)
			end
		end
		return res
	end
	return reduceAlongDim(t, dim, "max")
end

-- Squared L2 norm, used by vector-like regularization / normalization helpers.
function Tensor.sumSquares(t: Tensor): Tensor
	local s = 0
	for i = 1, #t.data do
		local v = t.data[i]
		s += v * v
	end
	local res = makeRawTensor({}, { s }, t.requiresGrad, { t }, "sumSquares")
	if t.requiresGrad then
		res._backward = function()
			local g = Tensor.ensureGrad(res)[1]
			local ga = table.create(#t.data)
			for i = 1, #t.data do ga[i] = 2 * t.data[i] * g end
			accumulateGrad(t, ga)
		end
	end
	return res
end

-- Shared implementation of sum / mean / max over a single dim.
reduceAlongDim = function(t: Tensor, dim: number, kind: string): Tensor
	local nd = #t.shape
	if dim < 1 or dim > nd then
		error(string.format("reduce dim %d out of range for %d-D tensor", dim, nd), 2)
	end
	local reducedSize = t.shape[dim]
	local outShape = table.create(nd - 1)
	for d = 1, nd do
		if d ~= dim then outShape[#outShape + 1] = t.shape[d] end
	end
	local outSize = sizeOf(outShape)
	local out = table.create(outSize, 0)
	local winners: {number} = if kind == "max" then table.create(outSize, 0) else {}

	for oi = 1, outSize do
		local oc = coordsOf(outShape, oi - 1)
		if kind == "max" then
			local best = -math.huge
			local bestPos = 1
			for r = 1, reducedSize do
				local idx = indexOf(t.shape, insertCoord(oc, dim, r - 1, nd))
				local v = t.data[idx]
				if v > best then
					best = v
					bestPos = r
				end
			end
			out[oi] = best
			winners[oi] = bestPos
		elseif kind == "sum" then
			local s = 0
			for r = 1, reducedSize do
				local idx = indexOf(t.shape, insertCoord(oc, dim, r - 1, nd))
				s += t.data[idx]
			end
			out[oi] = s
		else -- mean
			local s = 0
			for r = 1, reducedSize do
				local idx = indexOf(t.shape, insertCoord(oc, dim, r - 1, nd))
				s += t.data[idx]
			end
			out[oi] = s / reducedSize
		end
	end

	local res = makeRawTensor(outShape, out, t.requiresGrad, { t }, kind .. "Dim")
	if t.requiresGrad then
		res._backward = function()
			local grad = Tensor.ensureGrad(res)
			local ga = table.create(#t.data, 0)
			for oi = 1, outSize do
				local oc = coordsOf(outShape, oi - 1)
				if kind == "max" then
					local w = winners :: {number}
					ga[indexOf(t.shape, insertCoord(oc, dim, w[oi] - 1, nd))] += grad[oi]
				else
					local factor = kind == "mean" and 1 / reducedSize or 1
					local g = grad[oi] * factor
					for r = 1, reducedSize do
						local idx = indexOf(t.shape, insertCoord(oc, dim, r - 1, nd))
						ga[idx] += g
					end
				end
			end
			accumulateGrad(t, ga)
		end
	end
	return res
end

UtilFillRepeat = function(value: number, n: number): {number}
	local out = table.create(n)
	for i = 1, n do out[i] = value end
	return out
end

--- Softmax ----------------------------------------------------------------------

-- Softmax along a single dim (default: last) - used on [B,V] logits and the
-- on output rows of a [T,D] sequence. Backward uses the standard
-- dPi/dXj = Pi*(delta_ij - Pj).
function Tensor.softmax(t: Tensor, dim: number?): Tensor
	local dimIdx = dim or #t.shape
	if dimIdx < 1 or dimIdx > #t.shape then
		error(string.format("softmax dim %d out of range", dimIdx), 2)
	end
	local nd = #t.shape
	local softDim = t.shape[dimIdx]
	local preShape = table.create(nd - 1)
	for d = 1, nd do
		if d ~= dimIdx then preShape[#preShape + 1] = t.shape[d] end
	end
	local preSize = sizeOf(preShape)
	local out = table.create(#t.data, 0)

	for pi = 1, preSize do
		local oc = coordsOf(preShape, pi - 1)
		local m = -math.huge
		for r = 1, softDim do
			local v = t.data[indexOf(t.shape, insertCoord(oc, dimIdx, r - 1, nd))]
			if v > m then m = v end
		end
		local sumE = 0
		for r = 1, softDim do
			local e = math_exp(t.data[indexOf(t.shape, insertCoord(oc, dimIdx, r - 1, nd))] - m)
			out[indexOf(t.shape, insertCoord(oc, dimIdx, r - 1, nd))] = e
			sumE += e
		end
		local inv = sumE > 0 and 1 / sumE or 0
		for r = 1, softDim do
			local idx = indexOf(t.shape, insertCoord(oc, dimIdx, r - 1, nd))
			out[idx] *= inv
		end
	end

	local res = makeRawTensor(copyData(t.shape), out, t.requiresGrad, { t }, "softmax")
	if t.requiresGrad then
		res._backward = function()
			local grad = Tensor.ensureGrad(res)
			local ga = table.create(#t.data, 0)
			for pi = 1, preSize do
				local oc = coordsOf(preShape, pi - 1)
				-- gather p and upstream grad for this softmax row
				local rowP = table.create(softDim)
				local rowG = table.create(softDim)
				for r = 1, softDim do
					local idx = indexOf(t.shape, insertCoord(oc, dimIdx, r - 1, nd))
					rowP[r] = out[idx]
					rowG[r] = grad[idx]
				end
				-- dL/dP_j * P_i*(delta_ij - P_j)
				local dot = 0
				for j = 1, softDim do dot += rowG[j] * rowP[j] end
				for i = 1, softDim do
					local idx = indexOf(t.shape, insertCoord(oc, dimIdx, i - 1, nd))
					ga[idx] += rowP[i] * (rowG[i] - dot)
				end
			end
			accumulateGrad(t, ga)
		end
	end
	return res
end

--- Cat ----------------------------------------------------------------------

-- Concatenate tensors along a dim (default last). Must match sizes on all
-- non-cat dims.
function Tensor.cat(ts: {Tensor}, dim: number?): Tensor
	local dimIdx = dim or (#ts[1].shape)
	if #ts == 0 then
		error("cat needs at least one tensor", 2)
	end
	local nd = #ts[1].shape
	if dimIdx < 1 or dimIdx > nd then
		error(string.format("cat dim %d out of range", dimIdx), 2)
	end
	for ti = 2, #ts do
		if #ts[ti].shape ~= nd then
			error(string.format("cat: tensor %d has different rank (%s vs %s)",
				ti, table.concat(ts[ti].shape, "x"), table.concat(ts[1].shape, "x")), 2)
		end
		for d = 1, nd do
			if d ~= dimIdx and ts[ti].shape[d] ~= ts[1].shape[d] then
				error(string.format("cat: dim %d mismatch on tensor %d (%s vs %s)",
					d, ti, table.concat(ts[ti].shape, "x"), table.concat(ts[1].shape, "x")), 2)
			end
		end
	end

	local outShape = copyData(ts[1].shape)
	local catTotal = 0
	for ti = 1, #ts do catTotal += ts[ti].shape[dimIdx] end
	outShape[dimIdx] = catTotal
	local outSize = sizeOf(outShape)
	local out = table.create(outSize, 0)
	local needsGrad = false
	for ti = 1, #ts do
		if ts[ti].requiresGrad then needsGrad = true end
	end

	-- offsets of each tensor along the cat dim
	local catOffsets = table.create(#ts)
	local acc = 0
	for ti = 1, #ts do
		catOffsets[ti] = acc
		acc += ts[ti].shape[dimIdx]
	end

	local preShape = outShape
	local preShapeNo = table.create(nd - 1)
	for d = 1, nd do
		if d ~= dimIdx then preShapeNo[#preShapeNo + 1] = preShape[d] end
	end
	local preSize = sizeOf(preShapeNo)

	for pi = 1, preSize do
		local oc = coordsOf(preShapeNo, pi - 1)
		for r = 1, catTotal do
			local outIdx = indexOf(outShape, insertCoord(oc, dimIdx, r - 1, nd))
			-- which tensor?
			local ti = 1
			local localPos = r - 1
			while ti < #ts and localPos >= ts[ti].shape[dimIdx] do
				localPos -= ts[ti].shape[dimIdx]
				ti += 1
			end
			out[outIdx] = ts[ti].data[indexOf(ts[ti].shape, insertCoord(oc, dimIdx, localPos, nd))]
		end
	end

	local res = makeRawTensor(outShape, out, needsGrad, ts, "cat")
	if needsGrad then
		res._backward = function()
			local grad = Tensor.ensureGrad(res)
			local gsrc = table.create(#ts)
			for ti = 1, #ts do
				gsrc[ti] = table.create(#ts[ti].data, 0)
			end
			for pi = 1, preSize do
				local oc = coordsOf(preShapeNo, pi - 1)
				for r = 1, catTotal do
					local outIdx = indexOf(outShape, insertCoord(oc, dimIdx, r - 1, nd))
					local ti = 1
					local localPos = r - 1
					while ti < #ts and localPos >= ts[ti].shape[dimIdx] do
						localPos -= ts[ti].shape[dimIdx]
						ti += 1
					end
					gsrc[ti][indexOf(ts[ti].shape, insertCoord(oc, dimIdx, localPos, nd))] += grad[outIdx]
				end
			end
			for ti = 1, #ts do
				accumulateGrad(ts[ti], gsrc[ti])
			end
		end
	end
	return res
end

-- Elementwise greater-than: returns 1.0/0.0. No gradient (masks are not
-- differentiable inputs in the layers that use this).
function Tensor.gt(a: Tensor, b: Tensor): Tensor
	local outShape = broadcastShape(a.shape, b.shape)
	local outSize = sizeOf(outShape)
	local out = table.create(outSize)
	local as = #a.data
	local bs = #b.data
	local singleA = as == 1
	local singleB = bs == 1
	for i = 1, outSize do
		local ai = if singleA then 1 else mapIndex(i, a.shape, outShape)
		local bi = if singleB then 1 else mapIndex(i, b.shape, outShape)
		out[i] = if a.data[ai] > b.data[bi] then 1.0 else 0.0
	end
	return makeRawTensor(outShape, out, false, {}, "gt")
end

--- Slice / transpose(n-D) / stack ------------------------------------------------

-- Gather rows from a [N, D] table tensor given a flat list of 1-based row ids.
-- Output is [R, D]. Backward scatters the grad into the gathered rows.
function Tensor.gatherRows(tableTensor: Tensor, rows: {number}): Tensor
	local N, D = tableTensor.shape[1], tableTensor.shape[2]
	local R = #rows
	local out = table.create(R * D)
	local td = tableTensor.data
	for r = 1, R do
		local srcIdx = (rows[r] - 1) * D
		local dstIdx = (r - 1) * D
		for d = 1, D do
			out[dstIdx + d] = td[srcIdx + d]
		end
	end
	local res = makeRawTensor({ R, D }, out, tableTensor.requiresGrad, { tableTensor }, "gather")
	if tableTensor.requiresGrad then
		res._backward = function()
			local grad = Tensor.ensureGrad(res)
			local ga = table.create(N * D, 0)
			for r = 1, R do
				local srcIdx = (rows[r] - 1) * D
				local dstIdx = (r - 1) * D
				for d = 1, D do
					ga[srcIdx + d] += grad[dstIdx + d]
				end
			end
			accumulateGrad(tableTensor, ga)
		end
	end
	return res
end

-- Extract a single slice along a dim: result has that dim removed. Backward
-- routes the grad to exactly that slice's positions.
function Tensor.sliceDim(t: Tensor, dim: number, index: number): Tensor
	local nd = #t.shape
	if dim < 1 or dim > nd or index < 1 or index > t.shape[dim] then
		error(string.format("sliceDim: dim %d index %d invalid for [%s]",
			dim, index, table.concat(t.shape, "x")), 2)
	end
	local outShape = table.create(nd - 1)
	for d = 1, nd do
		if d ~= dim then outShape[#outShape + 1] = t.shape[d] end
	end
	local outSize = sizeOf(outShape)
	local out = table.create(outSize, 0)
	for oi = 1, outSize do
		local oc = coordsOf(outShape, oi - 1)
		out[oi] = t.data[indexOf(t.shape, insertCoord(oc, dim, index - 1, nd))]
	end
	local res = makeRawTensor(outShape, out, t.requiresGrad, { t }, "slice")
	if t.requiresGrad then
		res._backward = function()
			local grad = Tensor.ensureGrad(res)
			local ga = table.create(#t.data, 0)
			for oi = 1, outSize do
				local oc = coordsOf(outShape, oi - 1)
				ga[indexOf(t.shape, insertCoord(oc, dim, index - 1, nd))] += grad[oi]
			end
			accumulateGrad(t, ga)
		end
	end
	return res
end

function Tensor.selectLastDim(t: Tensor, index: number): Tensor
	return Tensor.sliceDim(t, #t.shape, index)
end

-- Slice a contiguous range along a dim: keeps the dim with size (stop-start+1).
-- Backward scatters the grad into the sliced range.
function Tensor.sliceRange(t: Tensor, dim: number, start: number, stop: number): Tensor
	local nd = #t.shape
	if dim < 1 or dim > nd or start < 1 or stop > t.shape[dim] or start > stop then
		error(string.format("sliceRange: dim %d range %d..%d invalid for [%s]",
			dim, start, stop, table.concat(t.shape, "x")), 2)
	end
	local outShape = copyData(t.shape)
	outShape[dim] = stop - start + 1
	local outSize = sizeOf(outShape)
	local out = table.create(outSize)
	local _dd = stop - start
	for oi = 1, outSize do
		local oc = coordsOf(outShape, oi - 1)
		oc[dim] += start - 1
		out[oi] = t.data[indexOf(t.shape, oc)]
	end
	local res = makeRawTensor(outShape, out, t.requiresGrad, { t }, "sliceRange")
	if t.requiresGrad then
		res._backward = function()
			local grad = Tensor.ensureGrad(res)
			local ga = table.create(#t.data, 0)
			for oi = 1, outSize do
				local oc = coordsOf(outShape, oi - 1)
				oc[dim] += start - 1
				ga[indexOf(t.shape, oc)] += grad[oi]
			end
			accumulateGrad(t, ga)
		end
	end
	return res
end

-- Gather one element along the last dim per leading position: out[n] = t[coords[n]..., idxs[n]].
-- Input t: [.., D]; idxs: length prod(shape[:-1]) (flat over leading dims, 1-based).
-- Output shape: t.shape[:-1]. Backward scatters grad to the picked positions.
function Tensor.takeLast(t: Tensor, idxs: { number }): Tensor
	local nd = #t.shape
	local outShapeLen = nd - 1
	local outShape = table.create(outShapeLen)
	local leading = 1
	for i = 1, outShapeLen do
		outShape[i] = t.shape[i]
		leading *= t.shape[i]
	end
	local _D = t.shape[nd]
	if #idxs ~= leading then
		error(string.format("takeLast: %d indices for %d leading positions", #idxs, leading), 2)
	end
	local outSize = leading
	local out = table.create(outSize)
	for oi = 1, outSize do
		local base = indexOf(t.shape, insertCoord(coordsOf(outShape, oi - 1), nd, idxs[oi] - 1, nd))
		out[oi] = t.data[base]
	end
	local res = makeRawTensor(outShape, out, t.requiresGrad, { t }, "takeLast")
	if t.requiresGrad then
		res._backward = function()
			local grad = Tensor.ensureGrad(res)
			local ga = table.create(#t.data, 0)
			for oi = 1, outSize do
				local base = indexOf(t.shape, insertCoord(coordsOf(outShape, oi - 1), nd, idxs[oi] - 1, nd))
				ga[base] += grad[oi]
			end
			accumulateGrad(t, ga)
		end
	end
	return res
end

-- General transpose: swap two dims. Backward transposes the gradient back.
function Tensor.transpose(t: Tensor, dim1: number, dim2: number): Tensor
	local nd = #t.shape
	if dim1 < 1 or dim1 > nd or dim2 < 1 or dim2 > nd then
		error("transpose: dim out of range", 2)
	end
	if dim1 == dim2 then return t end
	local outShape = copyData(t.shape)
	outShape[dim1], outShape[dim2] = outShape[dim2], outShape[dim1]
	local outSize = #t.data
	local out = table.create(outSize, 0)
	for oi = 1, outSize do
		local oc = coordsOf(outShape, oi - 1)
		local ic = copyData(oc)
		ic[dim1], ic[dim2] = ic[dim2], ic[dim1]
		out[oi] = t.data[indexOf(t.shape, ic)]
	end
	local res = makeRawTensor(outShape, out, t.requiresGrad, { t }, "transpose")
	if t.requiresGrad then
		res._backward = function()
			local grad = Tensor.ensureGrad(res)
			local ga = table.create(outSize, 0)
			for oi = 1, outSize do
				local oc = coordsOf(outShape, oi - 1)
				local ic = copyData(oc)
				ic[dim1], ic[dim2] = ic[dim2], ic[dim1]
				ga[indexOf(t.shape, ic)] += grad[oi]
			end
			accumulateGrad(t, ga)
		end
	end
	return res
end

-- Standardize alias: transpose2d uses the general path.
function Tensor.transpose2d(t: Tensor): Tensor
	return Tensor.transpose(t, 1, 2)
end

--- Engine ------------------------------------------------------------------

-- Reverse-mode auto-differentiation of a scalar loss.
-- Valid order: every node runs _backward only after all of its dependents in
-- the graph have already pushed their gradients into it. We build a post-order
-- over the "depends-on" DAG (leaves first) and then execute in reverse.
function Tensor.backward(root: Tensor, retainGraph: boolean?)
	if #root.shape ~= 0 then
		error("Tensor.backward() expects a 0-dim scalar loss", 2)
	end
	if not root.requiresGrad then
		return
	end

	-- Iterative post-order over parent edges. A node in a DAG can be pushed
	-- multiple times (e.g. ops whose _prev lists the same tensor twice, like
	-- mul(x, x), or a shared subexpression with several consumers). Each node
	-- must be emitted to `order` exactly once, at its finishing time, or its
	-- _backward would run more than once and over-accumulate gradients.
	local order: {Tensor} = {}
	local state: {[Tensor]: number} = {}
	local stack: {Tensor} = { root }
	while #stack > 0 do
		local node = stack[#stack]
		local s = state[node] or 0
		if s == 0 then
			state[node] = 1
			for _, p in node._prev do
				if not state[p] then
					table.insert(stack, p)
				end
			end
		elseif s == 1 then
			state[node] = 2
			table.remove(stack)
			table.insert(order, node)
		else
			state[node] = 2
			table.remove(stack)
		end
	end

	-- Zero nothing here: gradient ACCUMULATES into any pre-existing gradient
	-- buffers (that is what makes batched training "sum the gradients of every
	-- sample in the batch"). The caller is responsible for clearing buffers
	-- before each batch via Tensor.zeroGrad / the optimizer's zeroGrad, exactly
	-- like PyTorch. Intermediate op nodes are rebuilt on every forward, so their
	-- buffers start clean; persistent leaf tensors (params) accumulate across
	-- backward() calls until the caller zeroes them.

	-- seed and propagate in reverse (root first, leaves last)
	root.grad = { 1 }
	for i = #order, 1, -1 do
		local node = order[i]
		if node._backward then
			node._backward()
		end
	end
end

return Tensor