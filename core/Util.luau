--!strict
-- Util: small numeric + table helpers shared by RoNetV4.1.
-- Self-contained (no requires) so the hot core stays dependency-free.

local math_floor = math.floor
local math_max = math.max
local math_min = math.min

local Util = {}

local TYPES: {[string]: string} = {
	number = "number",
	string = "string",
	boolean = "boolean",
}

function Util.typeName(v: any): string
	local t = type(v)
	return TYPES[t] and t or t
end

function Util.lerp(a: number, b: number, t: number): number
	return a + (b - a) * t
end

-- Clamp value into [lo, hi].
function Util.clamp(v: number, lo: number, hi: number): number
	return math_min(math_max(v, lo), hi)
end

-- Swap two elements of an array.
function Util.swap(arr: {any}, i: number, j: number)
	local tmp = arr[i]
	arr[i] = arr[j]
	arr[j] = tmp
end

-- Create an array of `n` copies of `value` without iterator overhead.
function Util.fill<T>(n: number, value: T): {T}
	local out = table.create(n)
	for i = 1, n do out[i] = value end
	return out
end

function Util.zeros(n: number): {number}
	return Util.fill(n, 0)
end

function Util.ones(n: number): {number}
	return Util.fill(n, 1)
end

function Util.copyArray<T>(source: {T}): {T}
	local n = #source
	local out = table.create(n)
	table.move(source, 1, n, 1, out)
	return out
end

function Util.range(n: number): {number}
	local out = table.create(n)
	for i = 1, n do out[i] = i end
	return out
end

-- index of maximum element
function Util.argmax(arr: {number}): number
	local n = #arr
	local best = 1
	local bestV = arr[1]
	for i = 2, n do
		local v = arr[i]
		if v > bestV then
			bestV = v
			best = i
		end
	end
	return best
end

function Util.argmin(arr: {number}): number
	local n = #arr
	local best = 1
	local bestV = arr[1]
	for i = 2, n do
		local v = arr[i]
		if v < bestV then
			bestV = v
			best = i
		end
	end
	return best
end

function Util.sum(arr: {number}): number
	local s = 0
	for i = 1, #arr do s += arr[i] end
	return s
end

function Util.mean(arr: {number}): number
	local n = #arr
	if n == 0 then return 0 end
	return Util.sum(arr) / n
end

function Util.variance(arr: {number}): number
	local n = #arr
	if n <= 1 then return 0 end
	local m = Util.mean(arr)
	local s = 0
	for i = 1, n do
		local d = arr[i] - m
		s += d * d
	end
	return s / n
end

function Util.softmax(logits: {number}, out: {number}?): {number}
	local n = #logits
	local dest = out or table.create(n)
	local m = -math.huge
	for i = 1, n do
		if logits[i] > m then m = logits[i] end
	end
	local sum = 0
	for i = 1, n do
		local e = math.exp(logits[i] - m)
		dest[i] = e
		sum += e
	end
	local inv = sum > 0 and 1 / sum or 0
	for i = 1, n do dest[i] *= inv end
	return dest
end

-- One-hot vector of size `size` with a 1 at index `index`.
function Util.onehot(index: number, size: number): {number}
	local out = table.create(size, 0)
	out[index] = 1
	return out
end

-- Stable sigmoid
function Util.sigmoid(x: number): number
	if x >= 0 then
		local z = math.exp(-x)
		return 1 / (1 + z)
	else
		local z = math.exp(x)
		return z / (1 + z)
	end
end

function Util.tanhApprox(x: number): number
	local e2 = math.exp(-2 * x)
	return (1 - e2) / (1 + e2)
end

-- Pseudo-random int in [1, hi] or [lo, hi] (delegates to global PRNG).
function Util.randInt(...: any): number
	return math.random(...)
end

-- Total number of elements implied by a shape.
function Util.shapeSize(shape: {number}): number
	local n = 1
	for i = 1, #shape do n *= shape[i] end
	return n
end

-- Are two shapes equal?
function Util.shapesEqual(a: {number}, b: {number}): boolean
	if #a ~= #b then return false end
	for i = 1, #a do
		if a[i] ~= b[i] then return false end
	end
	return true
end

-- Pretty-print a shape as "2x3x4".
function Util.shapeString(shape: {number}): string
	local parts = table.create(#shape)
	for i = 1, #shape do parts[i] = tostring(shape[i]) end
	return table.concat(parts, "x")
end

-- Sum a list of numbers into dest (used to accumulate tensors).
function Util.addInplace(dest: {number}, source: {number})
	for i = 1, #dest do dest[i] += source[i] end
end

-- In-place scalar multiply.
function Util.scaleInplace(arr: {number}, factor: number)
	for i = 1, #arr do arr[i] *= factor end
end

function Util.copyInplace(dest: {number}, source: {number})
	table.move(source, 1, #source, 1, dest)
	return dest
end

function Util.concatTables<T>(a: {T}, b: {T}): {T}
	local out = table.create(#a + #b)
	table.move(a, 1, #a, 1, out)
	table.move(b, 1, #b, #a + 1, out)
	return out
end

function Util.roundToMultiple(x: number, mult: number): number
	return math_floor((x + mult / 2) / mult) * mult
end

-- Stable softmax over columns of a [T, D] row-major flat array, in place.
function Util.softmaxRows(data: {number}, t: number, d: number)
	local m = -math.huge
	for j = 1, d do
		local v = data[t * d - d + j]
		if v > m then m = v end
	end
	local sum = 0
	for j = 1, d do
		local e = math.exp(data[t * d - d + j] - m)
		data[t * d - d + j] = e
		sum += e
	end
	local inv = sum > 0 and 1 / sum or 0
	for j = 1, d do data[t * d - d + j] *= inv end
end

-- Lightweight matrix norm helpers (L2 over flat array).
function Util.l2Norm(data: {number}): number
	local s = 0
	for i = 1, #data do s += data[i] * data[i] end
	return math.sqrt(s)
end

function Util.l2NormSq(data: {number}): number
	local s = 0
	for i = 1, #data do s += data[i] * data[i] end
	return s
end

-- In-place global-norm rescale: if target > clip, scale all by clip/target.
-- Returns the multiplicative factor applied.
function Util.clipGradNorm(data: {number}, clip: number, eps: number?): number
	local e = eps or 1e-6
	local norm = math.sqrt(Util.l2NormSq(data))
	if norm > clip then
		local factor = clip / (norm + e)
		Util.scaleInplace(data, factor)
		return factor
	end
	return 1
end

export type TensorView = {
	shape: {number},
	strides: {number},
	storage: {number},
	offset: number,
}

return Util