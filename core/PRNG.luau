--!strict
-- PRNG: seedable generator using only arithmetic available in the Luau CLI
-- (this build has NO bitwise operators), so we use two classic generators that
-- stay exact within 53-bit doubles:
--   * Park-Miller LCG (Schrage method) for integer draws and seeding.
--   * Wichmann-Hill combined LCG for uniform [0,1) floats.
-- Both are deterministic: a fixed seed reproduces the exact same stream,
-- which is what lets every RoNetV4.1 run be reproducible.

local floor = math.floor

local PRNG = {}

export type State = {
	x: number,
	y: number,
	z: number,
	pm: number,
}

local PM_MOD = 2147483647
local PM_MULT = 48271
local PM_Q = 44488      -- PM_MOD div PM_MULT
local PM_R = 3399       -- PM_MOD mod PM_MULT

-- Next Park-Miller value via Schrage decomposition (exact in doubles).
local function pmNext(s: number): number
	local hi = floor(s / PM_Q)
	local lo = s % PM_Q
	local t = PM_MULT * lo - PM_R * hi
	if t < 0 then t = t + PM_MOD end
	return t
end

-- Advance a seed through the LCG `count` times.
local function pmSkip(s: number, count: number): number
	for _ = 1, count do s = pmNext(s) end
	return s
end

-- Seed the three Wichmann-Hill arms from a starting Park-Miller seed.
local function seedState(seed: number): State
	local s = seed % PM_MOD
	if s == 0 then s = 1 end
	s = pmSkip(s, 2)
	local x = s % 30268
	s = pmSkip(s, 3)
	local y = s % 30306
	s = pmSkip(s, 5)
	local z = s % 30322
	if x == 0 then x = 1 end
	if y == 0 then y = 1 end
	if z == 0 then z = 1 end
	return { x = x, y = y, z = z, pm = s }
end

export type PRNGInstance = {
	state: State,
	next: () -> number,
	nextFloat: () -> number,
	nextInt: (...any) -> number,
	nextGaussian: () -> number,
	seed: (seed: number) -> (),
	clone: () -> PRNGInstance,
}

-- Construct a seedable PRNG instance. Same seed => same stream (reproducible
-- runs). Defaults to seed 1.
function PRNG.new(seed: number?): PRNGInstance
	local self: any = setmetatable({ state = seedState(seed or 1) }, { __index = {} })

	local gaussCache: number? = nil

	-- Next integer from the Park-Miller LCG, in [0, PM_MOD-2].
	function self.next()
		local s = self.state
		s.pm = pmNext(s.pm)
		return s.pm - 1 -- [0, PM_MOD-2]
	end

	-- Uniform float in [0, 1) (Wichmann-Hill combined LCG).
	function self.nextFloat()
		local s = self.state
		s.x = (171 * s.x) % 30269
		s.y = (172 * s.y) % 30307
		s.z = (170 * s.z) % 30323
		local r = (s.x / 30269 + s.y / 30307 + s.z / 30323) % 1
		if r < 0 then r = r + 1 end
		return r
	end

	-- Uniform integer in [1, hi] or [lo, hi] (bounds normalized if swapped).
	function self.nextInt(lo: any, hi: any)
		if hi == nil then
			hi = lo
			lo = 1
		end
		local l = lo::number
		local h = hi::number
		if l > h then l, h = h, l end
		local span = h - l + 1
		return l + floor(self.nextFloat() * span)
	end

	-- Standard-normal draw via Box-Muller (caches every second sample).
	function self.nextGaussian()
		if gaussCache ~= nil then
			local v = gaussCache
			gaussCache = nil
			return v
		end
		local u1 = self.nextFloat()
		if u1 == 0 then u1 = 1e-12 end
		local u2 = self.nextFloat()
		local mag = math.sqrt(-2 * math.log(u1))
		gaussCache = mag * math.sin(2 * math.pi * u2)
		return mag * math.cos(2 * math.pi * u2)
	end

	-- Re-seed the generator; the stream restarts deterministically.
	function self.seed(s: number)
		self.state = seedState(s)
		gaussCache = nil
	end

	-- Independent copy with an identical state (same next draws).
	function self.clone()
		local c = PRNG.new(1)
		local os = self.state
		c.state = { x = os.x, y = os.y, z = os.z, pm = os.pm }
		return c
	end

	return (self :: PRNGInstance)
end

-- tiny FNV-1a-style string hash without bitwise (multiplicative folding).
-- Used to seed independent streams per component name.
local function hashSeed(name: string): number
	local h = 2166136261
	for i = 1, #name do
		local c = string.byte(name, i) + 1
		h = (h + c) * 2654435761 % 2147483647
	end
	if h == 0 then h = 1 end
	return h
end

local STREAMS: {[string]: PRNGInstance} = {}
PRNG._streams = STREAMS

-- Get the named component stream, creating it (seeded from its name hash or
-- the given seed) on first use. Lets independent parts of the graph draw from
-- fully disjoint, stable streams.
function PRNG.stream(name: string, seed: number?): PRNGInstance
	local s = STREAMS[name]
	if not s then
		s = PRNG.new(seed or hashSeed(name))
		STREAMS[name] = s
	end
	return s
end

export type RNG = typeof(PRNG.new())

return PRNG