# PRNG

Source: `core/PRNG.luau`

PRNG: seedable generator using only arithmetic available in the Luau CLI  (this build has NO bitwise operators), so we use two classic generators that  stay exact within 53-bit doubles:    * Park-Miller LCG (Schrage method) for integer draws and seeding.    * Wichmann-Hill combined LCG for uniform [0,1) floats.  Both are deterministic: a fixed seed reproduces the exact same stream,  which is what lets every RoNetV4.1 run be reproducible.

## Methods

### Class PRNG

#### `new(seed: number?) -> PRNGInstance`

Construct a seedable PRNG instance. Same seed =&gt; same stream (reproducible runs). Defaults to seed 1.

### Class self

#### `next()`

Next integer from the Park-Miller LCG, in [0, PM_MOD-2].

#### `nextFloat()`

Uniform float in [0, 1) (Wichmann-Hill combined LCG).

#### `nextInt(lo: any, hi: any)`

Uniform integer in [1, hi] or [lo, hi] (bounds normalized if swapped).

#### `nextGaussian()`

Standard-normal draw via Box-Muller (caches every second sample).

#### `seed(s: number)`

Re-seed the generator; the stream restarts deterministically.

#### `clone()`

Independent copy with an identical state (same next draws).

### Class PRNG

#### `stream(name: string, seed: number?) -> PRNGInstance`

Get the named component stream, creating it (seeded from its name hash or the given seed) on first use. Lets independent parts of the graph draw from fully disjoint, stable streams.

