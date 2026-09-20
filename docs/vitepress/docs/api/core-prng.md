# PRNG

Source: `core/PRNG.luau`

PRNG: seedable generator using only arithmetic available in the Luau CLI  (this build has NO bitwise operators), so we use two classic generators that  stay exact within 53-bit doubles:    * Park-Miller LCG (Schrage method) for integer draws and seeding.    * Wichmann-Hill combined LCG for uniform [0,1) floats.  Both are deterministic: a fixed seed reproduces the exact same stream,  which is what lets every RoNetV4.1 run be reproducible.

## Methods

### Class PRNG

#### `new(seed: number?) -> PRNGInstance`

### Class self

#### `next()`

#### `nextFloat()`

#### `nextInt(lo: any, hi: any)`

#### `nextGaussian()`

#### `seed(s: number)`

#### `clone()`

### Class PRNG

#### `stream(name: string, seed: number?) -> PRNGInstance`

