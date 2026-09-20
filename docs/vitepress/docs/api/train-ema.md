# EMA

Source: `train/EMA.luau`

Exponential moving average of parameter snapshots (Polyak averaging).    EMA.new(params, decay, opts:{warmupSteps?})    ema:update()    blend buffers toward current params (with warmup-scaled decay)    ema:copyBack()  overwrite model params with the averaged buffers    ema:stateDict() / ema:loadStateDict()   (for checkpointing)  Written for speed: single numeric pass per param, no allocations per update.

## Methods

#### `new(params: { any }, decay: number, opts: any?) -> any`

#### `_decay(: any) -> number`

Effective decay: ramps from ~1 towards maxDecay during warmup, so early snapshots are averaged broadly (hana-rhea recipe: min(decay, (1+t)/(t+2))).

#### `update()`

#### `copyBack()`

#### `stateDict() -> any`

#### `loadStateDict(st: any)`

