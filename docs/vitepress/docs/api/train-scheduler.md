# Scheduler

Source: `train/Scheduler.luau`

Learning-rate schedules. Each schedule is a pure function  `(step: number) -&gt; factor` returning the multiplier on the base lr for that  step (step counts optimizer steps, starting at 1). Factory helpers:    Scheduler.constant()    Scheduler.warmupLinear(warmup, total, minFactor?)    Scheduler.warmupCosine(warmup, total, minFactor?)    Scheduler.linear(total, minFactor?)            (decay from 1 to minFactor)    Scheduler.warmupLinearCosine(cooldown, warmup, total, minFactor?)    Scheduler.forOptimizer(optimizer, schedule)    (held schedule, updates lr)  Written for speed: pure arithmetic, no per-step allocations beyond one number.

## Methods

### Class Scheduler

#### `constant() -> any`

Constant schedule: always returns 1.

#### `warmupLinear(warmup: number, total: number, minFactor: number?) -> any`

Linear warmup for [1, warmup], then linear decay 1 -&gt; minFactor over [warmup, total], clipped at minFactor (cooldown = total - warmup).

#### `warmupCosine(warmup: number, total: number, minFactor: number?) -> any`

Linear warmup then cosine decay with a floor of minFactor.

#### `cosine(total: number, minFactor: number?) -> any`

Pure cosine decay from 1 to minFactor over total steps (no warmup).

#### `invSqrt(warmup: number) -> any`

Inverse-square-root decay after optional warmup (transformer-style). factor = sqrt(warmup) / sqrt(step); = 1 at the warmup end, then decays.

#### `forOptimizer(opt: any, schedule: any) -> any`

Attach a schedule to an optimizer: returns { step(), setBase(lr), getLr() }. step() applies schedule(stepIdx) to the base lr (captured at attach time, refreshable via setBase). Scheduled lr available as handle.lr.

### Class h

#### `setBase(: any, lr: number)`

Refresh the base lr the schedule scales.

#### `step(: any) -> number`

Advance one optimizer step: applies schedule(t) to the base lr.

