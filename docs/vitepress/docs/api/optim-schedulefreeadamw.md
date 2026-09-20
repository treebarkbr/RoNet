# ScheduleFreeAdamW

Source: `optim/ScheduleFreeAdamW.luau`

Schedule-Free AdamW (Defazio & Mishchenko, 2024; arXiv:2405.15682).  No learning-rate schedule needed: the iterate is blended toward the exponential  average z, so lr can stay constant. Must call train()/eval() around phases.  Faithful to the official facebookresearch/schedule_free implementation.         warmupSteps?, r?, weightLrPower?, innerMomentum?})

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, eps?, weightDecay?,
```

## Methods

#### `new(params: { any }, lr: number, opts: any?) -> any`

#### `train()`

#### `eval()`

#### `step()`

#### `zeroGrad()`

#### `setLr(lr: number)`

#### `getLr() -> number`

#### `paramCount() -> number`

#### `stateDict() -> any`

#### `loadStateDict(st: any)`

