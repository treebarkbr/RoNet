# ScheduleFreeAdamW

Source: `optim/ScheduleFreeAdamW.luau`

Schedule-Free AdamW (Defazio & Mishchenko, 2024; arXiv:2405.15682).  No learning-rate schedule needed: the iterate is blended toward the exponential  average z, so lr can stay constant. Must call train()/eval() around phases.  Faithful to the official facebookresearch/schedule_free implementation.         warmupSteps?, r?, weightLrPower?, innerMomentum?})

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, eps?, weightDecay?,
```

## Methods

#### `new(params: { any }, lr: number, opts: any?) -> any`

Create the optimizer over `params` (Module.collectParams(model)) at a base learning rate.

#### `train()`

Switch to train mode: blend the iterate toward the param values (z) before stepping.

#### `eval()`

Switch to eval mode: blend the iterate toward the averaged z so evaluation uses the averaged weights.

#### `step()`

Apply one update from the current parameter gradients (call Tensor.backward() first).

#### `zeroGrad()`

Zero every parameter gradient buffer.

#### `setLr(lr: number)`

Set the current learning rate.

#### `getLr() -> number`

Return the current learning rate.

#### `paramCount() -> number`

Number of tracked parameters.

#### `stateDict() -> any`

Export optimizer state (moments, step counter) for checkpointing.

#### `loadStateDict(st: any)`

Restore optimizer state from a stateDict.

