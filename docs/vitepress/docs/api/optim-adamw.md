# AdamW

Source: `optim/AdamW.luau`

AdamW (Loshchilov & Hutter, ICLR 2019; arXiv:1711.05101).  Decoupled weight decay. Bias-corrected m and v per step.

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, eps?, weightDecay?})
```

## API

- `step()`
- `zeroGrad()`
- `setLr(lr)`
- `getLr()`
- `stateDict()`
- `loadStateDict(state)`
- `paramCount()`

## Methods

#### `new(params: { any }, lr: number, opts: any?) -> any`

Create the optimizer over `params` (Module.collectParams(model)) at a base learning rate.

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

