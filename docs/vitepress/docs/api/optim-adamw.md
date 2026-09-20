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

#### `step()`

#### `zeroGrad()`

#### `setLr(lr: number)`

#### `getLr() -> number`

#### `paramCount() -> number`

#### `stateDict() -> any`

#### `loadStateDict(st: any)`

