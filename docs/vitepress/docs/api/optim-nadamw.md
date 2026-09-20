# NAdamW

Source: `optim/NAdamW.luau`

NAdamW: AdamW with Nesterov-accelerated first moment (Dozat, 2016).  mhat_nest = b1*mhat_t + (1-b1)*g/(1-b1^t); update = mhat_nest/(sqrt(vhat)+eps).

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, eps?, weightDecay?})
```

## Methods

#### `new(params: { any }, lr: number, opts: any?) -> any`

#### `step()`

#### `zeroGrad()`

#### `setLr(lr: number)`

#### `getLr() -> number`

#### `paramCount() -> number`

#### `stateDict() -> any`

#### `loadStateDict(st: any)`

