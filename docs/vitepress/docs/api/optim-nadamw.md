# NAdamW

Source: `optim/NAdamW.luau`

NAdamW: AdamW with Nesterov-accelerated first moment (Dozat, 2016).  mhat_nest = b1*mhat_t + (1-b1)*g/(1-b1^t); update = mhat_nest/(sqrt(vhat)+eps).

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, eps?, weightDecay?})
```

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

