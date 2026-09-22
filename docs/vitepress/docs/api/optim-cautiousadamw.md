# CautiousAdamW

Source: `optim/CautiousAdamW.luau`

Cautious AdamW (Liu et al., 2024; arXiv:2411.16085). Elementwise mask on the  first moment: keep only components aligned with the raw gradient (mhat*g &gt; 0),  scale the rest down by alpha. update = alpha*masked_mhat/(sqrt(vhat)+eps).

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, alpha?, eps?, weightDecay?})
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

