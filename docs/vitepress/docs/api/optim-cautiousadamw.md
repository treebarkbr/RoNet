# CautiousAdamW

Source: `optim/CautiousAdamW.luau`

Cautious AdamW (Liu et al., 2024; arXiv:2411.16085). Elementwise mask on the  first moment: keep only components aligned with the raw gradient (mhat*g &gt; 0),  scale the rest down by alpha. update = alpha*masked_mhat/(sqrt(vhat)+eps).

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, alpha?, eps?, weightDecay?})
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

