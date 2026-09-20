# Lion

Source: `optim/Lion.luau`

Lion (Chen et al., 2023; arXiv:2302.06675). Memory-efficient: sign update via  interleaved momentum. update = sign(b1*m + (1-b1)*g).

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, weightDecay?})
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

