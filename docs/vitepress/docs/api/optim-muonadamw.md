# MuonAdamW

Source: `optim/MuonAdamW.luau`

Composite optimizer following the modded-nanogpt / Moonlight production  recipe: 2D hidden matrices are optimized by Muon (Newton-Schulz removed  gradient post-processing) and every other parameter (scalars, biases,  embeddings, LM head) by AdamW at its own lr. Two-optimizer step keeps  gradients valid for both.    {muon: {lr?, momentum?, weightDecay?, nsSteps?, nesterov?},     adam: {lr?, beta1?, beta2?, eps?, weightDecay?}})

## Constructor

```luau
Class: new(muonParams: {Tensor}, adamParams: {Tensor}, opts:
```

## Methods

#### `new(muonParams: { any }, adamParams: { any }, opts: any?) -> any`

#### `step()`

#### `zeroGrad()`

#### `setLr(lr: number)`

#### `getLr() -> number`

#### `paramCount() -> number`

#### `stateDict() -> any`

#### `loadStateDict(st: any)`

