# MuonAdamW

Source: `optim/MuonAdamW.luau`

Composite optimizer following the modded-nanogpt / Moonlight production  recipe: 2D hidden matrices are optimized by Muon (Newton-Schulz removed  gradient post-processing) and every other parameter (scalars, biases,  embeddings, LM head) by AdamW at its own lr. Two-optimizer step keeps  gradients valid for both.    {muon: {lr?, momentum?, weightDecay?, nsSteps?, nesterov?},     adam: {lr?, beta1?, beta2?, eps?, weightDecay?}})

## Constructor

```luau
Class: new(muonParams: {Tensor}, adamParams: {Tensor}, opts:
```

## Methods

#### `new(muonParams: { any }, adamParams: { any }, opts: any?) -> any`

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

