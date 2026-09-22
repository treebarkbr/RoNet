# Muon

Source: `optim/Muon.luau`

Muon (Jordan & Keller, Dec 2024; Moonlight: arXiv:2502.16982). SGD-momentum  post-processed by Newton-Schulz orthogonalization, scaled by  max(1, rows/cols)^0.5. Use ONLY for 2D hidden matrices; feed biases, scalar  gains, embeddings and the LM head to a companion AdamW (see MuonAdamW).  Requires deps.Matrix (newtonSchulz).

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {momentum?, weightDecay?, nsSteps?, nesterov?})
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

