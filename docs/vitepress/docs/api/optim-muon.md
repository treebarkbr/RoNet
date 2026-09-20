# Muon

Source: `optim/Muon.luau`

Muon (Jordan & Keller, Dec 2024; Moonlight: arXiv:2502.16982). SGD-momentum  post-processed by Newton-Schulz orthogonalization, scaled by  max(1, rows/cols)^0.5. Use ONLY for 2D hidden matrices; feed biases, scalar  gains, embeddings and the LM head to a companion AdamW (see MuonAdamW).  Requires deps.Matrix (newtonSchulz).

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {momentum?, weightDecay?, nsSteps?, nesterov?})
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

