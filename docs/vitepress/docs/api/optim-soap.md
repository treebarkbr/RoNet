# SOAP

Source: `optim/SOAP.luau`

SOAP (Vyas et al., 2014/2024; arXiv:2409.11321). Shampoo-style update with  Adam-in-eigenbasis. Eigenspaces recomputed from GG = grad grad' (Shampoo  preconditioners) every precondition_frequency steps; first step builds the  preconditioner and is skipped. 2D parameters are preconditioned; 1D fall back  to Adam. Requires deps.Matrix (eigh).         preconditionFrequency?, shampooBeta?, maxPrecondDim?, precondition1d?})

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {betas?, eps?, weightDecay?,
```

## Methods

#### `new(params: { any }, lr: number, opts: any?) -> any`

#### `_buildInitial(p: any, st: any)`

#### `_updateGG(p: any, st: any)`

#### `step()`

#### `zeroGrad()`

#### `setLr(lr: number)`

#### `getLr() -> number`

#### `paramCount() -> number`

#### `stateDict() -> any`

#### `loadStateDict(st: any)`

