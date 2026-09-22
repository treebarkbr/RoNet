# SOAP

Source: `optim/SOAP.luau`

SOAP (Vyas et al., 2014/2024; arXiv:2409.11321). Shampoo-style update with  Adam-in-eigenbasis. Eigenspaces recomputed from GG = grad grad' (Shampoo  preconditioners) every precondition_frequency steps; first step builds the  preconditioner and is skipped. 2D parameters are preconditioned; 1D fall back  to Adam. Requires deps.Matrix (eigh).         preconditionFrequency?, shampooBeta?, maxPrecondDim?, precondition1d?})

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {betas?, eps?, weightDecay?,
```

## Methods

#### `new(params: { any }, lr: number, opts: any?) -> any`

Create the optimizer over `params` (Module.collectParams(model)) at a base learning rate.

#### `_buildInitial(p: any, st: any)`

Lazily build Shampoo preconditioner state (gg0/gg1, moments, eigenspaces) for a 2D param.

#### `_updateGG(p: any, st: any)`

Update the Shampoo preconditioner accumulators (gg0/gg1) from the current grad.

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

