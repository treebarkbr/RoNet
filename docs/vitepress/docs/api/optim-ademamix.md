# AdEMAMix

Source: `optim/AdEMAMix.luau`

AdEMAMix (Pagliardini, Ablin & Grangier, 2024; arXiv:2409.03137).  Two gradient EMAs: m1 (fast, beta1) and m2 (slow, beta3), plus the second  moment v. The update mixes bias-corrected m1 with alpha * m2 (no bias  correction on m2, per the paper). Optional warmup schedulers over tFinal  steps, faithful to Section 3 / Algorithm 1 with T_alpha = T_beta3 = tFinal:    alpha(t) = min(t * alpha / tFinal, alpha)    beta3(t) = min(exp(ln(betaStart) * ln(beta3) /                ((1 - t/tFinal) * ln(beta3) + (t/tFinal) * ln(betaStart))), beta3)  with betaStart defaulting to beta1; keep tFinal = 0 to disable scheduling.

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, beta3?, alpha?, tFinal?, betaStart?, eps?, weightDecay?})
```

## Methods

#### `new(params: { any }, lr: number, opts: any?) -> any`

Create the optimizer over `params` (Module.collectParams(model)) at a base learning rate.

#### `alphaAt(: any, t: number) -> number`

alpha(t): linear warmup from 0 to the final alpha over tFinal steps.

#### `beta3(: any, t: number) -> number`

beta3(t): schedules the EMA half-life linearly (paper App. A.1), from betaStart at t=0 up to beta3 at t=tFinal.

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

