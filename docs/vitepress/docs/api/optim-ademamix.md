# AdEMAMix

Source: `optim/AdEMAMix.luau`

AdEMAMix (Pagliardini et al., 2024; arXiv:2409.03137).  Three moments: m1 (fast), m2 (slow, squared), m3 (very slow, same LR-scaled  by alpha). Optional beta3 temporal ramp toward a final beta3 over t_final  steps: beta3(t) = 1 - 1/(1 + alpha_s * t/t_final) with the paper's defaults  T_final = 10*k steps, s = 20 (alpha_s = s/log(t));

## Constructor

```luau
Class: new(params: {Tensor}, lr: number, opts: {beta1?, beta2?, beta3?, alpha?, tFinal?, alphaS?, eps?, weightDecay?})
```

## Methods

#### `new(params: { any }, lr: number, opts: any?) -> any`

#### `beta3(t: number) -> number`

#### `step()`

#### `zeroGrad()`

#### `setLr(lr: number)`

#### `getLr() -> number`

#### `paramCount() -> number`

#### `stateDict() -> any`

#### `loadStateDict(st: any)`

