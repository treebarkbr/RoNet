# Norm

Source: `nn/Norm.luau`

Normalization layers. All operate over the LAST dim.  RMSNorm (Zhang & Sennrich 2019) and LayerNorm.

## Methods

#### `new(name: string, ndim: number, rng: any, op: string?, eps: number?)`

op: "rms" | "layer"; eps default 1e-6 (rms) / 1e-5 (layer).

#### `forward(: any, x: any) -> any`

Normalize x over the last dim (RMSNorm by default; LayerNorm if op is "layer").

