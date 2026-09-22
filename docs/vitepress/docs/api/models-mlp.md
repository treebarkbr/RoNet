# MLP

Source: `models/MLP.luau`

Simple MLP for regression/classification tests and small models.

## Methods

#### `new(name: string, sizes: { number }, rng: any, opts: any?)`

Build an MLP with sizes = list of per-layer widths (no. of params: sizes), e.g. {2, 16, 1}. opts.act is "relu" (default), "tanh", "silu", "gelu" or "sigmoid" and applies between every pair of dense layers.

#### `forward(: any, x: any) -> any`

Run the input through the layer chain in order.

#### `setTrain(: any, train: boolean)`

Flip train/eval mode on all sub-layers that track `training` (dropout).

