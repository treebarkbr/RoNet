# Trainer

Source: `train/Trainer.luau`

Tiny training loop engine. Batches are built as real leading-dim tensors so a  whole minibatch is ONE forward, ONE backward, ONE optimizer step (single  autograd graph — the efficient path now that backward() accumulates per-call).    t = Trainer.new({ model, optimizer, schedule?, lossFn, data, epochs,                      batchSize?, evalFn?, evalEvery?, onEpochEnd? })    t:fit() -&gt; { lossHistory: {number}, evals: {number} }  lossFn(model, xs, ys) -&gt; 0-dim scalar Tensor ready for backward()    - xs: batched input tensor, ys: targets array {number}  Trainer.batchTensor(rows, featureShape) builds the input tensor.

## Methods

#### `batchTensor(rows: { { number } }, featureShape: { number }) -> any`

Build a single tensor from rows of equal-length numeric arrays.

#### `new(cfg: any) -> any`

#### `fit() -> any`

#### `evaluate(batchSize: number?) -> (number, number?)`

Run a gradient-free evaluation: total loss over data + evalFn value.

