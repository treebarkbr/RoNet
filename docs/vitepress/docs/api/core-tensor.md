# Tensor

Source: `core/Tensor.luau`

Tensor: an N-dimensional tensor on a flat row-major array with reverse-mode  autograd (define-by-run). This is what makes RoNetV4.1 capable of any  network: write a forward pass with tensor ops and the recorded graph knows  how to differentiate everything.  Conventions:    * data is a flat {number}, row-major (last dim contiguous).    * shape {} denotes a scalar (size 1).    * Every op that produces a tensor with requiresGrad=true records a      _backward closure that flows .grad into its parents during backward().  Part 1: constructors, plumbing, pointwise ops, matmul.  IMPORTANT (CLI limitation): modules must not require() siblings; only the  entry script can. Tensor is therefore self-contained. PRNG is injected  through Tensor.setRNG() from the root loader so runs stay reproducible.

## Methods

#### `setRNG(rng: RNG)`

#### `fromData(data: {number}, shape: {number}, requiresGrad: boolean?) -> Tensor`

#### `zeros(shape: {number}, requiresGrad: boolean?) -> Tensor`

#### `ones(shape: {number}, requiresGrad: boolean?) -> Tensor`

#### `full(shape: {number}, value: number, requiresGrad: boolean?) -> Tensor`

#### `scalar(value: number, requiresGrad: boolean?) -> Tensor`

#### `randn(shape: {number}, rng: RNG?, requiresGrad: boolean?) -> Tensor`

#### `rand(shape: {number}, rng: RNG?, requiresGrad: boolean?) -> Tensor`

#### `uniform(shape: {number}, lo: number, hi: number, rng: RNG?, requiresGrad: boolean?) -> Tensor`

#### `numel(t: Tensor) -> number`

#### `shapeOf(t: Tensor) -> {number}`

#### `isScalar(t: Tensor) -> boolean`

#### `item(t: Tensor) -> number`

#### `clone(t: Tensor) -> Tensor`

Copies data, detaches from the graph.

#### `detach(t: Tensor) -> Tensor`

A view sharing data with requiresGrad=false.

#### `ensureGrad(t: Tensor) -> {number}`

#### `zeroGrad(t: Tensor)`

#### `gradData(t: Tensor) -> {number}?`

#### `add(a: Tensor, b: Tensor) -> Tensor`

#### `sub(a: Tensor, b: Tensor) -> Tensor`

#### `mul(a: Tensor, b: Tensor) -> Tensor`

#### `div(a: Tensor, b: Tensor) -> Tensor`

#### `mulScalar(a: Tensor, s: number) -> Tensor`

#### `addScalar(a: Tensor, s: number) -> Tensor`

#### `neg(a: Tensor) -> Tensor`

#### `pow(a: Tensor, p: number) -> Tensor`

#### `exp(a: Tensor) -> Tensor`

#### `log(a: Tensor) -> Tensor`

#### `sqrt(a: Tensor) -> Tensor`

#### `tanh(a: Tensor) -> Tensor`

#### `sigmoid(a: Tensor) -> Tensor`

#### `relu(a: Tensor) -> Tensor`

#### `silu(a: Tensor) -> Tensor`

Swish / SiLU: x * sigmoid(x)

#### `gelu(a: Tensor) -> Tensor`

GELU (tanh approximation, as in GPT-2/J and most libs).

#### `clip(a: Tensor, lo: number, hi: number) -> Tensor`

#### `matmul(a: Tensor, b: Tensor) -> Tensor`

C[i,j] = sum_k A[i,k] * B[k,j]  (2D only; use bmm for batches).

#### `bmm(a: Tensor, b: Tensor) -> Tensor`

Batched matmul: [B, M, K] @ [B, K, N] -&gt; [B, M, N].

#### `reshape(t: Tensor, newShape: {number}) -> Tensor`

#### `flatten(t: Tensor, rows: number, cols: number) -> Tensor`

Flatten to [rows, cols]; either may be -1 (inferred).

#### `sum(t: Tensor, dim: number?) -> Tensor`

Sum reduction: -1 dim means full reduction to a scalar. Backward routes the scalar upstream grad to every element (for full reduce) or to the matching column (for a specific dim).

#### `mean(t: Tensor, dim: number?) -> Tensor`

Mean reduction.

#### `max(t: Tensor, dim: number?) -> Tensor`

Max reduction along a single dim (or full max). Records the argmax so the backward can route the upstream grad to exactly the winner positions.

#### `sumSquares(t: Tensor) -> Tensor`

Squared L2 norm, used by vector-like regularization / normalization helpers.

#### `softmax(t: Tensor, dim: number?) -> Tensor`

Softmax along a single dim (default: last) - used on [B,V] logits and the on output rows of a [T,D] sequence. Backward uses the standard dPi/dXj = Pi*(delta_ij - Pj).

#### `cat(ts: {Tensor}, dim: number?) -> Tensor`

Concatenate tensors along a dim (default last). Must match sizes on all non-cat dims.

#### `gt(a: Tensor, b: Tensor) -> Tensor`

Elementwise greater-than: returns 1.0/0.0. No gradient (masks are not differentiable inputs in the layers that use this).

#### `gatherRows(tableTensor: Tensor, rows: {number}) -> Tensor`

Gather rows from a [N, D] table tensor given a flat list of 1-based row ids. Output is [R, D]. Backward scatters the grad into the gathered rows.

#### `sliceDim(t: Tensor, dim: number, index: number) -> Tensor`

Extract a single slice along a dim: result has that dim removed. Backward routes the grad to exactly that slice's positions.

#### `selectLastDim(t: Tensor, index: number) -> Tensor`

#### `sliceRange(t: Tensor, dim: number, start: number, stop: number) -> Tensor`

Slice a contiguous range along a dim: keeps the dim with size (stop-start+1). Backward scatters the grad into the sliced range.

#### `takeLast(t: Tensor, idxs: { number }) -> Tensor`

Gather one element along the last dim per leading position: out[n] = t[coords[n]..., idxs[n]]. Input t: [.., D]; idxs: length prod(shape[:-1]) (flat over leading dims, 1-based). Output shape: t.shape[:-1]. Backward scatters grad to the picked positions.

#### `transpose(t: Tensor, dim1: number, dim2: number) -> Tensor`

General transpose: swap two dims. Backward transposes the gradient back.

#### `transpose2d(t: Tensor) -> Tensor`

Standardize alias: transpose2d uses the general path.

#### `backward(root: Tensor, retainGraph: boolean?)`

Reverse-mode auto-differentiation of a scalar loss. Valid order: every node runs _backward only after all of its dependents in the graph have already pushed their gradients into it. We build a post-order over the "depends-on" DAG (leaves first) and then execute in reverse.

