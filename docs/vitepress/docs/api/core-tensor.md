# Tensor

Source: `core/Tensor.luau`

Tensor: an N-dimensional tensor on a flat row-major array with reverse-mode  autograd (define-by-run). This is what makes RoNetV4.1 capable of any  network: write a forward pass with tensor ops and the recorded graph knows  how to differentiate everything.  Conventions:    * data is a flat {number}, row-major (last dim contiguous).    * shape {} denotes a scalar (size 1).    * Every op that produces a tensor with requiresGrad=true records a      _backward closure that flows .grad into its parents during backward().  Part 1: constructors, plumbing, pointwise ops, matmul.  IMPORTANT (CLI limitation): modules must not require() siblings; only the  entry script can. Tensor is therefore self-contained. PRNG is injected  through Tensor.setRNG() from the root loader so runs stay reproducible.

## Methods

#### `setRNG(rng: RNG)`

Install the global RNG used by the randomizers (randn / rand / uniform) when no explicit RNG is passed. The root loader injects the PRNG here so every run stays reproducible.

#### `fromData(data: {number}, shape: {number}, requiresGrad: boolean?) -> Tensor`

Create a tensor from a flat row-major data array and a shape. Errors if the element count does not match the shape. requiresGrad defaults to false.

#### `zeros(shape: {number}, requiresGrad: boolean?) -> Tensor`

Tensor filled with zeros of the given shape.

#### `ones(shape: {number}, requiresGrad: boolean?) -> Tensor`

Tensor filled with ones of the given shape.

#### `full(shape: {number}, value: number, requiresGrad: boolean?) -> Tensor`

Tensor filled with a constant value.

#### `scalar(value: number, requiresGrad: boolean?) -> Tensor`

0-dim tensor (shape {}) holding a single value.

#### `randn(shape: {number}, rng: RNG?, requiresGrad: boolean?) -> Tensor`

Random tensor with standard-normal entries, drawn from the given RNG or the global default.

#### `rand(shape: {number}, rng: RNG?, requiresGrad: boolean?) -> Tensor`

Random tensor with uniform [0, 1) entries, drawn from the given RNG or the global default.

#### `uniform(shape: {number}, lo: number, hi: number, rng: RNG?, requiresGrad: boolean?) -> Tensor`

Random tensor with uniform entries over [lo, hi).

#### `numel(t: Tensor) -> number`

Number of elements (product of the shape dims).

#### `shapeOf(t: Tensor) -> {number}`

The tensor's shape array.

#### `isScalar(t: Tensor) -> boolean`

True when the tensor is 0-dim (shape {}).

#### `item(t: Tensor) -> number`

First data element as a number (for scalars and size-1 tensors).

#### `clone(t: Tensor) -> Tensor`

Copies data, detaches from the graph.

#### `detach(t: Tensor) -> Tensor`

A view sharing data with requiresGrad=false.

#### `ensureGrad(t: Tensor) -> {number}`

The tensor's gradient buffer, allocating and attaching a zeroed buffer if none exists yet.

#### `zeroGrad(t: Tensor)`

Zero the gradient buffer in place (no-op when the tensor has no grad).

#### `gradData(t: Tensor) -> {number}?`

The raw gradient array, or nil when the tensor has no gradient yet.

#### `add(a: Tensor, b: Tensor) -> Tensor`

Elementwise addition with broadcasting.

#### `sub(a: Tensor, b: Tensor) -> Tensor`

Elementwise subtraction with broadcasting.

#### `mul(a: Tensor, b: Tensor) -> Tensor`

Elementwise multiplication with broadcasting.

#### `div(a: Tensor, b: Tensor) -> Tensor`

Elementwise division with broadcasting.

#### `mulScalar(a: Tensor, s: number) -> Tensor`

Elementwise multiply by a scalar.

#### `addScalar(a: Tensor, s: number) -> Tensor`

Elementwise add a scalar.

#### `neg(a: Tensor) -> Tensor`

Elementwise negation.

#### `pow(a: Tensor, p: number) -> Tensor`

Elementwise power by a scalar exponent p.

#### `exp(a: Tensor) -> Tensor`

Elementwise exponential e^x.

#### `log(a: Tensor) -> Tensor`

Elementwise natural log (input/derivative floored at 1e-12).

#### `sqrt(a: Tensor) -> Tensor`

Elementwise square root (input/derivative floored at 0/1e-12).

#### `tanh(a: Tensor) -> Tensor`

Elementwise tanh.

#### `sigmoid(a: Tensor) -> Tensor`

Elementwise sigmoid (numerically stable).

#### `relu(a: Tensor) -> Tensor`

Elementwise ReLU (max(x, 0)).

#### `silu(a: Tensor) -> Tensor`

Swish / SiLU: x * sigmoid(x)

#### `gelu(a: Tensor) -> Tensor`

GELU (tanh approximation, as in GPT-2/J and most libs).

#### `clip(a: Tensor, lo: number, hi: number) -> Tensor`

Elementwise clamp into [lo, hi].

#### `matmul(a: Tensor, b: Tensor) -> Tensor`

C[i,j] = sum_k A[i,k] * B[k,j]  (2D only; use bmm for batches). When neither input requiresGrad the kernel auto-routes to the SIMD matmulFast path (no autograd node; ~1.3x native / ~4x interpreter).

#### `matmulFast(a: Tensor, b: Tensor) -> Tensor`

Forward-only SIMD matmul (no autograd). Same result as matmul(a, b) but uses the Luau `vector` type so the K loop carries 3 lanes in one machine op under native codegen (SSE/AVX) and runs off packed 3-wide copies of B's column blocks. ~3x faster than the scalar kernel on large matrices in the interpreter and ~1.75x under --codegen. Semi-sparse/preallocating: keeps the output a dense 1..N array (fast array-part table, no boxed hash entries). Requires the `vector` library (present in the Luau CLI / native codegen). On runtimes without the library (e.g. Roblox) this falls back to Tensor.matmul.

#### `bmm(a: Tensor, b: Tensor) -> Tensor`

Batched matmul: [B, M, K] @ [B, K, N] -&gt; [B, M, N].

#### `bmmFast(a: Tensor, b: Tensor) -> Tensor`

Forward-only SIMD batched matmul (no autograd): A [B,M,N] @ B [B,N,K] -&gt; [B,M,K]. Same 3-lane vector trick as matmulFast over the K columns; the scalar bmm measured 3.5-4.3x slower in the interpreter, ~1.2x under native codegen. Falls back to Tensor.bmm when the `vector` library is absent.

#### `bmmNT(a: Tensor, b: Tensor) -> Tensor`

SIMD A @ B^T for b stored already-oriented as [B, K, N]: out[b,i,k] = sum_j a[b,i,j]*b[b,k,j]. This avoids materializing a transposed copy (the attention Q@K^T call spent a full transpose + broadcast add per forward). Requires no autograd; falls back to bmm + transpose when baked-in.

#### `reshape(t: Tensor, newShape: {number}) -> Tensor`

View with a new shape (size must match); shares data, preserves requiresGrad.

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

#### `rmsNorm(x: Tensor, gamma: Tensor, eps: number) -> Tensor`

Fused RMSNorm over the LAST dim: y = x * gamma * rsqrt(mean(x^2)+eps). Works for any ndim (leading dims are just rows of the flat last-dim- contiguous array). The composite path (mul/mean/pow/addScalar/mul/mul) measured 33-39x slower (two temporaries + graph nodes per call).

#### `layerNorm(x: Tensor, gamma: Tensor, beta: Tensor?, eps: number) -> Tensor`

Fused LayerNorm over the LAST dim: y = (x - mu) * gamma * invstd + beta. beta may be nil (affine-free). Same fused-loop rationale as rmsNorm.

#### `rotary(x: Tensor, cosT: Tensor, sinT: Tensor) -> Tensor`

Fused RoPE pair-rotation on the trailing dim: every (2k-1, 2k) pair of x[..., t, :] is rotated by the angle whose cos/sin come from cosT/sinT (flat row-major [T, half], position t, frequency k). The composite path (reshape/select/mul/sub/stack/cat) measured 155-171x slower. Backward is the inverse rotation (angle negated) on the incoming grad.

#### `cat(ts: {Tensor}, dim: number?) -> Tensor`

Concatenate tensors along a dim (default last). Must match sizes on all non-cat dims.

#### `gt(a: Tensor, b: Tensor) -> Tensor`

Elementwise greater-than: returns 1.0/0.0. No gradient (masks are not differentiable inputs in the layers that use this).

#### `gatherRows(tableTensor: Tensor, rows: {number}) -> Tensor`

Gather rows from a [N, D] table tensor given a flat list of 1-based row ids. Output is [R, D]. Backward scatters the grad into the gathered rows.

#### `sliceDim(t: Tensor, dim: number, index: number) -> Tensor`

Extract a single slice along a dim: result has that dim removed. Backward routes the grad to exactly that slice's positions.

#### `selectLastDim(t: Tensor, index: number) -> Tensor`

Slice index along the last dim; shorthand for sliceDim(t, #t.shape, index).

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

