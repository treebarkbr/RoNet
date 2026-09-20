# Util

Source: `core/Util.luau`

Util: small numeric + table helpers shared by RoNetV4.1.  Self-contained (no requires) so the hot core stays dependency-free.

## Methods

#### `typeName(v: any) -> string`

#### `lerp(a: number, b: number, t: number) -> number`

#### `clamp(v: number, lo: number, hi: number) -> number`

Clamp value into [lo, hi].

#### `swap(arr: {any}, i: number, j: number)`

Swap two elements of an array.

#### `fill(n: number, value: T) -> {T}`

Create an array of `n` copies of `value` without iterator overhead.

#### `zeros(n: number) -> {number}`

#### `ones(n: number) -> {number}`

#### `copyArray(source: {T}) -> {T}`

#### `range(n: number) -> {number}`

#### `argmax(arr: {number}) -> number`

index of maximum element

#### `argmin(arr: {number}) -> number`

#### `sum(arr: {number}) -> number`

#### `mean(arr: {number}) -> number`

#### `variance(arr: {number}) -> number`

#### `softmax(logits: {number}, out: {number}?) -> {number}`

#### `onehot(index: number, size: number) -> {number}`

One-hot vector of size `size` with a 1 at index `index`.

#### `sigmoid(x: number) -> number`

Stable sigmoid

#### `tanhApprox(x: number) -> number`

#### `randInt(...: any) -> number`

Pseudo-random int in [1, hi] or [lo, hi] (delegates to global PRNG).

#### `shapeSize(shape: {number}) -> number`

Total number of elements implied by a shape.

#### `shapesEqual(a: {number}, b: {number}) -> boolean`

Are two shapes equal?

#### `shapeString(shape: {number}) -> string`

Pretty-print a shape as "2x3x4".

#### `addInplace(dest: {number}, source: {number})`

Sum a list of numbers into dest (used to accumulate tensors).

#### `scaleInplace(arr: {number}, factor: number)`

In-place scalar multiply.

#### `copyInplace(dest: {number}, source: {number})`

#### `concatTables(a: {T}, b: {T}) -> {T}`

#### `roundToMultiple(x: number, mult: number) -> number`

#### `softmaxRows(data: {number}, t: number, d: number)`

Stable softmax over columns of a [T, D] row-major flat array, in place.

#### `l2Norm(data: {number}) -> number`

Lightweight matrix norm helpers (L2 over flat array).

#### `l2NormSq(data: {number}) -> number`

#### `clipGradNorm(data: {number}, clip: number, eps: number?) -> number`

In-place global-norm rescale: if target &gt; clip, scale all by clip/target. Returns the multiplicative factor applied.

