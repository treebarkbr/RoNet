# Util

Source: `core/Util.luau`

Util: small numeric + table helpers shared by RoNetV4.1.  Self-contained (no requires) so the hot core stays dependency-free.

## Methods

#### `typeName(v: any) -> string`

typeName(v): the Luau type name, or a table's metatable __name if present (lets library classes report meaningful names).

#### `lerp(a: number, b: number, t: number) -> number`

Linear interpolation between a and b at t (t in [0,1]).

#### `clamp(v: number, lo: number, hi: number) -> number`

Clamp value into [lo, hi].

#### `swap(arr: {any}, i: number, j: number)`

Swap two elements of an array.

#### `fill(n: number, value: T) -> {T}`

Create an array of `n` copies of `value` without iterator overhead.

#### `zeros(n: number) -> {number}`

Array of n zeros.

#### `ones(n: number) -> {number}`

Array of n ones.

#### `copyArray(source: {T}) -> {T}`

Shallow copy of an array (table.move).

#### `range(n: number) -> {number}`

Array `{1, 2, ..., n}`.

#### `argmax(arr: {number}) -> number`

index of maximum element

#### `argmin(arr: {number}) -> number`

Index of the minimum element.

#### `sum(arr: {number}) -> number`

Sum of all elements in the array.

#### `mean(arr: {number}) -> number`

Arithmetic mean of the array (0 for an empty array).

#### `variance(arr: {number}) -> number`

Population variance of the array (0 for fewer than 2 elements).

#### `softmax(logits: {number}, out: {number}?) -> {number}`

Numerically stable softmax over a logits array. Writes into `out` when provided (reused buffers), otherwise returns a new array.

#### `onehot(index: number, size: number) -> {number}`

One-hot vector of size `size` with a 1 at index `index`.

#### `sigmoid(x: number) -> number`

Stable sigmoid

#### `tanhApprox(x: number) -> number`

Cheap tanh approximation via exp (-2x).

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

In-place copy of source into dest; returns dest.

#### `concatTables(a: {T}, b: {T}) -> {T}`

Concatenate two arrays into a new one.

#### `roundToMultiple(x: number, mult: number) -> number`

Round to the nearest multiple of `mult`.

#### `softmaxRows(data: {number}, t: number, d: number)`

Stable softmax over columns of a [T, D] row-major flat array, in place.

#### `l2Norm(data: {number}) -> number`

Lightweight matrix norm helpers (L2 over flat array).

#### `l2NormSq(data: {number}) -> number`

Squared L2 norm (sum of squares) over a flat array.

#### `clipGradNorm(data: {number}, clip: number, eps: number?) -> number`

In-place global-norm rescale: if target &gt; clip, scale all by clip/target. Returns the multiplicative factor applied.

