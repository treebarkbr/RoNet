# Matrix

Source: `core/Matrix.luau`

Raw 2D-matrix linear algebra on flat row-major arrays. No graph, no requires.  Used by matrix preconditioners (Muon Newton-Schulz, SOAP eigenbasis). All  arrays are row-major row*cols+col, 1-indexed.

## Methods

#### `mm(a: {number}, b: {number}, ma: number, na: number, mb: number, nb: number) -> {number}`

matmul: A (ma x na) times B (mb x nb), requires na == mb.

#### `trans(a: {number}, m: number, n: number) -> {number}`

Transpose of an m x n flat matrix into an n x m one.

#### `eye(n: number) -> {number}`

n x n identity matrix (flat row-major).

#### `fro(a: {number}) -> number`

Frobenius norm (sqrt of sum of squares) of a flat array.

#### `scale(a: {number}, s: number) -> {number}`

Scalar multiply: returns a new flat array `a * s`.

#### `copy(a: {number}) -> {number}`

Copy of a flat array (element-wise clone).

#### `zeros(n: number) -> {number}`

Zeroed flat array of length n.

#### `dot(a: {number}, b: {number}) -> number`

Frobenius inner product of two equally-sized arrays.

#### `eigh(a: {number}, n: number) -> ({number}, {number})`

Jacobi eigenvalues of a real SYMMETRIC matrix a (flat n x n, row-major). Returns (eigenvalues, eigenvectors) where eigenvectors columns (i.e. V such that a = V diag(w) V'); each eigenvector is a column of V (row-major V).

#### `qr(x: {number}, m: number, n: number) -> ({number}, {number})`

QR factorization (column Gram-Schmidt). x flat m x n (m &gt;= n). Returns (q, r), q is m x n with orthonormal columns, r is n x n upper triangular.

#### `newtonSchulz(g: {number}, rows: number, cols: number, iters: number) -> {number}`

Newton-Schulz orthogonalization of a matrix g (flat rows x cols). Mirrors the Muon blog implementation: work in the min-dimension orientation (rows &lt;= cols), normalize so the top singular value is &lt;= 1, then iterate 5+ steps of phi(x) = a*x + b*x^3 + c*x^5. Returns the orthogonalized matrix (same orientation as input).

