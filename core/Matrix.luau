--!strict
-- Raw 2D-matrix linear algebra on flat row-major arrays. No graph, no requires.
-- Used by matrix preconditioners (Muon Newton-Schulz, SOAP eigenbasis). All
-- arrays are row-major row*cols+col, 1-indexed.
local Matrix = {}

-- matmul: A (ma x na) times B (mb x nb), requires na == mb.
function Matrix.mm(a: {number}, b: {number}, ma: number, na: number, mb: number, nb: number): {number}
	local out = table.create(ma * nb, 0)
	for i = 0, ma - 1 do
		local iout = i * nb
		for k = 0, na - 1 do
			local aik = a[i * na + k + 1]
			for j = 0, nb - 1 do
				out[iout + j + 1] += aik * b[k * nb + j + 1]
			end
		end
	end
	return out
end

function Matrix.trans(a: {number}, m: number, n: number): {number}
	local out = table.create(m * n)
	for i = 0, m - 1 do
		for j = 0, n - 1 do
			out[j * m + i + 1] = a[i * n + j + 1]
		end
	end
	return out
end

function Matrix.eye(n: number): {number}
	local out = table.create(n * n, 0)
	for i = 1, n do out[(i - 1) * n + i] = 1 end
	return out
end

function Matrix.fro(a: {number}): number
	local s = 0
	for i = 1, #a do s += a[i] * a[i] end
	return math.sqrt(s)
end

function Matrix.scale(a: {number}, s: number): {number}
	local out = table.create(#a)
	for i = 1, #a do out[i] = a[i] * s end
	return out
end

function Matrix.copy(a: {number}): {number}
	local out = table.create(#a)
	for i = 1, #a do out[i] = a[i] end
	return out
end

function Matrix.zeros(n: number): {number}
	return table.create(n, 0)
end

-- Frobenius inner product of two equally-sized arrays.
function Matrix.dot(a: {number}, b: {number}): number
	local s = 0
	for i = 1, #a do s += a[i] * b[i] end
	return s
end

-- Jacobi eigenvalues of a real SYMMETRIC matrix a (flat n x n, row-major).
-- Returns (eigenvalues, eigenvectors) where eigenvectors columns (i.e. V such
-- that a = V diag(w) V'); each eigenvector is a column of V (row-major V).
function Matrix.eigh(a: {number}, n: number): ({number}, {number})
	local v = Matrix.eye(n)
	local w = Matrix.copy(a)
	local maxIter = 50 * n * n
	for _ = 1, maxIter do
		local off = Matrix.fro(w)
		local best = -math.huge
		local p = 1
		local q = 2
		for i = 1, n do
			for j = i + 1, n do
				local x = math.abs(w[(i - 1) * n + j])
				if x > best then
					best = x
					p = i
					q = j
				end
			end
		end
		if best <= 1e-13 * off or best <= 1e-17 then
			break
		end
		local app = w[(p - 1) * n + p]
		local aqq = w[(q - 1) * n + q]
		local apq = w[(p - 1) * n + q]
		local tau = (aqq - app) / (2 * apq)
		local t
		if tau >= 0 then
			t = 1 / (tau + math.sqrt(1 + tau * tau))
		else
			t = -1 / (-tau + math.sqrt(1 + tau * tau))
		end
		local c = 1 / math.sqrt(1 + t * t)
		local s = t * c
		for k = 1, n do
			if k ~= p and k ~= q then
				local hkp = w[(k - 1) * n + p]
				local hkq = w[(k - 1) * n + q]
				w[(k - 1) * n + p] = c * hkp - s * hkq
				w[(k - 1) * n + q] = s * hkp + c * hkq
				w[(p - 1) * n + k] = c * hkp - s * hkq
				w[(q - 1) * n + k] = s * hkp + c * hkq
				local vkp = v[(k - 1) * n + p]
				local vkq = v[(k - 1) * n + q]
				v[(k - 1) * n + p] = c * vkp - s * vkq
				v[(k - 1) * n + q] = s * vkp + c * vkq
			end
		end
		w[(p - 1) * n + p] = app - t * apq
		w[(q - 1) * n + q] = aqq + t * apq
		w[(p - 1) * n + q] = 0
		w[(q - 1) * n + p] = 0
		local vpp = v[(p - 1) * n + p]
		local vpq = v[(p - 1) * n + q]
		v[(p - 1) * n + p] = c * vpp - s * vpq
		v[(p - 1) * n + q] = s * vpp + c * vpq
		local vqp = v[(q - 1) * n + p]
		local vqq = v[(q - 1) * n + q]
		v[(q - 1) * n + p] = c * vqp - s * vqq
		v[(q - 1) * n + q] = s * vqp + c * vqq
	end
	return w, v
end

-- QR factorization (column Gram-Schmidt). x flat m x n (m >= n). Returns
-- (q, r), q is m x n with orthonormal columns, r is n x n upper triangular.
function Matrix.qr(x: {number}, m: number, n: number): ({number}, {number})
	local q = Matrix.zeros(m * n)
	local r = Matrix.zeros(n * n)
	for k = 1, n do
		local col = table.create(m)
		for i = 1, m do col[i] = x[(i - 1) * n + k] end
		for j = 1, k - 1 do
			local proj = 0
			for i = 1, m do proj += q[(i - 1) * n + j] * col[i] end
			r[(j - 1) * n + k] = proj
			for i = 1, m do col[i] -= proj * q[(i - 1) * n + j] end
		end
		local norm = 0
		for i = 1, m do norm += col[i] * col[i] end
		norm = math.sqrt(norm)
		if norm < 1e-15 then
			norm = 1e-15
		end
		r[(k - 1) * n + k] = norm
		for i = 1, m do q[(i - 1) * n + k] = col[i] / norm end
	end
	return q, r
end

-- Newton-Schulz orthogonalization of a matrix g (flat rows x cols). Mirrors the
-- Muon blog implementation: work in the min-dimension orientation (rows <= cols),
-- normalize so the top singular value is <= 1, then iterate 5+ steps of
-- phi(x) = a*x + b*x^3 + c*x^5. Returns the orthogonalized matrix (same
-- orientation as input).
function Matrix.newtonSchulz(g: {number}, rows: number, cols: number, iters: number): {number}
	local transpose = rows > cols
	local m = if transpose then cols else rows
	local n = if transpose then rows else cols
	local x = if transpose then Matrix.trans(g, rows, cols) else Matrix.copy(g)
	local a, b, c = 3.4445, -4.7750, 2.0315
	local norm = Matrix.fro(x)
	if norm < 1e-12 then return Matrix.copy(x) end
	for i = 1, m * n do x[i] = x[i] / norm end
	for _ = 1, iters do
		local xt = Matrix.trans(x, m, n)
		local A = Matrix.mm(x, xt, m, n, n, m)
		local a2 = Matrix.mm(A, A, m, m, m, m)
		local B = Matrix.zeros(m * m)
		for i = 1, m * m do B[i] = b * A[i] + c * a2[i] end
		local bx = Matrix.mm(B, x, m, m, m, n)
		local out = Matrix.zeros(m * n)
		for i = 1, m * n do out[i] = a * x[i] + bx[i] end
		x = out
	end
	if transpose then
		return Matrix.trans(x, n, m)
	end
	return x
end

return Matrix