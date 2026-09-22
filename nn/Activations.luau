--!strict
-- Activation functions (functional, no params).
return function(deps: any)
	local Tensor = deps.Tensor

	local Activations = {}

	-- ReLU (max(x, 0)).
	function Activations.relu(x: any): any
		return Tensor.relu(x)
	end

	-- SiLU / Swish.
	function Activations.silu(x: any): any
		return Tensor.silu(x)
	end

	-- GELU (tanh approximation).
	function Activations.gelu(x: any): any
		return Tensor.gelu(x)
	end

	-- Hyperbolic tangent.
	function Activations.tanh(x: any): any
		return Tensor.tanh(x)
	end

	-- Numerically stable sigmoid.
	function Activations.sigmoid(x: any): any
		return Tensor.sigmoid(x)
	end

	-- Softmax over the last dim by default (dim may be given as positive index).
	function Activations.softmax(x: any, dim: number?): any
		return Tensor.softmax(x, dim or #x.shape)
	end

	return Activations
end