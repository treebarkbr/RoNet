--!strict
-- Activation functions (functional, no params).
return function(deps: any)
	local Tensor = deps.Tensor

	local Activations = {}

	function Activations.relu(x: any): any
		return Tensor.relu(x)
	end

	function Activations.silu(x: any): any
		return Tensor.silu(x)
	end

	function Activations.gelu(x: any): any
		return Tensor.gelu(x)
	end

	function Activations.tanh(x: any): any
		return Tensor.tanh(x)
	end

	function Activations.sigmoid(x: any): any
		return Tensor.sigmoid(x)
	end

	-- Softmax over the last dim by default (dim may be given as positive index).
	function Activations.softmax(x: any, dim: number?): any
		return Tensor.softmax(x, dim or #x.shape)
	end

	return Activations
end