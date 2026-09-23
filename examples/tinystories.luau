--!strict
-- Streaming text generation with a real TinyStories checkpoint (Karpathy's
-- stories15M lineage, ~15M params, tested on TinyStories). Loads the llama2.c
-- weights imported by tools/pt2ronet.py and streams each sampled token as it is
-- produced, ending with a throughput report.
--
-- Setup (the weight fixtures are gitignored and must be regenerated once):
--   1. Grab the checkpoint:
--        mkdir -p /tmp/tinystories
--        curl -L -o /tmp/tinystories/model.safetensors \
--          https://huggingface.co/0rn0/llama2-15m-tinystories/resolve/main/model.safetensors
--        curl -L -o /tmp/tinystories/config.json \
--          https://huggingface.co/0rn0/llama2-15m-tinystories/resolve/main/config.json
--        curl -L -o /tmp/tinystories/tokenizer.model \
--          https://huggingface.co/0rn0/llama2-15m-tinystories/resolve/main/tokenizer.model
--   2. Convert to a RoNet weight module:
--        python3 tools/pt2ronet.py /tmp/tinystories/model.safetensors \
--          --config /tmp/tinystories/config.json --out tests/fixtures/tinystories15m.weights
--   3. Run (streaming; ~0.7s per token on the full 15M model):
--        .tools/luau -O2 --codegen examples/tinystories.luau
--      and decode the IDS line to text with:
--        .tools/luau -O2 --codegen examples/tinystories.luau \
--          | python3 tools/ts_decode.py /tmp/tinystories/tokenizer.model
--
-- Swap in the sliced demo fixture (2 layers / vocab 2048, loads in ~3s) for a
-- quick smoke run by pointing WEIGHTS_PATH at tinystories15m.demo.weights.
local Util = require("../core/Util")
local PRNG = require("../core/PRNG")
local Tensor = require("../core/Tensor")
local Module = require("../nn/Module")
local Inits = require("../nn/Inits")
local Norm = require("../nn/Norm")
local Dropout = require("../nn/Dropout")
local Embedding = require("../nn/Embedding")
local RoPE = require("../nn/RoPE")
local Attention = require("../nn/Attention")
local FFN = require("../nn/FFN")
local TransformerBlock = require("../models/TransformerBlock")
local Transformer = require("../models/Transformer")
local Serialize = require("../train/Serialize")

local deps = {
	Util = Util,
	PRNG = PRNG,
	Tensor = Tensor,
}
deps.Module = Module(deps)
deps.Inits = Inits(deps)
deps.Norm = Norm(deps)
deps.Dropout = Dropout(deps)
deps.Embedding = Embedding(deps)
deps.RoPE = RoPE(deps)
deps.Attention = Attention(deps)
deps.FFN = FFN(deps)
deps.TransformerBlock = TransformerBlock(deps)
deps.Transformer = Transformer(deps)
deps.Serialize = Serialize(deps)

Tensor.setRNG(PRNG.new(7))

local WEIGHTS_PATH = "../tests/fixtures/tinystories15m.weights"
local SEED = { 9039, 2502, 264, 932, 29893, 728, 472, 264, 2218, 7827, 4258,
	366, 2355, 29890, 2297, 10601, 298, 264, 2320, 3700, 412, 903, 5638, 29890 }
local N_TOKENS = 40
local TEMPERATURE = 0.8
local TOP_K = 50

print("loading weights ...")
local t0 = os.clock()
local W = require(WEIGHTS_PATH)
local model = deps.Transformer.new("ts", W.cfg, PRNG.new(1))
deps.Serialize.load(deps.Module.collectParams(model), W.weights)
local ps = deps.Module.collectParams(model)
local nFloats = 0
for _, t in ps do nFloats += #t.data end
print(string.format("loaded %s (%.2fM floats, %d tensors) in %.2fs",
	model.name, nFloats / 1e6, #ps, os.clock() - t0))

model:eval()

print(string.format("generating +%d tokens (temp=%.2f, topK=%d) ...", N_TOKENS, TEMPERATURE, TOP_K))
local tGen = os.clock()
local out = model:generate(SEED, N_TOKENS, {
	temperature = TEMPERATURE,
	topK = TOP_K,
	cache = true,
	onToken = function(rid, step)
		print(string.format("tok %3d/%d  (+%.2fs)  id %d", step, N_TOKENS,
			os.clock() - tGen, rid))
	end,
})
local dt = os.clock() - tGen

print("IDS " .. table.concat(out, " "))
print(string.format(
	"STATS generated=%d tokens in %.2fs = %.2f tokens/s (full sequence incl. seed: %d)",
	N_TOKENS, dt, N_TOKENS / dt, #out))