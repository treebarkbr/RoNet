--!strict
-- End-to-end language model smoke: BPE tokenizer -> tiny transformer -> shifted
-- next-token loss -> Trainer loop -> autoregressive generate. Verifies the whole
-- pipeline is wired and learning (train loss drops, generated ids decode).
-- ENTRY script; composes the DI graph (mirrors init.luau).
local Util = require("../core/Util")
local PRNG = require("../core/PRNG")
local Tensor = require("../core/Tensor")
local Module = require("../nn/Module")
local Inits = require("../nn/Inits")
local Linear = require("../nn/Linear")
local Activations = require("../nn/Activations")
local Norm = require("../nn/Norm")
local Dropout = require("../nn/Dropout")
local Embedding = require("../nn/Embedding")
local RoPE = require("../nn/RoPE")
local Attention = require("../nn/Attention")
local FFN = require("../nn/FFN")
local Sequential = require("../nn/Sequential")
local MLP = require("../models/MLP")
local TransformerBlock = require("../models/TransformerBlock")
local Transformer = require("../models/Transformer")
local CrossEntropy = require("../loss/CrossEntropy")
local Losses = require("../loss/Losses")
local Scheduler = require("../train/Scheduler")
local Serialize = require("../train/Serialize")
local Trainer = require("../train/Trainer")
local AdamW = require("../optim/AdamW")
local BPE = require("../data/BPE")

local deps = {
	Util = Util,
	PRNG = PRNG,
	Tensor = Tensor,
}
deps.Module = Module(deps)
deps.Inits = Inits(deps)
deps.Linear = Linear(deps)
deps.Activations = Activations(deps)
deps.Norm = Norm(deps)
deps.Dropout = Dropout(deps)
deps.Embedding = Embedding(deps)
deps.RoPE = RoPE(deps)
deps.Attention = Attention(deps)
deps.FFN = FFN(deps)
deps.Sequential = Sequential(deps)
deps.MLP = MLP(deps)
deps.TransformerBlock = TransformerBlock(deps)
deps.Transformer = Transformer(deps)
deps.CrossEntropy = CrossEntropy(deps)
deps.Losses = Losses(deps)
deps.Scheduler = Scheduler(deps)
deps.Serialize = Serialize(deps)
deps.Trainer = Trainer(deps)
deps.AdamW = AdamW(deps)
deps.BPE = BPE

Tensor.setRNG(PRNG.new(7))

local TransformerM = deps.Transformer
local LossesM = deps.Losses
local TrainerM = deps.Trainer
local SchedulerM = deps.Scheduler
local AdamWM = deps.AdamW
local ModuleM = deps.Module
local SerializeM = deps.Serialize
local BPE_M = deps.BPE

-- Tiny closed-world language: permutations of four 3-letter words, so the BPE
-- vocab stays small but the corpus is long enough for a real windowed batch.
local toy = {
	"fox hex yap zoo ",
	"yap fox hex zoo ",
	"zoo yap fox hex ",
	"hex zoo yap fox ",
	"fox yap zoo hex ",
	"zoo fox hex yap ",
	"yap zoo fox hex ",
	"hex fox zoo yap ",
	"fox zoo hex yap ",
	"zoo hex yap fox ",
}

local function check(cond: boolean, msg: string)
	if not cond then error("FAIL: " .. msg) end
end

local total = 0
local function ok(msg: string)
	total += 1
	print("ok:   " .. msg)
end

-- 1. Tokenize the toy language with a tiny BPE vocab.
local tok = BPE_M.train(toy, 32)
ok("trained " .. #(tok:merges()) .. " merges, vocab " .. tok:vocabSize())
local corpus = table.concat(toy)
local idsAll = tok:encode(corpus)
check(tok:decode(idsAll) == corpus, "tokenizer round-trip on toy corpus")

-- 2. Build tiny transformer (L=2, C=16, 2 heads, dFFN=32, vocab from BPE).
local V = tok:vocabSize()
local maxSeq = 10
local model = TransformerM.new("sm_lm", {
	C = 16, numBlocks = 2, nbHeads = 2, kvHeads = 2, headDim = 8,
	ffnHidden = 32, vocab = V, maxSeq = maxSeq, tieEmbeds = false,
	initType = "default",
}, PRNG.new(3))
local params = ModuleM.collectParams(model)
ok("params " .. #params)

-- 3. Sliding windows as plain {number} rows; Trainer batches them to [B, T].
local T = 4
local step = 2
local X: { { number } } = {}
local count = 0
for s = 1, #idsAll - T, step do
	count += 1
	local row = table.create(T)
	for t = 1, T do row[t] = idsAll[s + t - 1] end
	X[count] = row
end
check(count >= 8, "enough windows (" .. count .. ")")

local opt = AdamWM.new(params, 0.02, { beta1 = 0.9, beta2 = 0.99 })
local sched = SchedulerM.forOptimizer(opt, SchedulerM.warmupCosine(10, 300, 0.1))

local lossFn = function(m, xs, ys)
	local logits = m:forwardBatch(xs) -- [B, T, V]
	return LossesM.lmCrossEntropy(logits, xs, 0.0)
end
local evalFn = function(m)
	local loss = 0
	for i = 1, #X do
		local x1 = Tensor.fromData(X[i], { 1, T }, false)
		local l = LossesM.lmCrossEntropy(m:forwardBatch(x1), x1, 0.0)
		loss += Tensor.item(l)
	end
	return loss / #X
end

local trainer = TrainerM.new({
	model = model, optimizer = opt, schedule = sched,
	X = X, Y = X, -- Y ignored by lossFn (uses only xs; targets shift inside lmCE)
	lossFn = lossFn, evalFn = evalFn, evalEvery = 5,
	epochs = 20, batchSize = 8, featureShape = { T },
	onEpochEnd = function(_ep, _loss) end,
})
local history, evals = trainer:fit()
local first = history[1]
local last = history[#history]
check(last < first, string.format("loss dropped %.4f -> %.4f", first, last))
ok(string.format("loss %.4f -> %.4f", first, last))
check(#evals >= 1 and evals[#evals] < first, "eval loss dropped too")

-- 4. Generate: seed from the corpus, output must decode to a string.
model:eval()
local seed = { idsAll[1], idsAll[2] }
local gen = model:generate(seed, 8, { temperature = 0.5, topK = 5 })
check(#gen == #seed + 8, "generated expected count")
local text = tok:decode(gen)
check(type(text) == "string" and #text > 0, "generated text decodes")
model:train()
ok("generated " .. #gen .. " tokens decodable")

-- 5. Full state round-trip: Serialize params into a fresh identical model.
local dumpStr = SerializeM.dump(params)
SerializeM.load(ModuleM.collectParams(TransformerM.new("sm_lm2", {
	C = 16, numBlocks = 2, nbHeads = 2, kvHeads = 2, headDim = 8,
	ffnHidden = 32, vocab = V, maxSeq = maxSeq, tieEmbeds = false,
}, PRNG.new(3))), dumpStr)
ok("state dump/load round-trip")

-- 6. BPE state persistence.
local st = tok:state()
local tok2 = BPE_M.fromState(st)
local ids3 = tok2:encode(corpus)
check(#ids3 == #idsAll and tok2:decode(ids3) == corpus, "BPE state round-trip encodes identically")
ok("BPE state round-trip")

print(string.format("PASS LM: %d checks", total))