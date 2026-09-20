--!strict
-- RoNetV4.1 --!strict NN library.
-- Root dependency-injection entry. This file is the single ENTRY script; only
-- entry scripts may require(), so the DI graph is wired here exactly once.
-- (Named RoNet.luau rather than init.luau because the stock Luau CLI/0.739
-- resolver cannot require a module literally named "init".)
-- @return deps: fully-wired module graph (Util, PRNG, Tensor, nn.*, models.*, loss.*)
local Util = require("./core/Util")
local PRNG = require("./core/PRNG")
local Tensor = require("./core/Tensor")

local deps: { [string]: any } = {
	Util = Util,
	PRNG = PRNG,
	Tensor = Tensor,
}
deps.Module = require("./nn/Module")(deps)
deps.Inits = require("./nn/Inits")(deps)
deps.Linear = require("./nn/Linear")(deps)
deps.Activations = require("./nn/Activations")(deps)
deps.Norm = require("./nn/Norm")(deps)
deps.Dropout = require("./nn/Dropout")(deps)
deps.Embedding = require("./nn/Embedding")(deps)
deps.RoPE = require("./nn/RoPE")(deps)
deps.Attention = require("./nn/Attention")(deps)
deps.FFN = require("./nn/FFN")(deps)
deps.Sequential = require("./nn/Sequential")(deps)
deps.MLP = require("./models/MLP")(deps)
deps.TransformerBlock = require("./models/TransformerBlock")(deps)
deps.Transformer = require("./models/Transformer")(deps)
deps.CrossEntropy = require("./loss/CrossEntropy")(deps)
deps.Matrix = require("./core/Matrix")
deps.BPE = require("./data/BPE")
deps.AdamW = require("./optim/AdamW")(deps)
deps.AdEMAMix = require("./optim/AdEMAMix")(deps)
deps.Lion = require("./optim/Lion")(deps)
deps.NAdamW = require("./optim/NAdamW")(deps)
deps.CautiousAdamW = require("./optim/CautiousAdamW")(deps)
deps.ScheduleFreeAdamW = require("./optim/ScheduleFreeAdamW")(deps)
deps.SOAP = require("./optim/SOAP")(deps)
deps.Muon = require("./optim/Muon")(deps)
deps.MuonAdamW = require("./optim/MuonAdamW")(deps)
deps.Losses = require("./loss/Losses")(deps)
deps.Scheduler = require("./train/Scheduler")(deps)
deps.EMA = require("./train/EMA")(deps)
deps.Serialize = require("./train/Serialize")(deps)
deps.Trainer = require("./train/Trainer")(deps)

return deps