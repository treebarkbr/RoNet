--!strict
-- ENTRY: run a SLICED TinyStories 15M sub-model (2 blocks, first 2048 vocab
-- rows -- genuine weights, portable fixture ~35MB). NOT part of run_all.sh.
-- Seed uses common SentencePiece pieces (ids < 2048), converted to 1-based.
require("./_tinystories_drv")(
	"./fixtures/tinystories15m.demo.weights",
	{ 279, 323, 1184, 472, 298, 264, 374 },
	40
)