--!strict
-- ENTRY: run the FULL TinyStories 15M import (6 layers, vocab 32000) and
-- generate. NOT part of run_all.sh (spike, ~40s+). Seed = "Once upon a time,
-- there was a little girl named Lily." tokenized by the checkpoint's
-- SentencePiece model (llama2.c ids, converted to RoNet 1-based).
require("./_tinystories_drv")(
	"./fixtures/tinystories15m.weights",
	{ 9039, 2502, 264, 932, 29893, 728, 472, 264, 2218, 7827, 4258, 366, 2355,
		29890, 2297, 10601, 298, 264, 2320, 3700, 412, 903, 5638, 29890 },
	40
)