# BPE

Source: `data/BPE.luau`

Byte-level BPE tokenizer (GPT-style sub-word units), pure Luau (no deps).    BPE.train(text: string | {string}, numMerges) -&gt; tokenizer    tok:encode(text)  -&gt; {number}   (token ids, ids 0..255 are raw bytes)    tok:decode(ids)   -&gt; string    tok:vocabSize()   -&gt; number     (256 + numMerges)    tok:merges()      -&gt; { {number, number} } ordered (a, b) build pairs    tok:piece(id)     -&gt; string     (byte string for a token id)  Deterministic: identical corpus + numMerges yields identical merges (ties are  broken by the lexicographically smallest pair, then earliest appearance).  Performance: byte sequences live as arrays of numbers; pair counting uses a  single composite-integer key (a * BASE + b) to avoid string allocation.

## Methods

#### `train(corpus: any, numMerges: number) -> any`

Train: greedily merge the most frequent adjacent pair numMerges times. Returns the tokenizer instance (chainable).

#### `new() -> any`

#### `addMerge(a: number, b: number) -> number`

Add a single merge rule (a, b) -&gt; newId (manual extension; ids must be new).

#### `vocabSize() -> number`

#### `merges() -> { any }`

#### `piece(id: number) -> string`

#### `encode(text: string) -> { number }`

Encode: bytes -&gt; ids, then repeatedly apply merges in training order, each to exhaustion. Matches the training path, so trained text round-trips exactly.

#### `decode(ids: { number }) -> string`

#### `state() -> any`

Serializable tokenizer state: { vocab, merges }. Feeds BPE.fromState.

#### `fromState(st: any) -> any`

Rebuild a tokenizer from a state produced by tok:state().

