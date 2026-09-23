# BPE

Source: `data/BPE.luau`

Byte-level BPE tokenizer (GPT-style sub-word units), pure Luau (no deps).    BPE.train(text: string | {string}, numMerges) -&gt; tokenizer    tok:encode(text)  -&gt; {number}   (token ids, ids 0..255 are raw bytes)    tok:decode(ids)   -&gt; string    tok:vocabSize()   -&gt; number     (256 + numMerges)    tok:merges()      -&gt; { {number, number} } ordered (a, b) build pairs    tok:piece(id)     -&gt; string     (byte string for a token id)  Deterministic: identical corpus + numMerges yields identical merges (ties are  broken by the lexicographically smallest pair, then earliest appearance).  Performance: byte sequences live as arrays of numbers; pair counting uses a  single composite-integer key (a * BASE + b) to avoid string allocation.  Memory: token pieces are stored packed as byte strings (one allocation per  token, not one table per byte), and merge rules are a single flat array  {a1, b1, a2, b2, ...} instead of one {a, b} table per rule. Keeps the  per-vocab-entry table count to a bare minimum.

## Methods

#### `train(corpus: any, numMerges: number) -> any`

Train: greedily merge the most frequent adjacent pair numMerges times. Returns the tokenizer instance (chainable).

#### `new() -> any`

An empty byte-level tokenizer (vocab 256, no merges). Useful for building a tokenizer by hand with addMerge.

#### `addMerge(a: number, b: number) -> number`

Add a single merge rule (a, b) -&gt; newId (manual extension; ids must be new).

#### `vocabSize() -> number`

Total vocabulary size (256 bytes + number of merge rules).

#### `merges() -> { any }`

Ordered list of merge rules {a, b} as applied during training.

#### `piece(id: number) -> string`

The byte string a token id decodes to.

#### `encode(text: string) -> { number }`

Encode: bytes -&gt; ids, then apply merge rules in training order, each to exhaustion, left-to-right. Matches the training path exactly. Linear-time structure: tokens live in a doubly-linked list; a per-rule candidate queue tracks only the positions where a rule could fire, and a merge creates at most two new candidate positions (its new neighbours), so a full scan per rule is never needed.

#### `decode(ids: { number }) -> string`

Decode token ids back into a string.

#### `state() -> any`

Serializable tokenizer state: { vocab, merges }. Feeds BPE.fromState.

#### `fromState(st: any) -> any`

Rebuild a tokenizer from a state produced by tok:state().

