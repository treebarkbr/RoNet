# Serialize

Source: `train/Serialize.luau`

Weight serialization. Pure string round-trip (no filesystem in the Luau CLI),  so checkpoints are portable strings, ready to feed into real Roblox storage later.  Deterministic, compact, full precision (tostring(number)).    snapshot(params)                       -&gt; {shape, data} states (deep copy)    dump(params)                           -&gt; one string for the whole param list    load(params, str)                      -&gt; writes parsed values into params in order    restore(params, snap)                  -&gt; restores from snapshot    parse(str)                             -&gt; {shape, data} list for inspection  Format per param:  "shape=&lt;d&gt;,&lt;d&gt;; &lt;n space-separated values&gt;", joined by "\n".  Parse is a single forward pass over the string (no per-value table builds).

## Methods

#### `snapshot(params: { any }) -> { any }`

Snapshot current param values into an independent table of {shape, data}.

#### `restore(params: { any }, snap: { any })`

Restore param values in-place from a snapshot ({shape, data} list).

#### `dump(params: { any }) -> string`

Serialize the whole param list into one portable string.

#### `parse(str: string) -> { any }`

Parse the dump format into a {shape, data} snapshot list. Single pass.

#### `load(params: { any }, str: string)`

Write parsed values back into an EXISTING param list (in order). Counts must match or it errors.

