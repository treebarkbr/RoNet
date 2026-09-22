# Parallel

Source: `core/Parallel.luau`

Parallel Luau support (Actors) for RoNetV4.1.  Single source of truth for training/eval compute that can run EITHER in the  calling thread ("single") OR on a pool of Actor workers ("multi"). The mode  is a user-facing toggle:      Parallel.setMode("single")   -- default; everything in-process      Parallel.setMode("multi")    -- dispatch jobs to N Actor workers      Parallel.setMode("auto")     -- multi on a Roblox runtime, else single  Grounded in the official Parallel Luau docs / API reference:    * Each Actor owns a separate Luau VM. ModuleScripts required inside an      Actor are re-required per VM, so every worker gets a private copy of the      RoNet graph and module-level state NEVER races across VMs.    * require() is illegal in a desynchronized (parallel) phase, so the worker      script requires RoNet at its top level (serial phase), keeps the refs,      then registers BindToMessageParallel. All math runs inside the parallel      handler using already-loaded refs.    * Instance mutation is illegal in parallel, so JOBS ARE PURE COMPUTE:      they only build Lua tables and run Tensor/Serialize/math.* calls. The      disposable MLP a job builds is fully local to the worker VM.    * Instances cannot cross VM boundaries, so payloads travel as plain tables      via Actor:SendMessage (copied by value), models travel as weight strings      (Serialize.dump/load), and results come back through a pool-owned      BindableEvent whose instance the worker resolves through the DataModel.    * SINGLE/MULTI PARITY IS BY CONSTRUCTION: the same runJob() does the math.      multi just slices the batch across workers and re-aggregates.  Ops (identical code path in both modes):    forward  : outputs for a batch of rows    loss     : scalar loss over the batch (row-count-weighted mean across shards)    backward : scalar loss + per-param gradients; the pool writes SUMMED grads               into the master model's params so the caller's optimizer:step()               works afterwards    scoreGenomes : per-genome loss (GA fitness) - the headline op for running               an evolutionary population eval in parallel  "multi" requires a real Roblox runtime (game + Instance + Actor + task). In  the Luau CLI (or anywhere those globals are absent) every mode falls back to  "single", so this module always compiles and runs.

## Methods

### Class Parallel

#### `available() -> boolean`

True when running on a real Roblox runtime (game, Instance, Actor and task are all present), i.e. when "multi" mode can actually spawn Actor workers. Always false in the Luau CLI.

#### `setMode(m: string, workers: number?)`

Set execution mode. Unknown modes error; "multi" without a runtime downgrades to "single" with a warning.

#### `getMode() -> string`

The configured mode string ("single", "multi" or "auto"), exactly as last set. Use Parallel.effective() for the mode that will actually run.

#### `setWorkers(n: number)`

Set the worker count (&gt;= 1) used by newPool/attach when no explicit worker count is given.

#### `getWorkers() -> number`

The configured worker count (&gt;= 1).

#### `setParent(p: any?)`

Optional parent instance the pool folder is created under.

#### `effective() -> string`

Effective mode right now ("auto" resolves against the runtime).

#### `weightsString(params: { any }) -> string`

Portable weights string for a param list (Serialize.dump). Models (and GA genomes) travel across the Actor VM boundary as such strings.

#### `modelFromWeights(layerSizes: { number }, act: string, weightsStr: string) -> any`

Rebuild a fresh MLP from a portable weights string (used on workers and by the genome op). act must match the model the string came from.

#### `genomeToWeightsString(genome: any) -> string`

GA genome {layerSizes, weights, biases, act?} -&gt; a Serialize.dump string (genomes therefore cross the VM boundary as ONE string each).

#### `runJob(job: any) -> any`

THE shared job implementation. Payload fields:   op: "forward" | "loss" | "backward" | "genome"   layerSizes / act / weightsStr : the single model everyone shares   rows        : this shard's rows ({ {number} })   targets     : {number} or { {number} } slice (loss/backward/genome)   lossName    : "mse" | "bce" | "crossentropy" (loss/backward/genome)   genomes     : { string } weight-strings (genome only) Returns a plain-Lua table (numbers/tables only): safe to copy back.

#### `newPool(opts: any?) -> any`

Build a pool. opts: { workers?, mode?, parent?, modulePath? } modulePath: DataModel path of the requireable RoNet root ModuleScript (default "ReplicatedStorage.RoNet").

### Class pool

#### `shutdown()`

Tear the pool down: disconnects the result listener and destroys the worker folder (and its Actor instances).

#### `dispatch(jobs: { any }) -> { any }`

Run a list of jobs on the pool and block until every result is back. Jobs run in shard order across workers; results are returned in the same order as the jobs argument. Only one dispatch may be in flight at a time. In "single" mode this just runs the jobs in-process.

### Class Parallel

#### `attach(model: any, opts: any?) -> any`

One-shot API: attach the op facade to a live model using a default pool. Call handle:forward/loss/backward/scoreGenomes, then :shutdown().

