# Losses

Source: `loss/Losses.luau`

Additional losses built from Tensor primitives (Tensor is passed via deps).  All return 0-dim scalar Tensors ready for backward().    crossEntropy(logits, targets, smooth?)     numeric-stable, mean over rows    logSoftmax(logits) -&gt; [N, V]    klDiv(logitsP, logitsQ)                    mean row-wise KL(P || Q)    jsDiv(logitsP, logitsQ)?                   (identity-aggregated P & Q means)    focalLoss(logits, targets, gamma?, alpha?, smooth?)    binaryCrossEntropy(logits, targets01)      (logits = unscaled scores)

## Methods

#### `logSoftmax(logits: any)`

log-softmax of [N, V] rows, numerically stable (subtract row max).

#### `klDiv(logitsP: any, logitsQ: any)`

KL divergence, mean over the N rows (rows need not sum to 1).

#### `jsDiv(logitsP: any, logitsQ: any)`

Jensen-Shannon divergence (symmetrized, in [0, 1] loge units to fade apart).

#### `focalLoss(logits: any, targets: { number }, gamma: number?, alpha: any?)`

Focal loss on softmax logits (Lin et al., 2017; gamma default 2). alpha is a per-target weight (multi-class alpha table or scalar); omitted = none.

#### `binaryCrossEntropy(logits: any, targets: any)`

Binary cross-entropy from raw logits (targets in {0, 1}); mean over all elements. Numerically stable via clipping sigmoid probability.

#### `lmCrossEntropy(logits: any, inputs: any, smooth: number?)`

Language-model next-token loss over a batch of sequences. logits: [B, T, V] from a batched model forward; inputs: [B, T] int tensor. Predicts token t+1 from position t for t = 1..T-1 (mean per position). Returns a 0-dim scalar ready for backward().

