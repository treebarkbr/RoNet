# Losses

Source: `loss/Losses.luau`

Additional losses built from Tensor primitives (Tensor is passed via deps).  All return 0-dim scalar Tensors ready for backward().    crossEntropy(logits, targets, smooth?)     numeric-stable, mean over rows    logSoftmax(logits) -&gt; [N, V]    klDiv(logitsP, logitsQ)                    mean row-wise KL(P || Q)    jsDiv(logitsP, logitsQ)?                   (identity-aggregated P & Q means)    focalLoss(logits, targets, gamma?, alpha?, smooth?)    binaryCrossEntropy(logits, targets01)      (logits = unscaled scores)

## Methods

#### `logSoftmax(logits: any)`

#### `klDiv(logitsP: any, logitsQ: any)`

#### `jsDiv(logitsP: any, logitsQ: any)`

#### `focalLoss(logits: any, targets: { number }, gamma: number?, alpha: any?)`

#### `binaryCrossEntropy(logits: any, targets: any)`

#### `lmCrossEntropy(logits: any, inputs: any, smooth: number?)`

