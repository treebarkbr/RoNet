# CrossEntropy

Source: `loss/CrossEntropy.luau`

Numerically stable mean cross-entropy. logits: [N, vocab]; targets: N token ids (1-based).  Returns a scalar Tensor (shape {}), ready for backward(). Optional label smoothing.

## Methods

#### `crossEntropy(logits: any, targets: { number }, smooth: number?)`

Mean cross-entropy over a batch of logits. Numerically stable via the log-sum-exp trick; optional label smoothing blends toward a uniform distribution. Returns a scalar tensor ready for backward().

