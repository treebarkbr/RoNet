# Dropout

Source: `nn/Dropout.luau`

Dropout (inverted dropout: mask = (u &gt; p)/(1-p) while training, identity when eval).

## Methods

#### `new(name: string, p: number?, rng: any?)`

Create a dropout module (p default 0). train()/eval() control whether the mask is applied.

#### `forward(: any, x: any) -> any`

Inverted dropout when training (mask ~ Bernoulli(keep) scaled by 1/keep); identity when p &lt;= 0 or not in train mode.

