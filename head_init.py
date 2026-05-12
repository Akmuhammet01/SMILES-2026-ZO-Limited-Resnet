"""
head_init.py — Final layer initialization for the 100-class CIFAR100 head.

Rationale
---------
Zero-order optimisation has *much* lower effective signal-to-noise than
first-order training, so the starting point matters more than usual. Two
properties are desirable for the head we hand to the SPSA optimizer:

1. **Symmetry-broken weights.** With identical rows in ``fc.weight`` every
   class would receive the same logit and the SPSA gradient signal would
   collapse for many directions. We therefore use small random weights.

2. **Small initial logit magnitude.** ImageNet features arriving at the head
   have non-negligible norm. A default Kaiming-uniform init produces logits
   with std ≈ 2 → 3, which means cross-entropy starts well above
   ``log(100) ≈ 4.6`` (often 7-9) and the early gradient signal is dominated
   by the random head rather than the (informative) feature similarities.

A Gaussian init with ``std = 1e-2`` gives logits with std ≈ ``0.01 *
sqrt(512) ≈ 0.23``: small enough to keep the initial loss within a couple of
tenths of the uniform-prediction baseline, while still breaking symmetry.
The bias is zeroed (CIFAR100 has uniform class priors over the training
split, so any non-zero bias would be a bad prior).
"""

import torch.nn as nn


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the new 100-class linear head in-place.

    Args:
        layer: The ``nn.Linear(512, 100)`` head appended to ResNet18.
    """
    nn.init.normal_(layer.weight, mean=0.0, std=1e-2)
    nn.init.zeros_(layer.bias)
