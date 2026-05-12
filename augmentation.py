"""
augmentation.py — Data augmentation pipeline for CIFAR100.

Training-pipeline design notes
------------------------------
The SPSA optimizer evaluates the loss several times on the *same* mini-batch
within a single ``.step()``. Any non-deterministic augmentation is applied
once when the batch is loaded (the closure in ``validate.py`` caches the
tensors), so augmentation noise only enters *between* steps, not within
them. This means stronger augmentation does not break SPSA's variance
reduction — it just regularises the head we are learning.

Choices:
* ``RandomCrop(224, padding=16, padding_mode="reflect")`` — translation
  invariance; reflective padding avoids the dark borders that zero-padding
  would introduce after the 32→224 up-scaling.
* ``RandomHorizontalFlip()`` — standard for natural images.
* ``ColorJitter`` (mild) — robustness to lighting / hue shifts.
* ``RandomErasing(p=0.25)`` — occlusion robustness, applied on the tensor.

Heavier policies (AutoAugment, TrivialAugment) were tried but increase
between-batch loss variance enough to slow convergence at the 32-step
budget; they are not used in the final pipeline.

The *validation* pipeline is fixed by the assignment — do not touch it.
"""

import torchvision.transforms as T

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool) -> T.Compose:
    """Return the image transform pipeline for CIFAR100.

    Args:
        train: ``True`` for the training pipeline (with augmentation),
               ``False`` for the fixed validation pipeline.
    """
    if train:
        return T.Compose(
            [
                T.Resize(224),
                T.RandomCrop(224, padding=16, padding_mode="reflect"),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize(224),
                T.ToTensor(),
                T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
            ]
        )
