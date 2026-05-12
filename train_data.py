"""
train_data.py — Training dataset / DataLoader factory.

The validation runner cycles through ``train_loader`` for exactly
``n_batches`` steps, so we do not need any custom subset or sampler: with
``shuffle=True`` and a seeded ``generator`` we get a deterministic but
well-mixed sequence of mini-batches drawn from the full CIFAR100 training
split, and 32 × 32 = 1024 samples is far smaller than one epoch (50,000),
so each batch is composed of distinct, unrepeated images.

``num_workers=0`` is intentional — the validation runner sets
``torch.use_deterministic_algorithms(True)`` and worker subprocesses on
Windows would otherwise reseed unpredictably and break reproducibility.
"""

from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY = True


def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,
):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=USE_TRAIN_SUBSET_ONLY,  # True → CIFAR100 training split
        download=True,
        transform=get_transforms(train=True),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator_train,
        drop_last=False,
    )

    return train_dataset, train_loader
