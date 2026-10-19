import os
import torch
from torch.utils.data import DataLoader

try:
    import torchvision.transforms as transforms
    from torchvision.datasets import Imagenette
except ImportError:
    transforms = None
    Imagenette = None


def get_imagenette_loaders(
    root="./data",
    size="160px",
    batch_size=64,
    num_workers=4,
    input_size=(128, 128),
    norm_mean=(0.485, 0.456, 0.406),
    norm_std=(0.229, 0.224, 0.225),
    download=True,
    seed=0
):
    if transforms is None or Imagenette is None:
        raise ImportError("torchvision is required for get_imagenette_loaders(). Please install it via 'pip install torchvision'.")
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    crop_size = input_size[0] if isinstance(input_size, (tuple, list)) else input_size

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    train_dataset = Imagenette(
        root=root,
        split="train",
        size=size,
        download=download,
        transform=train_transform
    )

    val_dataset = Imagenette(
        root=root,
        split="val",
        size=size,
        download=download,
        transform=val_transform
    )

    has_workers = num_workers > 0
    # persistent workers avoid python multiprocessing teardown between epochs
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=has_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=has_workers
    )

    return train_loader, val_loader
