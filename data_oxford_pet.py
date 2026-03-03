# Oxford-IIIT Pet via TFDS. Trimap 1,2,3 -> 0,1,2 (pet, bg, boundary).

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import tensorflow_datasets as tfds
except ImportError:
    raise ImportError("Install tensorflow-datasets: pip install tensorflow-datasets tensorflow")

# Trimap: TFDS uses 1=pet, 2=background, 3=boundary -> we map to 0,1,2
NUM_SEGMENTATION_CLASSES = 3
DATA_SEED = 42  # same train/val/test for all scripts


def _trimap_to_class_indices(mask: np.ndarray) -> np.ndarray:
    """Convert TFDS trimap (1,2,3) to 0-indexed classes (0,1,2)."""
    # TFDS oxford_iiit_pet segmentation_mask: 1=pet, 2=background, 3=boundary
    out = np.clip(mask.astype(np.int64) - 1, 0, 2)
    return out


class OxfordPetDataset(Dataset):
    """
    PyTorch Dataset for Oxford-IIIT Pet (images + trimap masks).
    Loads from TFDS and caches in memory; applies resize and optional augment.
    """

    def __init__(
        self,
        split: str,
        target_size: tuple[int, int] = (512, 512),
        augment: bool = False,
        in_memory: bool = True,
    ):
        """
        split: 'train' or 'test' (TFDS splits).
        target_size: (H, W) for image and mask.
        augment: if True, apply flips/rotation (use only for train).
        in_memory: if True, load all examples into memory in __init__.
        """
        self.split = split
        self.target_size = target_size
        self.augment = augment
        self.in_memory = in_memory

        self._images: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._labels: list[int] = []

        ds = tfds.load("oxford_iiit_pet", split=split)
        if in_memory:
            for ex in tfds.as_numpy(ds):
                img = ex["image"]  # (H, W, 3) uint8
                seg = ex["segmentation_mask"]  # (H, W, 1) uint8
                label = int(ex["label"])
                self._images.append(img)
                # (H,W,1) -> (H,W), then to 0,1,2
                mask_flat = seg.squeeze(-1)
                self._masks.append(_trimap_to_class_indices(mask_flat))
                self._labels.append(label)

    def __len__(self) -> int:
        return len(self._images) if self._images else 0

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image
        pil = Image.fromarray(img)
        pil = pil.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        return np.array(pil)

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        from PIL import Image
        pil = Image.fromarray(mask.astype(np.uint8))
        pil = pil.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        return np.array(pil).astype(np.int64)

    def __getitem__(self, idx: int):
        img = self._images[idx]
        mask = self._masks[idx]

        img = img.astype(np.float32) / 255.0
        img = self._resize_image(img)
        mask = self._resize_mask(mask)

        if self.augment:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            img, mask = _augment_pair(img, mask)

        # NCHW
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        return img, mask


def _augment_pair(img: np.ndarray, mask: np.ndarray):
    """Simple flips + 90deg rotation; keep img (H,W,3) and mask (H,W) in sync."""
    import random
    from PIL import Image

    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_mask = Image.fromarray(mask.astype(np.uint8))

    if random.random() > 0.5:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        pil_mask = pil_mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
        pil_mask = pil_mask.transpose(Image.FLIP_TOP_BOTTOM)
    k = random.randint(0, 3)
    if k != 0:
        pil_img = pil_img.rotate(90 * k, expand=False)
        pil_mask = pil_mask.rotate(90 * k, resample=Image.NEAREST)

    img = np.array(pil_img).astype(np.float32) / 255.0
    mask = np.array(pil_mask).astype(np.int64)
    return img, mask


def get_fixed_splits(
    target_size: int | tuple[int, int] = 512,
    val_fraction: float = 0.1,
):
    """Same train/val/test indices for all scripts (seed=DATA_SEED). Returns (train_ds, val_ds, test_ds)."""
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    train_full = OxfordPetDataset("train", target_size=target_size, augment=True, in_memory=True)
    n = len(train_full)
    val_len = max(1, int(n * val_fraction))
    train_len = n - val_len
    train_ds, val_ds = torch.utils.data.random_split(
        train_full, [train_len, val_len], generator=torch.Generator().manual_seed(DATA_SEED)
    )
    test_ds = OxfordPetDataset("test", target_size=target_size, augment=False, in_memory=True)
    return train_ds, val_ds, test_ds


def get_train_val_test_loaders(
    target_size: int | tuple[int, int] = 512,
    batch_size: int = 8,
    val_fraction: float = 0.1,
    num_workers: int = 0,
):
    """
    train_loader, val_loader, test_loader. Same split as get_oxford_pet_datasets_for_cross_teaching
    (DATA_SEED). Use target_size=512 for U-Net, 224 for ViT.
    """
    train_ds, val_ds, test_ds = get_fixed_splits(target_size=target_size, val_fraction=val_fraction)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    return train_loader, val_loader, test_loader


class _UnlabeledOnlyDataset(Dataset):
    """Wraps a dataset; __getitem__ returns only (image,). Masks are never used in cross-teaching unlabeled step."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return (img,)


def get_oxford_pet_datasets_for_cross_teaching(
    unet_img_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,
    unlabeled_fraction: float = 0.5,
    val_fraction: float = 0.1,
):
    """
    Same train split as get_train_val_test_loaders (DATA_SEED). Labeled = part of train (with masks
    for supervised loss). Unlabeled = rest of train, images only (no masks used for consistency).
    Returns labeled_loader, unlabeled_loader.
    """
    train_ds, val_ds, test_ds = get_fixed_splits(
        target_size=(unet_img_size, unet_img_size), val_fraction=val_fraction
    )
    n_labeled = max(1, len(train_ds) - int(len(train_ds) * unlabeled_fraction))
    n_unlabeled = len(train_ds) - n_labeled
    labeled_subset, unlabeled_subset = torch.utils.data.random_split(
        train_ds, [n_labeled, n_unlabeled], generator=torch.Generator().manual_seed(DATA_SEED)
    )
    labeled_loader = DataLoader(
        labeled_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    unlabeled_only = _UnlabeledOnlyDataset(unlabeled_subset)
    unlabeled_loader = DataLoader(
        unlabeled_only, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    return labeled_loader, unlabeled_loader
