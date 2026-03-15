"""
Dataset utilities for Oxford-IIIT Pet segmentation.
"""

from __future__ import annotations

import random
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import tensorflow_datasets as tfds
except ImportError as exc:
    raise ImportError(
        "Install tensorflow-datasets and tensorflow before using this module."
    ) from exc


DATA_SEED = 42
NUM_CLASSES = 3
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_LABELED_FRACTION = 0.20
DEFAULT_UNLABELED_FRACTION = 1.0 - DEFAULT_LABELED_FRACTION


def set_seed(seed: int = DATA_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"Expected int or (height, width), got: {size}")
    return int(size[0]), int(size[1])


def _trimap_to_class_indices(mask: np.ndarray) -> np.ndarray:
    """Convert TFDS trimap labels (1, 2, 3) to class indices (0, 1, 2)."""
    return np.clip(mask.astype(np.int64) - 1, 0, NUM_CLASSES - 1)


def _augment_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    pil_mask = Image.fromarray(mask.astype(np.uint8))

    if random.random() > 0.5:
        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        pil_mask = pil_mask.transpose(Image.FLIP_LEFT_RIGHT)

    k = random.randint(0, 3)
    if k:
        pil_image = pil_image.rotate(90 * k)
        pil_mask = pil_mask.rotate(90 * k, resample=Image.NEAREST)

    aug_image = np.asarray(pil_image, dtype=np.float32) / 255.0
    aug_mask = np.asarray(pil_mask, dtype=np.int64)
    return aug_image, aug_mask


@lru_cache(maxsize=2)
def _load_examples(split: str) -> tuple[dict[str, np.ndarray], ...]:
    ds = tfds.load("oxford_iiit_pet", split=split, shuffle_files=False)
    return tuple(tfds.as_numpy(ds))


class OxfordPetDataset(Dataset):
    def __init__(
        self,
        split: str,
        size: int | tuple[int, int] = 512,
        augment: bool = False,
    ) -> None:
        self.split = split
        self.size = normalize_size(size)
        self.augment = augment
        self._examples = _load_examples(split)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self._examples[idx]
        height, width = self.size

        image = Image.fromarray(example["image"])
        image = image.resize((width, height), Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0

        mask = example["segmentation_mask"].squeeze(-1)
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.resize((width, height), Image.NEAREST)
        mask_np = _trimap_to_class_indices(np.asarray(mask))

        if self.augment:
            image_np, mask_np = _augment_pair(image_np, mask_np)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_np).long()
        return image_tensor, mask_tensor


class SubsetDataset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: list[int] | np.ndarray) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.indices[idx]]


class ImageOnlyDataset(Dataset):
    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, _ = self.base_dataset[idx]
        return image


def get_split_indices(
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DATA_SEED,
) -> tuple[list[int], list[int]]:
    num_examples = len(_load_examples("train"))
    indices = np.arange(num_examples)

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_size = max(1, int(round(num_examples * val_fraction)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices.tolist(), val_indices.tolist()


def get_train_val_test_datasets(
    size: int | tuple[int, int] = 512,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    labeled_fraction: float = 1.0,
) -> tuple[Dataset, Dataset, Dataset]:
    size = normalize_size(size)
    train_indices, val_indices = get_split_indices(val_fraction=val_fraction)

    train_base = OxfordPetDataset("train", size=size, augment=True)
    val_base = OxfordPetDataset("train", size=size, augment=False)
    test_ds = OxfordPetDataset("test", size=size, augment=False)

    if not (0.0 < labeled_fraction <= 1.0):
        raise ValueError(f"labeled_fraction must be in (0, 1], got {labeled_fraction}")

    if labeled_fraction < 1.0:
        num_labeled = max(1, int(round(len(train_indices) * labeled_fraction)))
        train_indices = train_indices[:num_labeled]

    train_ds = SubsetDataset(train_base, train_indices)
    val_ds = SubsetDataset(val_base, val_indices)
    return train_ds, val_ds, test_ds


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )


def get_train_val_test_loaders(
    target_size: int | tuple[int, int] = 512,
    batch_size: int = 8,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    num_workers: int = 0,
    labeled_fraction: float = 1.0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = get_train_val_test_datasets(
        size=target_size,
        val_fraction=val_fraction,
        labeled_fraction=labeled_fraction,
    )

    train_loader = _make_loader(train_ds, batch_size, True, num_workers)
    val_loader = _make_loader(val_ds, batch_size, False, num_workers)
    test_loader = _make_loader(test_ds, batch_size, False, num_workers)
    return train_loader, val_loader, test_loader


def get_oxford_pet_datasets_for_cross_teaching(
    unet_img_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,
    labeled_fraction: float = DEFAULT_LABELED_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    size = normalize_size(unet_img_size)
    train_indices, val_indices = get_split_indices(val_fraction=val_fraction)

    if not (0.0 < labeled_fraction < 1.0):
        raise ValueError(f"labeled_fraction must be in (0, 1) for cross-teaching, got {labeled_fraction}")

    num_labeled = max(1, int(round(len(train_indices) * labeled_fraction)))
    labeled_indices = train_indices[:num_labeled]
    unlabeled_indices = train_indices[num_labeled:]

    if not unlabeled_indices:
        raise ValueError("Cross-teaching requires at least one unlabeled training sample.")

    train_base = OxfordPetDataset("train", size=size, augment=True)
    val_base = OxfordPetDataset("train", size=size, augment=False)
    test_ds = OxfordPetDataset("test", size=size, augment=False)

    labeled_ds = SubsetDataset(train_base, labeled_indices)
    unlabeled_ds = ImageOnlyDataset(SubsetDataset(train_base, unlabeled_indices))
    val_ds = SubsetDataset(val_base, val_indices)

    labeled_loader = _make_loader(labeled_ds, batch_size, True, num_workers)
    unlabeled_loader = _make_loader(unlabeled_ds, batch_size, True, num_workers)
    val_loader = _make_loader(val_ds, batch_size, False, num_workers)
    test_loader = _make_loader(test_ds, batch_size, False, num_workers)
    return labeled_loader, unlabeled_loader, val_loader, test_loader
