"""
Dataset utilities for Oxford-IIIT Pet segmentation.
"""

from __future__ import annotations
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import tensorflow_datasets as tfds
except ImportError:
    raise ImportError("Install tensorflow-datasets: pip install tensorflow-datasets tensorflow")
DATA_SEED = 42
def _trimap_to_class_indices(mask: np.ndarray) -> np.ndarray:
    """Convert TFDS trimap (1,2,3) to 0-indexed classes (0,1,2)."""
    return np.clip(mask.astype(np.int64) - 1, 0, 2)

class OxfordPetDataset(Dataset):
    def __init__(
        self,
        split: str,
        target_size: tuple[int, int] = (512, 512),
        augment: bool = False,
        in_memory: bool = True, # Kept for compatibility, but we behave 'lazily' now
    ):
    def __init__(self, split, size=(256, 256), augment=False):
        self.split = split
        self.size = normalize_size(size)
        self.augment = augment

        # Load the dataset metadata/generator
        print(f"Initializing {split} split (Lazy Loading enabled to prevent 'Killed' error)...")

        ds = tfds.load("oxford_iiit_pet", split=split)
        
        # Convert to a list of numpy dictionaries. 
        # Crucially, we store the RAW compressed data, not the expanded 512x512 float32 tensors.
        self._examples = list(tfds.as_numpy(ds)) 
        print(f"Metadata for {len(self._examples)} examples loaded.")
        self.data = list(tfds.as_numpy(ds))

    def __len__(self):
        return len(self.data)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int):
        # We process the image ONLY when requested
        ex = self._examples[idx]

        # 1. Process Image: Resize -> Normalize
        img_raw = Image.fromarray(ex["image"])
        img_pil = img_raw.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        img_np = np.array(img_pil).astype(np.float32) / 255.0

        # 2. Process Mask: Resize -> Map Classes
        mask_raw = ex["segmentation_mask"].squeeze(-1)
        mask_pil = Image.fromarray(mask_raw.astype(np.uint8))
        mask_pil = mask_pil.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        mask_np = _trimap_to_class_indices(np.array(mask_pil))

        # 3. Augmentation (happens on CPU)
        if self.augment:
            img_np, mask_np = _augment_pair(img_np, mask_np)
            img, mask = self.augment_pair(img, mask)

        # 4. Final Conversion to PyTorch Tensors
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() # CHW
        mask_tensor = torch.from_numpy(mask_np).long()

        return img_tensor, mask_tensor

def _augment_pair(img: np.ndarray, mask: np.ndarray):
    import random
    # Note: img is already 0.0-1.0 here
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_mask = Image.fromarray(mask.astype(np.uint8))
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return img, mask

    def augment_pair(self, img, mask):
        # light augmentation for training only
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8))

    if random.random() > 0.5:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        pil_mask = pil_mask.transpose(Image.FLIP_LEFT_RIGHT)
    
    k = random.randint(0, 3)
    if k != 0:
        pil_img = pil_img.rotate(90 * k)
        pil_mask = pil_mask.rotate(90 * k, resample=Image.NEAREST)

    return np.array(pil_img).astype(np.float32) / 255.0, np.array(pil_mask).astype(np.int64)

# Keep other functions (get_fixed_splits, etc.) as they were, 
# just ensure they call the updated OxfordPetDataset.


def get_fixed_splits(
    target_size: int | tuple[int, int] = 512,
    val_fraction: float = 0.1,
):
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    # Initialize the lazy dataset
    train_full = OxfordPetDataset("train", target_size=target_size, augment=True)
    
    n = len(train_full)
    val_len = max(1, int(n * val_fraction))
    train_len = n - val_len
    
    # Deterministic split using the seed
    train_ds, val_ds = torch.utils.data.random_split(
        train_full, 
        [train_len, val_len], 
        generator=torch.Generator().manual_seed(DATA_SEED)
    )
    
    test_ds = OxfordPetDataset("test", target_size=target_size, augment=False)
class SubsetDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


class ImageOnlyDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        return image


def get_split_indices(val_fraction=DEFAULT_VAL_FRACTION, seed=DATA_SEED):
    # one fixed train/val split for all experiments
    full_train = tfds.load("oxford_iiit_pet", split="train")
    n = len(list(tfds.as_numpy(full_train)))

    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_size = max(1, int(round(n * val_fraction)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return train_indices.tolist(), val_indices.tolist()


def get_train_val_test_datasets(
    size=(256, 256),
    val_fraction=DEFAULT_VAL_FRACTION,
    labeled_fraction=1.0,
):
    size = normalize_size(size)
    train_indices, val_indices = get_split_indices(val_fraction=val_fraction)

    # separate train and val base datasets so val does not get train augmentation
    train_base = OxfordPetDataset("train", size=size, augment=True)
    val_base = OxfordPetDataset("train", size=size, augment=False)

    train_ds = SubsetDataset(train_base, train_indices)
    val_ds = SubsetDataset(val_base, val_indices)
    test_ds = OxfordPetDataset("test", size=size, augment=False)

    if labeled_fraction < 1.0:
        # for supervised baselines, only use part of the train masks
        n_labeled = max(1, int(round(len(train_ds) * labeled_fraction)))
        labeled_indices = list(range(n_labeled))
        train_ds = SubsetDataset(train_ds, labeled_indices)

    return train_ds, val_ds, test_ds


def get_train_val_test_loaders(
    target_size: int | tuple[int, int] = 512,
    batch_size: int = 8,
    val_fraction: float = 0.1,
    num_workers: int = 4,
):
    """
    train_loader, val_loader, test_loader. Same split as get_oxford_pet_datasets_for_cross_teaching
    (DATA_SEED). Use target_size=512 for U-Net, 224 for ViT.
    """
    train_ds, val_ds, test_ds = get_fixed_splits(target_size=target_size, val_fraction=val_fraction)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def get_oxford_pet_datasets_for_cross_teaching(
    unet_img_size=512,
    batch_size=8,
    num_workers=0,
    unlabeled_fraction=0.8,
    val_fraction=DEFAULT_VAL_FRACTION,
):
    train_ds, val_ds, test_ds = get_train_val_test_datasets(
        size=(unet_img_size, unet_img_size),
        val_fraction=val_fraction,
        labeled_fraction=1.0,
    )

    n_train = len(train_ds)
    n_unlabeled = max(1, int(round(n_train * unlabeled_fraction)))
    n_labeled = max(1, n_train - n_unlabeled)

    labeled_indices = list(range(n_labeled))
    unlabeled_indices = list(range(n_labeled, n_labeled + n_unlabeled))

    labeled_subset = SubsetDataset(train_ds, labeled_indices)
    unlabeled_subset = SubsetDataset(train_ds, unlabeled_indices)
    unlabeled_subset = ImageOnlyDataset(unlabeled_subset)

    labeled_loader = DataLoader(
        labeled_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    unlabeled_loader = DataLoader(
        unlabeled_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return labeled_loader, unlabeled_loader, val_loader, test_loader