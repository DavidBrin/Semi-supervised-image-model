# Oxford-IIIT Pet via TFDS. Trimap 1,2,3 -> 0,1,2 (pet, bg, boundary).

from __future__ import annotations
import numpy as np
import torch
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
        self.split = split
        self.target_size = target_size
        self.augment = augment

        # Load the dataset metadata/generator
        print(f"Initializing {split} split (Lazy Loading enabled to prevent 'Killed' error)...")
        ds = tfds.load("oxford_iiit_pet", split=split)
        
        # Convert to a list of numpy dictionaries. 
        # Crucially, we store the RAW compressed data, not the expanded 512x512 float32 tensors.
        self._examples = list(tfds.as_numpy(ds)) 
        print(f"Metadata for {len(self._examples)} examples loaded.")

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

        # 4. Final Conversion to PyTorch Tensors
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() # CHW
        mask_tensor = torch.from_numpy(mask_np).long()

        return img_tensor, mask_tensor

def _augment_pair(img: np.ndarray, mask: np.ndarray):
    import random
    # Note: img is already 0.0-1.0 here
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_mask = Image.fromarray(mask.astype(np.uint8))

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
