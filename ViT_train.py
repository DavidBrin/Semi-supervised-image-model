import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
from torchvision import transforms
from skimage.filters import threshold_otsu
import timm
import logging

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Configuration ---
class Config:
    """Configuration class for ViT training."""
   
    image_height = 224  # ViT works best with 224x224
    image_width = 224
    num_classes = 3   # trimap: pet, background, boundary
    batch_size = 8
    epochs = 30
    validation_split = 0.2
    test_split = 0.1
    learning_rate = 1e-4
    freeze_backbone = True  # Freeze ViT backbone for transfer learning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
_script_dir = Path(__file__).resolve().parent
CHECKPOINTS_DIR = _script_dir / "checkpoints"
MODEL_SAVE_PATH = CHECKPOINTS_DIR / "vit_oxford_pet.pth"

print(f"Using device: {config.device}")

# --- 2. ViT seg head ---
class ViTSegmentationHead(nn.Module):
    """Decoder head on top of ViT features."""
    def __init__(self, embed_dim=768, num_classes=3, img_size=224, patch_size=16):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.img_size = img_size

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, num_patches + 1, embed_dim]
        x = x[:, 1:, :]  # drop CLS token
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        x = self.decoder(x)
        return x


# --- 3. ViT + head ---
class ViTSegmentation(nn.Module):
    """ViT backbone + seg head (timm)."""
    def __init__(
        self,
        num_classes: int = 3,
        img_size: int = 224,
        freeze_backbone: bool = True,
        use_pretrained: bool = True,
    ):
        super().__init__()

        # Load pretrained ViT backbone from timm
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=use_pretrained,
            num_classes=0,      # remove classifier head
            img_size=img_size
        )

        # Freeze backbone for transfer learning
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("ViT backbone frozen for transfer learning")

        # Segmentation head (decoder) - this will be trained
        self.seg_head = ViTSegmentationHead(
            embed_dim=self.backbone.embed_dim,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=self.backbone.patch_embed.patch_size[0]
        )

        logger.info(f"ViT model initialized with pretrained weights: {use_pretrained}")

    def forward(self, x):
        feats = self.backbone.forward_features(x)   # [B, num_patches+1, C]
        seg_map = self.seg_head(feats)
        return seg_map


# --- 4. Data (Oxford-IIIT Pet via TFDS) ---
# --- 5. Metrics and Loss Functions ---
def dice_loss(y_pred, y_true, smooth=1e-6):
    """Macro Dice loss over all classes."""
    y_true_one_hot = F.one_hot(y_true, num_classes=config.num_classes).permute(0, 3, 1, 2).float()
    y_pred_soft = F.softmax(y_pred, dim=1)
    dice_per_class = []
    for c in range(config.num_classes):
        pred_c = y_pred_soft[:, c, ...].contiguous().view(-1)
        true_c = y_true_one_hot[:, c, ...].contiguous().view(-1)
        inter = (pred_c * true_c).sum()
        total = pred_c.sum() + true_c.sum()
        dice_per_class.append((2.0 * inter + smooth) / (total + smooth))
    return 1.0 - torch.stack(dice_per_class).mean()


def dice_coeff(y_pred, y_true, smooth=1e-6):
    """Macro Dice coefficient (multi-class)."""
    y_true_one_hot = F.one_hot(y_true, num_classes=config.num_classes).permute(0, 3, 1, 2).float()
    pred_labels = y_pred.argmax(dim=1)
    dice_per_class = []
    for c in range(config.num_classes):
        pred_c = (pred_labels == c).float().view(-1)
        true_c = y_true_one_hot[:, c, ...].contiguous().view(-1)
        inter = (pred_c * true_c).sum()
        total = pred_c.sum() + true_c.sum()
        dice_per_class.append((2.0 * inter + smooth) / (total + smooth))
    return torch.stack(dice_per_class).mean().item()


# --- 7. Training and Evaluation ---
def train_epoch(model, dataloader, criterion, optimizer):
    """One epoch."""
    model.train()
    total_loss = 0
    total_dice = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_dice += dice_coeff(outputs, targets)

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    return avg_loss, avg_dice


def evaluate_model(model, dataloader, criterion):
    """Eval on val/test."""
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_dice += dice_coeff(outputs, targets)

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    return avg_loss, avg_dice


# --- 8. Main ---
def main():
    """Train ViT on Oxford-IIIT Pet trimap segmentation."""
    print("--- ViT Oxford-IIIT Pet trimap segmentation (transfer learning) ---")
    from data_oxford_pet import get_train_val_test_loaders
    train_loader, val_loader, test_loader = get_train_val_test_loaders(
        target_size=config.image_height,
        batch_size=config.batch_size,
        val_fraction=config.validation_split,
        num_workers=0,
    )
    
    # Create model with transfer learning
    print("\nCreating ViT model with transfer learning...")
    model = ViTSegmentation(
        num_classes=config.num_classes,
        img_size=config.image_height,
        freeze_backbone=config.freeze_backbone,
        use_pretrained=True
    )
    model.to(config.device)
    
    # Define Loss, Optimizer
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Print trainable parameters (should mainly be the segmentation head)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    print(f"\n Model Parameters: {num_total_params:,}")
    print(f"Trainable Parameters (Segmentation Head only): {num_trainable_params:,} "
          f"({(num_trainable_params/num_total_params)*100:.2f}%)")

    # Training Loop
    print(f"\nStarting training for {config.epochs} epochs...")
    best_val_dice = 0.0
    
    for epoch in range(config.epochs):
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice = evaluate_model(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            print(f"  New best validation Dice: {best_val_dice:.4f}")
              
    print(" Training complete.")

    # Evaluation
    print("\n Evaluating model on Test Set...")
    test_loss, test_dice = evaluate_model(model, test_loader, criterion)

    print(f"\n--- Test Results ---")
    print(f"  - Test Loss (Dice Loss): {test_loss:.4f}")
    print(f"  - Test Dice Coeff (F1-score): {test_dice:.4f}")

    # Model Saving
    print("\n Saving model checkpoint...")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_dice': test_dice,
        'test_loss': test_loss,
        'config': {
            'num_classes': config.num_classes,
            'img_size': config.image_height,
            'freeze_backbone': config.freeze_backbone,
        }
    }, MODEL_SAVE_PATH)
    
    print(f" ViT Model successfully saved to: {MODEL_SAVE_PATH}")
    print("\n--- Script Finished ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
