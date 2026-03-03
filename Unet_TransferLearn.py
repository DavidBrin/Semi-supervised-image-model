import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

# External library for segmentation models (install with: pip install segmentation-models-pytorch)
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Warning: 'segmentation_models_pytorch' not found. Please install it with 'pip install segmentation-models-pytorch'.")
    exit()

# --- 1. Configuration ---
class Config:
    """Config for U-Net (Oxford-IIIT Pet trimap: 3 classes)."""
    image_height = 512
    image_width = 512
    num_classes = 3   # trimap: 0=pet, 1=background, 2=boundary
    batch_size = 4
    epochs = 5
    validation_split = 0.2
    test_split = 0.1
    learning_rate = 1e-4
    encoder_name = "resnet34"
    weights = "imagenet"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
MODEL_SAVE_PATH = os.path.join(CHECKPOINTS_DIR, "unet_oxford_pet.pth")

print(f"Using device: {config.device}")

# --- 2. Loss & metrics ---
def dice_loss(y_pred, y_true, smooth=1e-6):
    """Macro Dice loss (all 3 classes)."""
    if y_true.dim() == 3:
        y_true_one_hot = torch.nn.functional.one_hot(y_true, num_classes=config.num_classes)
        y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2).float()
    else:
        y_true_one_hot = y_true
    y_pred_soft = torch.nn.functional.softmax(y_pred, dim=1)
    dice_per_class = []
    for c in range(config.num_classes):
        pred_c = y_pred_soft[:, c, ...].contiguous().view(-1)
        true_c = y_true_one_hot[:, c, ...].contiguous().view(-1)
        inter = (pred_c * true_c).sum()
        total = pred_c.sum() + true_c.sum()
        dice_per_class.append((2.0 * inter + smooth) / (total + smooth))
    macro_dice = torch.stack(dice_per_class).mean()
    return 1.0 - macro_dice

def dice_coeff(y_pred, y_true, smooth=1e-6):
    """Macro Dice (3-class)."""
    if y_true.dim() == 3:
        y_true_one_hot = torch.nn.functional.one_hot(y_true, num_classes=config.num_classes)
        y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2).float()
    else:
        y_true_one_hot = y_true
    pred_labels = y_pred.argmax(dim=1)
    dice_per_class = []
    for c in range(config.num_classes):
        pred_c = (pred_labels == c).float().view(-1)
        true_c = y_true_one_hot[:, c, ...].contiguous().view(-1)
        inter = (pred_c * true_c).sum()
        total = pred_c.sum() + true_c.sum()
        dice_per_class.append((2.0 * inter + smooth) / (total + smooth))
    return torch.stack(dice_per_class).mean().item()

# --- 3. U-Net (frozen encoder, train decoder) ---
def create_unet_tl():
    """U-Net with pretrained ResNet; encoder frozen, decoder trained."""
    model = smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.weights,
        in_channels=3,   # RGB for Oxford Pet
        classes=config.num_classes,
        activation="softmax",
    )
    print(f"Frozen encoder: {config.encoder_name} from {config.weights}.")
    
    # FREEZE ENCODER WEIGHTS (Transfer Learning)
    # This prevents the pre-trained weights from changing during training,
    # focusing learning on the decoder/segmentation head.
    for param in model.encoder.parameters():
        param.requires_grad = False

    return model

# --- 4. Data (Oxford-IIIT Pet via TFDS) ---

def get_dataset():
    """Load Oxford-IIIT Pet train/val/test via data_oxford_pet."""
    from data_oxford_pet import get_train_val_test_loaders
    train_loader, val_loader, test_loader = get_train_val_test_loaders(
        target_size=config.image_height,
        batch_size=config.batch_size,
        val_fraction=config.validation_split,
        num_workers=0,
    )
    datasets = (train_loader.dataset, val_loader.dataset, test_loader.dataset)
    print("Oxford-IIIT Pet loaded (train/val/test). Trimap: 3 classes.")
    return train_loader, val_loader, test_loader, datasets

# --- 5. Training and Evaluation Loops ---

def train_epoch(model, dataloader, criterion, optimizer):
    """One training epoch."""
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

# --- 6. Main Execution ---

def main():
    """Load data, train U-Net (Oxford Pet trimap), save checkpoint."""
    print("--- U-Net Oxford-IIIT Pet trimap segmentation (transfer learning) ---")
    train_loader, val_loader, test_loader, datasets = get_dataset()
    
    # Create model and set up transfer learning
    model = create_unet_tl()
    model.to(config.device)
    
    # Define Loss, Optimizer, and Metrics
    criterion = dice_loss # Use the custom Dice Loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Print trainable parameters (should mainly be the decoder)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model Parameters: {num_total_params:,}")
    print(f"✨ Trainable Parameters (Decoder/Head only): {num_trainable_params:,} ({(num_trainable_params/num_total_params)*100:.2f}%)")


    # Training Loop
    print(f"\n🚀 Starting training for {config.epochs} epochs...")
    for epoch in range(config.epochs):
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice = evaluate_model(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
              
    print("✅ Training complete.")

    # Evaluation
    print("\n📊 Evaluating model on Test Set...")
    test_loss, test_dice = evaluate_model(model, test_loader, criterion)

    print(f"\n--- Test Results ---")
    print(f"  - Test Loss (Dice Loss): {test_loss:.4f}")
    print(f"  - Test Dice Coeff (F1-score): {test_dice:.4f}")

    # Model Saving
    print("\n💾 Saving model checkpoint...")
    # 1. Ensure the target directory exists
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    # 2. Save the model's state dictionary
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"🎉 PyTorch Model state dictionary successfully saved to: {MODEL_SAVE_PATH}")

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        print("Please ensure you have all required libraries installed, especially 'segmentation-models-pytorch'.")
