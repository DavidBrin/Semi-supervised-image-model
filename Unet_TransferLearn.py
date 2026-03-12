import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Please install segmentation-models-pytorch first.")
    raise

from data_oxford_pet import (
    DEFAULT_LABELED_FRACTION,
    DEFAULT_VAL_FRACTION,
    NUM_CLASSES,
    get_train_val_test_loaders,
    set_seed,
)


class Config:
    image_size = 512
    batch_size = 4
    epochs = 8
    learning_rate = 1e-4
    val_fraction = DEFAULT_VAL_FRACTION
    labeled_fraction = DEFAULT_LABELED_FRACTION
    freeze_encoder = True
    num_workers = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
MODEL_SAVE_PATH = os.path.join(CHECKPOINTS_DIR, "unet_oxford_pet.pth")
METRICS_SAVE_PATH = os.path.join(CHECKPOINTS_DIR, "unet_metrics.json")


def create_unet_tl():
    # use pretrained resnet encoder for transfer learning
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )

    if config.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    return model


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)

        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * target_one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)

        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        dice_loss = 1.0 - dice.mean()

        return ce_loss + dice_loss


@torch.no_grad()
def compute_metrics(logits, target, num_classes):
    pred = logits.argmax(dim=1)

    pixel_acc = (pred == target).float().mean().item()

    dice_scores = []
    iou_scores = []

    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        inter = (pred_cls * target_cls).sum().item()
        pred_sum = pred_cls.sum().item()
        target_sum = target_cls.sum().item()
        union = pred_sum + target_sum - inter

        dice = (2.0 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)
        iou = (inter + 1e-6) / (union + 1e-6)

        dice_scores.append(dice)
        iou_scores.append(iou)

    return {
        "dice": float(sum(dice_scores) / len(dice_scores)),
        "iou": float(sum(iou_scores) / len(iou_scores)),
        "pixel_accuracy": float(pixel_acc),
    }


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        metrics = compute_metrics(logits, masks, NUM_CLASSES)

        total_loss += loss.item()
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        total_acc += metrics["pixel_accuracy"]
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "dice": total_dice / max(total_batches, 1),
        "iou": total_iou / max(total_batches, 1),
        "pixel_accuracy": total_acc / max(total_batches, 1),
    }


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    set_seed()
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = get_train_val_test_loaders(
        target_size=config.image_size,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        num_workers=config.num_workers,
        labeled_fraction=config.labeled_fraction,
    )

    model = create_unet_tl().to(config.device)
    criterion = CombinedSegmentationLoss(num_classes=NUM_CLASSES)

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    best_val_dice = -1.0
    history = []

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.device)
        val_metrics = evaluate_model(model, val_loader, criterion, config.device)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
        }
        history.append(row)

        print(
            f"[U-Net] epoch {epoch + 1}/{config.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_dice": best_val_dice,
                },
                MODEL_SAVE_PATH,
            )

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(model, test_loader, criterion, config.device)

    save_json(
        METRICS_SAVE_PATH,
        {
            "model": "U-Net",
            "setting": "supervised baseline",
            "labeled_fraction": config.labeled_fraction,
            "best_val_dice": best_val_dice,
            "history": history,
            "test_metrics": test_metrics,
        },
    )

    print(f"Saved best model to {MODEL_SAVE_PATH}")
    print(f"Saved metrics to {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    main()