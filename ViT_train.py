import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm

from data_oxford_pet import (
    DEFAULT_LABELED_FRACTION,
    DEFAULT_VAL_FRACTION,
    NUM_CLASSES,
    get_train_val_test_loaders,
    set_seed,
)


class Config:
    image_size = 224
    batch_size = 4
    epochs = 8
    learning_rate = 1e-4
    val_fraction = DEFAULT_VAL_FRACTION
    labeled_fraction = DEFAULT_LABELED_FRACTION
    freeze_backbone = True
    num_workers = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
MODEL_SAVE_PATH = os.path.join(CHECKPOINTS_DIR, "vit_oxford_pet.pth")
METRICS_SAVE_PATH = os.path.join(CHECKPOINTS_DIR, "vit_metrics.json")


class ViTSegmentationHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=3):
        super().__init__()

        # upsample ViT patch features back to segmentation map size
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

            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # drop cls token since segmentation only needs patch tokens
        x = x[:, 1:, :]
        b, n, c = x.shape

        # reshape flat patch sequence into 2d feature map
        hw = int(n ** 0.5)
        x = x.transpose(1, 2).reshape(b, c, hw, hw)

        return self.decoder(x)


class ViTSegmentation(nn.Module):
    def __init__(self, num_classes=3, img_size=224, freeze_backbone=True, use_pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=use_pretrained,
            num_classes=0,
            img_size=img_size,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.seg_head = ViTSegmentationHead(
            embed_dim=self.backbone.embed_dim,
            num_classes=num_classes,
        )

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        return self.seg_head(feats)


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


def resize_for_vit(images, masks):
    # ViT backbone uses 224x224 input
    images = F.interpolate(
        images,
        size=(config.image_size, config.image_size),
        mode="bilinear",
        align_corners=False,
    )

    masks = F.interpolate(
        masks.unsqueeze(1).float(),
        size=(config.image_size, config.image_size),
        mode="nearest",
    ).squeeze(1).long()

    return images, masks


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_batches = 0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")

        images, masks = resize_for_vit(images, masks)

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
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")

        images, masks = resize_for_vit(images, masks)

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


def describe_device(device):
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device.index or 0)})"
    if device.type == "mps":
        return "mps"
    return "cpu"


def main():
    set_seed()
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    print(f"[ViT] using device: {describe_device(config.device)}")

    # use same data split setup as U-Net so comparison is fair
    train_loader, val_loader, test_loader = get_train_val_test_loaders(
        target_size=512,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        num_workers=config.num_workers,
        labeled_fraction=config.labeled_fraction,
    )

    model = ViTSegmentation(
        num_classes=NUM_CLASSES,
        img_size=config.image_size,
        freeze_backbone=config.freeze_backbone,
        use_pretrained=True,
    ).to(config.device)

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
            f"[ViT] epoch {epoch + 1}/{config.epochs} "
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
            "model": "ViT",
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
